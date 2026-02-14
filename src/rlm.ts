/**
 * RLM (Recursive Language Model) - TypeScript Implementation
 *
 * Based on the paper "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
 * Uses the Vercel AI SDK for the implementation.
 *
 * REPL Environment: JavaScript with vm2 or node:vm sandbox
 */

import { generateText } from "ai";
import type { ModelMessage, LanguageModel } from "ai";
import * as vm from "node:vm";

// // Try to import vm2, but don't fail if it's not available (e.g., in Bun)
// let VM: typeof import("vm2").VM | undefined;
// try {
//   const vm2 = await import("vm2");
//   VM = vm2.VM;
// } catch {
//   // vm2 not available, will use node:vm fallback
// VM = undefined;
// }

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Settings for RLMAgent
 */
export interface RLMAgentSettings {
  /** Model for the root agent */
  model: LanguageModel;
  /** Model for sub-LLM queries (defaults to model) */
  subModel?: LanguageModel;
  /** Maximum iterations for the REPL loop (default: 20) */
  maxIterations?: number;
  /** Maximum sub-LLM calls per execution (default: 50) */
  maxLLMCalls?: number;
  /** Maximum characters in REPL output (default: 100000) */
  maxOutputChars?: number;
  /** Enable verbose logging (default: false) */
  verbose?: boolean;
}

/**
 * Parameters for RLMAgent.generate() and RLMAgent.stream()
 */
export interface RLMAgentCallParameters {
  /** The large context, document, or dataset to analyze */
  context: RLMContext;
  /** The specific question or task to perform on the context */
  query: string;
  /** Optional abort signal */
  abortSignal?: AbortSignal;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Callback for each step completion */
  onStepFinish?: (step: REPLStep) => void | Promise<void>;
}

/**
 * A single step in the REPL trajectory
 */
export interface REPLStep {
  iteration: number;
  reasoning: string;
  code: string;
  output: string;
}

/**
 * Result from RLMAgent.generate()
 */
export interface RLMGenerateResult {
  /** The generated answer text */
  text: string;
  /** Array of REPL steps taken during generation */
  steps: REPLStep[];
  /** Total number of LLM calls made */
  llmCallCount: number;
  /** Total iterations performed */
  iterations: number;
}

/**
 * Result from RLMAgent.stream() - extends generate result with streaming capabilities
 */
export interface RLMStreamResult extends RLMGenerateResult {
  /** Readable stream of text chunks */
  textStream: ReadableStream<string>;
}

/**
 * @deprecated Use the new RLMAgent API instead. This interface is kept for backward compatibility.
 */
export interface RLMResult {
  answer: string;
  trajectory: REPLStep[];
  llmCallCount: number;
  iterations: number;
}

/**
 * Context can be a string, array of strings, or structured data
 */
export type RLMContext = string | string[] | Record<string, unknown>;

// ============================================================================
// REPL Environment - Supports both vm2 and node:vm
// ============================================================================

interface Sandbox {
  console: {
    log: (...args: unknown[]) => void;
    error: (...args: unknown[]) => void;
  };
  context: unknown;
  llm_query: (prompt: string) => string;
  llm_query_batched: (prompts: string[]) => string[];
  FINAL: (answer: string) => { type: "final"; value: string };
  FINAL_VAR: (varName: string) => { type: "final_var"; value: string };
}

/**
 * Sandbox environment for executing JavaScript code safely
 * Uses vm2 when available (Node.js), falls back to node:vm (Bun/Node)
 */
class REPLEnvironment {
  // private vm2Instance: import("vm2").VM | undefined;
  private vmContext: vm.Context | undefined;
  private llmCallCount: number;
  private maxLLMCalls: number;
  private subModel: LanguageModel;
  private contextLoaded: boolean = false;
  private consoleOutput: string[] = [];
  // private useVM2: boolean;
  private timeout: number;

  constructor(subModel: LanguageModel, maxLLMCalls: number, timeout = 30000) {
    this.llmCallCount = 0;
    this.maxLLMCalls = maxLLMCalls;
    this.subModel = subModel;
    // this.useVM2 = VM !== undefined;
    this.timeout = timeout;
  }

  /**
   * Load context into the REPL environment
   */
  loadContext(context: RLMContext): void {
    if (this.contextLoaded) {
      throw new Error("Context already loaded");
    }

    let contextData: unknown;

    if (typeof context === "string") {
      contextData = context;
    } else if (Array.isArray(context)) {
      contextData = context.join("\n");
    } else {
      contextData = context;
    }

    // Create sandbox with all necessary functions
    const sandbox: Sandbox = {
      console: {
        log: (...args: unknown[]) => {
          this.consoleOutput.push(args.map((a) => String(a)).join(" "));
        },
        error: (...args: unknown[]) => {
          this.consoleOutput.push(
            "ERROR: " + args.map((a) => String(a)).join(" "),
          );
        },
      },
      context: contextData,
      llm_query: (prompt: string): string => {
        return `<<<LLM_QUERY_START>>>\n${prompt}\n<<<LLM_QUERY_END>>>`;
      },
      llm_query_batched: (prompts: string[]): string[] => {
        return prompts.map(
          (p) => `<<<LLM_QUERY_START>>>\n${p}\n<<<LLM_QUERY_END>>>`,
        );
      },
      FINAL: (answer: string): { type: "final"; value: string } => {
        return { type: "final", value: answer };
      },
      FINAL_VAR: (varName: string): { type: "final_var"; value: string } => {
        return { type: "final_var", value: varName };
      },
    };

    // if (this.useVM2 && VM) {
    //   // Use vm2 (better security, built-in timeout)
    //   this.vm2Instance = new VM({
    //     timeout: this.timeout,
    //     sandbox,
    //   });
    // } else {
    // Fallback to node:vm
    this.vmContext = vm.createContext(sandbox);
    // }

    this.contextLoaded = true;
  }

  /**
   * Query a sub-LLM
   */
  async llmQuery(prompt: string): Promise<string> {
    if (this.llmCallCount >= this.maxLLMCalls) {
      throw new Error(
        `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`,
      );
    }

    this.llmCallCount++;

    const result = await generateText({
      model: this.subModel,
      prompt: prompt,
    });

    return result.text;
  }

  /**
   * Execute JavaScript code in the sandbox with timeout protection
   */
  executeJavaScript(code: string): {
    stdout: string;
    stderr: string;
    error?: string;
    result?: unknown;
  } {
    this.consoleOutput = [];

    try {
      let result: unknown;

      // if (this.useVM2 && this.vm2Instance) {
      //   // Use vm2 (has built-in timeout)
      //   result = this.vm2Instance.run(code);
      // } else if (this.vmContext) {
      // Use node:vm with timeout
      // timeout is specified in runInContext options, not Script constructor
      const script = new vm.Script(code);
      if (this.vmContext) {
        result = script.runInContext(this.vmContext, { timeout: this.timeout });
      }
      // } else {
      //   throw new Error("REPL environment not initialized");
      // }

      const stdout = this.consoleOutput.join("\n");

      return {
        stdout,
        stderr: "",
        result: result !== undefined ? result : undefined,
      };
    } catch (error) {
      if (error instanceof Error) {
        return {
          stdout: this.consoleOutput.join("\n"),
          stderr: error.message,
          error: error.message,
        };
      }
      return {
        stdout: this.consoleOutput.join("\n"),
        stderr: "Unknown error",
        error: "Unknown error",
      };
    }
  }

  /**
   * Get a variable value from the VM
   */
  getVariable(name: string): unknown {
    try {
      if (this.vmContext) {
        // if ...
        //   return this.vm2Instance.run(name);
        // } else if (this.vmContext) {
        const script = new vm.Script(name);
        return script.runInContext(this.vmContext, { timeout: 1000 });
      }
      return undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * Get current LLM call count
   */
  getLLMCallCount(): number {
    return this.llmCallCount;
  }

  /**
   * Clean up
   */
  cleanup(): void {
    this.consoleOutput = [];
    // vm2 and node:vm handle cleanup automatically via garbage collection
    // this.vm2Instance = undefined;
    this.vmContext = undefined;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

function extractCodeBlocks(text: string): string[] {
  const codeBlockRegex = /```(?:javascript|js)?\s*\n([\s\S]*?)\n```/g;
  const blocks: string[] = [];
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    const code = match[1];
    if (code) {
      blocks.push(code.trim());
    }
  }

  return blocks;
}

function extractFinalAnswer(
  text: string,
): { type: "direct" | "variable"; content: string } | null {
  const finalVarMatch = text.match(
    /FINAL_VAR\s*\(\s*["']?([^"')\s]+)["']?\s*\)/i,
  );
  if (finalVarMatch) {
    const content = finalVarMatch[1];
    if (content) {
      return { type: "variable", content };
    }
  }

  const finalMatch = text.match(/FINAL\s*\(\s*["']?([^"')]+)["']?\s*\)/i);
  if (finalMatch) {
    const content = finalMatch[1];
    if (content) {
      return { type: "direct", content };
    }
  }

  return null;
}

// ============================================================================
// System Prompts
// ============================================================================

const RLM_SYSTEM_PROMPT = `You are a Recursive Language Model (RLM) agent. You have access to a JavaScript REPL environment to analyze and process large contexts iteratively.

Your task is to answer queries by:
1. EXPLORING the context through code execution
2. ITERATING with small code snippets to understand the data
3. USING llm_query() for semantic analysis when needed
4. SUBMITTING your final answer when complete

Available in the REPL environment:
- context variable: Contains the input context (loaded as string or object)
- llm_query(prompt): Query a sub-LLM (~500K char capacity) for semantic analysis
- llm_query_batched(prompts): Query multiple prompts concurrently (returns array)
- console.log(): ALWAYS log to see results
- Standard JavaScript: JSON, Array methods, String methods, Math, etc.

IMPORTANT GUIDELINES:
1. EXPLORE FIRST - Look at your data before processing it. Log samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong, reconsider your approach.
4. USE llm_query FOR SEMANTICS - Code finds WHERE things are; llm_query understands WHAT things mean.
5. CHUNK SMARTLY - The sub-LLM can handle ~500K characters. Feed it substantial chunks, not tiny pieces.

When done, provide your final answer using:
- FINAL(your_answer) - to submit directly
- FINAL_VAR(variable_name) - to submit a variable from the REPL

Think step-by-step and show your reasoning before each code block.`;

// ============================================================================
// RLMAgent Class (New API)
// ============================================================================

export class RLMAgent {
  private settings: Required<RLMAgentSettings>;

  constructor(settings: RLMAgentSettings) {
    this.settings = {
      model: settings.model,
      subModel: settings.subModel ?? settings.model,
      maxIterations: settings.maxIterations ?? 20,
      maxLLMCalls: settings.maxLLMCalls ?? 50,
      maxOutputChars: settings.maxOutputChars ?? 100000,
      verbose: settings.verbose ?? false,
    };
  }

  /**
   * Generate an answer by iteratively analyzing the context.
   * This is the primary method for using RLMAgent.
   */
  async generate({
    context,
    query,
    abortSignal,
    timeout,
    onStepFinish,
  }: RLMAgentCallParameters): Promise<RLMGenerateResult> {
    const repl = new REPLEnvironment(
      this.settings.subModel,
      this.settings.maxLLMCalls,
      timeout ?? 30000,
    );
    const steps: REPLStep[] = [];
    let mainLLMCallCount = 0; // Track main agent LLM calls

    try {
      repl.loadContext(context);

      const messages: ModelMessage[] = [
        { role: "system", content: RLM_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Context loaded. Answer the following query: "${query}"`,
        },
      ];

      for (
        let iteration = 0;
        iteration < this.settings.maxIterations;
        iteration++
      ) {
        if (this.settings.verbose) {
          console.log(
            `\n=== Iteration ${iteration + 1}/${this.settings.maxIterations} ===`,
          );
        }

        // Generate next action
        const result = await generateText({
          model: this.settings.model,
          messages,
          abortSignal,
        });
        mainLLMCallCount++; // Track main LLM call

        const response = result.text;

        if (this.settings.verbose) {
          console.log("LLM Response:", response.substring(0, 500));
        }

        // Check for final answer
        const finalAnswer = extractFinalAnswer(response);
        if (finalAnswer && finalAnswer.content) {
          let answer: string;

          if (finalAnswer.type === "direct") {
            answer = finalAnswer.content;
          } else {
            const varValue = repl.getVariable(finalAnswer.content);
            answer =
              varValue !== undefined
                ? String(varValue)
                : `[Variable ${finalAnswer.content} not found]`;
          }

          // Return RLMGenerateResult
          return {
            text: answer,
            steps: steps,
            llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
            iterations: iteration + 1,
          };
        }

        // Execute code
        const codeBlocks = extractCodeBlocks(response);

        if (codeBlocks.length > 0 && codeBlocks[0]) {
          const code: string = codeBlocks[0];
          const executionResult = repl.executeJavaScript(code);

          // Process llm_query calls
          let processedOutput = executionResult.stdout;
          const llmQueryRegex =
            /<<<LLM_QUERY_START>>>\n([\s\S]*?)\n<<<LLM_QUERY_END>>>/g;
          let llmMatch;

          while (
            (llmMatch = llmQueryRegex.exec(executionResult.stdout)) !== null
          ) {
            const prompt = llmMatch[1];
            if (prompt) {
              try {
                const llmResult = await repl.llmQuery(prompt);
                processedOutput = processedOutput.replace(
                  llmMatch[0],
                  `\n[LLM Result]: ${llmResult}\n`,
                );
              } catch (e) {
                const errorMessage = e instanceof Error ? e.message : String(e);
                processedOutput = processedOutput.replace(
                  llmMatch[0],
                  `\n[LLM Error]: ${errorMessage}\n`,
                );
              }
            }
          }

          // Build full output
          let fullOutput = processedOutput;
          if (
            executionResult.result !== undefined &&
            executionResult.result !== null
          ) {
            fullOutput += `\n[Return value]: ${JSON.stringify(executionResult.result)}`;
          }
          if (executionResult.error) {
            fullOutput += `\n[Error]: ${executionResult.error}`;
          }

          // Truncate
          const truncatedOutput =
            fullOutput.length > this.settings.maxOutputChars
              ? fullOutput.substring(0, this.settings.maxOutputChars) +
                "\n...[truncated]"
              : fullOutput;

          // Get reasoning
          const reasoningParts = response.split("```");
          const reasoning =
            reasoningParts.length > 0 ? (reasoningParts[0] ?? "").trim() : "";

          // Create step result
          const step: REPLStep = {
            iteration: iteration + 1,
            reasoning,
            code,
            output: truncatedOutput,
          };

          // Add to steps array
          steps.push(step);

          // Call onStepFinish callback if provided
          if (onStepFinish) {
            await onStepFinish(step);
          }

          // Add to messages
          messages.push(
            { role: "assistant", content: response },
            {
              role: "user",
              content: `Code executed:\n\`\`\`javascript\n${code}\n\`\`\`\n\nOutput:\n${truncatedOutput}\n\nContinue with the next step.`,
            },
          );
        } else {
          messages.push(
            { role: "assistant", content: response },
            {
              role: "user",
              content:
                "Please write JavaScript code in a ```javascript block to explore the context and answer the query.",
            },
          );
        }
      }

      // Max iterations reached
      messages.push({
        role: "user",
        content:
          "Maximum iterations reached. Based on all the information gathered, provide your final answer using FINAL(your_answer).",
      });

      const finalResult = await generateText({
        model: this.settings.model,
        messages,
        abortSignal,
      });
      mainLLMCallCount++; // Track final LLM call

      const finalAnswer = extractFinalAnswer(finalResult.text);
      let answer: string;

      if (finalAnswer && finalAnswer.content) {
        if (finalAnswer.type === "direct") {
          answer = finalAnswer.content;
        } else {
          const varValue = repl.getVariable(finalAnswer.content);
          answer =
            varValue !== undefined
              ? String(varValue)
              : `[Variable ${finalAnswer.content} not found]`;
        }
      } else {
        answer = finalResult.text;
      }

      return {
        text: answer,
        steps: steps,
        llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
        iterations: this.settings.maxIterations,
      };
    } finally {
      repl.cleanup();
    }
  }

  /**
   * Stream the answer generation process.
   * Each step is yielded as it's completed.
   */
  async stream({
    context,
    query,
    abortSignal,
    timeout,
    onStepFinish,
  }: RLMAgentCallParameters): Promise<RLMStreamResult> {
    // For now, delegate to generate() and create a simple stream wrapper
    // Full streaming implementation would require more complex handling
    const result = await this.generate({
      context,
      query,
      abortSignal,
      timeout,
      onStepFinish,
    });

    // Create a simple text stream from the result
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(result.text);
        controller.close();
      },
    });

    return {
      textStream: stream,
      ...result,
    };
  }
}

// ============================================================================
// RLM Class (Deprecated - kept for backward compatibility)
// ============================================================================

/**
 * @deprecated Use RLMAgent instead. This class is kept for backward compatibility.
 * Example:
 *   Before: const rlm = new RLM({...}); const result = await rlm.completion(context, query);
 *   After:  const agent = new RLMAgent({...}); const result = await agent.generate({context, query});
 */
export class RLM {
  private agent: RLMAgent;

  constructor(
    config: {
      model?: LanguageModel;
      subModel?: LanguageModel;
      maxIterations?: number;
      maxLLMCalls?: number;
      maxOutputChars?: number;
      verbose?: boolean;
    } = {},
  ) {
    console.warn("Warning: RLM class is deprecated. Use RLMAgent instead.");
    this.agent = new RLMAgent({
      model: config.model!,
      subModel: config.subModel,
      maxIterations: config.maxIterations,
      maxLLMCalls: config.maxLLMCalls,
      maxOutputChars: config.maxOutputChars,
      verbose: config.verbose,
    });
  }

  /**
   * @deprecated Use RLMAgent.generate() instead
   */
  async completion(context: RLMContext, query: string): Promise<RLMResult> {
    console.warn(
      "Warning: completion() is deprecated. Use generate() instead.",
    );
    const result = await this.agent.generate({ context, query });

    // Convert to old format
    return {
      answer: result.text,
      trajectory: result.steps.map((step: REPLStep, idx: number) => ({
        iteration: step.iteration || idx + 1,
        reasoning: step.reasoning || "",
        code: step.code || "",
        output: step.output || "",
      })),
      llmCallCount: result.llmCallCount,
      iterations: result.iterations,
    };
  }
}

// ============================================================================
// Export
// ============================================================================

export default RLMAgent;
