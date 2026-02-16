/**
 * RLM (Recursive Language Model) - TypeScript Implementation
 *
 * Based on the paper "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
 * Uses the Vercel AI SDK for the implementation.
 *
 * REPL Environment: JavaScript with node:vm sandbox
 */

import { generateText } from "ai";
import type {
  ModelMessage,
  LanguageModel,
  ToolSet,
  Agent,
  Output,
  AgentCallParameters,
  AgentStreamParameters,
  GenerateTextResult,
  StreamTextResult,
} from "ai";
import * as vm from "node:vm";

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
  /** Maximum characters of stdout preview appended to LLM history per iteration (default: 500) */
  maxHistoryPreview?: number;
  /** Maximum recursion depth for sub_rlm calls (default: 1, meaning sub-calls are direct LLM calls) */
  maxDepth?: number;
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
  onStepResult?: (step: REPLStep) => Promise<void>;
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

  response: GenerateTextResult<{}, any>;
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

interface Sandbox {
  console: {
    log: (...args: unknown[]) => void;
    error: (...args: unknown[]) => void;
  };
  context: unknown;
  llm_query: (prompt: string) => Promise<string>;
  llm_query_batched: (prompts: string[]) => Promise<string[]>;
  sub_rlm: (prompt: string, subContext?: RLMContext) => Promise<string>;
  FINAL: (answer: string) => { type: "final"; value: string };
  FINAL_VAR: (varName: string) => { type: "final_var"; value: string };
}

interface REPLEnvironmentOptions {
  model: LanguageModel;
  subModel: LanguageModel;
  maxLLMCalls: number;
  timeout?: number;
  maxDepth?: number;
  currentDepth?: number;
  maxIterations?: number;
  maxOutputChars?: number;
  verbose?: boolean;
}

/**
 * Sandbox environment for executing JavaScript code safely
 * Uses node:vm (Bun/Node)
 */
class REPLEnvironment {
  private vmContext: vm.Context | undefined;
  private llmCallCount: number;
  private maxLLMCalls: number;
  private subModel: LanguageModel;
  private model: LanguageModel;
  private contextLoaded: boolean = false;
  private consoleOutput: string[] = [];
  private timeout: number;
  private maxDepth: number;
  private currentDepth: number;
  private maxIterations: number;
  private maxOutputChars: number;
  private verbose: boolean;

  constructor(options: REPLEnvironmentOptions) {
    const {
      model,
      subModel,
      maxLLMCalls,
      timeout = 30000,
      maxDepth = 1,
      currentDepth = 0,
      maxIterations = 20,
      maxOutputChars = 100000,
      verbose = false,
    } = options;
    this.llmCallCount = 0;
    this.maxLLMCalls = maxLLMCalls;
    this.model = model;
    this.subModel = subModel;
    this.timeout = timeout;
    this.maxDepth = maxDepth;
    this.currentDepth = currentDepth;
    this.maxIterations = maxIterations;
    this.maxOutputChars = maxOutputChars;
    this.verbose = verbose;
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
      // Keep arrays as arrays so runtime type matches metadata and prompt guidance
      contextData = context;
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
            "ERROR: " + args.map((a) => String(a)).join(" ")
          );
        },
      },
      context: contextData,
      llm_query: async (prompt: string): Promise<string> => {
        // Direct LLM call - returns actual result, not placeholder
        return this.llmQuery(prompt);
      },
      llm_query_batched: async (prompts: string[]): Promise<string[]> => {
        // Execute all prompts in parallel and return actual results
        return Promise.all(prompts.map((p) => this.llmQuery(p)));
      },
      sub_rlm: async (
        prompt: string,
        subContext?: RLMContext
      ): Promise<string> => {
        // If at max depth, fall back to simple llm_query (not recursive RLM)
        if (this.currentDepth >= this.maxDepth - 1) {
          return this.llmQuery(prompt);
        }
        // Full recursive RLM call with its own REPL and iteration loop
        return this.subRlmQuery(prompt, subContext);
      },
      FINAL: (answer: string): { type: "final"; value: string } => {
        return { type: "final", value: answer };
      },
      FINAL_VAR: (varName: string): { type: "final_var"; value: string } => {
        return { type: "final_var", value: varName };
      },
    };

    this.vmContext = vm.createContext(sandbox);
    this.contextLoaded = true;
  }

  /**
   * Query a sub-LLM
   */
  async llmQuery(prompt: string): Promise<string> {
    if (this.llmCallCount >= this.maxLLMCalls) {
      throw new Error(
        `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`
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
   * Query a recursive sub-RLM agent
   */
  async subRlmQuery(prompt: string, subContext?: RLMContext): Promise<string> {
    if (this.llmCallCount >= this.maxLLMCalls) {
      throw new Error(
        `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`
      );
    }

    // Create a sub-RLM agent with decreased depth
    const subAgent = new RLMAgent({
      model: this.model,
      subModel: this.subModel,
      maxIterations: Math.max(5, Math.floor(this.maxIterations / 2)),
      maxLLMCalls: Math.max(10, Math.floor(this.maxLLMCalls / 2)),
      maxOutputChars: this.maxOutputChars,
      maxDepth: this.maxDepth - 1,
      verbose: this.verbose,
    });

    // The sub-agent inherits our call count budget
    const result = await subAgent._generate({
      context: subContext || "No context provided",
      query: prompt,
    });

    // Track calls used by sub-agent
    this.llmCallCount += result.llmCallCount;

    return result.text;
  }

  /**
   * Execute JavaScript code in the sandbox with timeout protection
   * Supports top-level await by wrapping code in async IIFE when needed
   */
  async executeJavaScript(code: string): Promise<{
    stdout: string;
    stderr: string;
    error?: string;
    result?: unknown;
  }> {
    this.consoleOutput = [];
    if (!this.vmContext) {
      return {
        stdout: "",
        stderr: "",
        error: "VM context not initialized",
      };
    }

    try {
      let result: unknown;

      // Check if code contains top-level await
      const hasTopLevelAwait =
        /^\s*await\b/m.test(code) || /\bawait\s+/.test(code);

      if (hasTopLevelAwait) {
        // Wrap in async IIFE to support top-level await
        const wrappedCode = `(async () => {\n${code}\n})()`;
        const script = new vm.Script(wrappedCode);

        const promise = script.runInContext(this.vmContext, {
          timeout: this.timeout,
        }) as Promise<unknown>;
        result = await promise;
      } else {
        const script = new vm.Script(code);
        result = script.runInContext(this.vmContext, {
          timeout: this.timeout,
        });
      }

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
    this.vmContext = undefined;
  }
}

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
  text: string
): { type: "direct" | "variable"; content: string } | null {
  const finalVarMatch = text.match(
    /FINAL_VAR\s*\(\s*["']?([^"')\s]+)["']?\s*\)/i
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
- context variable: Contains the input context (loaded as string, array, or object)
- await llm_query(prompt): Async function to query a sub-LLM (~500K char capacity) for semantic analysis. Returns the actual LLM response string.
- await llm_query_batched(prompts[]): Async function to query multiple prompts in parallel. Returns array of response strings.
- await sub_rlm(prompt, subContext?): Async function to launch a recursive sub-RLM agent with its own REPL environment for complex sub-tasks. Returns the final answer string.
- console.log(): ALWAYS log to see results
- Standard JavaScript: JSON, Array methods, String methods, Math, etc.

ASYNC FUNCTION USAGE:
These functions return Promises and must be awaited:
- const sentiment = await llm_query("Analyze sentiment of: " + text);
- const results = await llm_query_batched(["prompt1", "prompt2", "prompt3"]);
- const subAnswer = await sub_rlm("Complex sub-task", partialData);

Note: The context variable persists between iterations. Variables you create remain available.

IMPORTANT GUIDELINES:
1. EXPLORE FIRST - Look at your data before processing it. Log samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. STORE RESULTS IN VARIABLES - You will only see a short preview of each execution's output. Always assign important results to variables so you can access them in later iterations.
4. VERIFY BEFORE SUBMITTING - If results seem wrong, reconsider your approach.
5. USE llm_query FOR SEMANTICS - Code finds WHERE things are; llm_query understands WHAT things mean.
6. CHUNK SMARTLY - The sub-LLM can handle ~500K characters. Feed it substantial chunks, not tiny pieces.

EFFICIENCY RULES:
1. ONE CODE BLOCK PER ITERATION - Do not emit multiple code blocks in one response.
2. INCREMENTAL CHANGES ONLY - Reuse existing variables and functions; avoid re-running full scripts each step.
3. LOG BRIEFLY - Never print full context or large objects. Prefer concise summaries (counts, keys, first 3 items, short previews).
4. DEBUG MINIMALLY - If an error occurs, inspect the specific failing line/variable and patch the smallest possible part.
5. FINALIZE EARLY - As soon as you have the answer in a variable, return it with FINAL_VAR(variable_name) immediately.

OUTPUT VISIBILITY: After each code execution, you will only see a short preview of the output (first ~500 characters) and its total length. The full output exists in the REPL but is NOT included in your conversation history. To retain information across iterations:
- Store results in variables: \`const results = ...\`
- Use console.log() for short summaries only
- Access previously stored variables in later iterations

When done, provide your final answer using:
- FINAL(your_answer) - to submit directly
- FINAL_VAR(variable_name) - to submit a variable from the REPL

Think step-by-step and show your reasoning before each code block.`;

type GenerateParams = {
  /** The large context to load into the REPL environment (not passed to the LLM) */
  context?: RLMContext;
  /** Callback for each step completion */
  onStepResult?: (step: REPLStep) => Promise<void>;
};

interface RLMAgentOutput extends Output.Output<RLMGenerateResult, any, any> {}

export class RLMAgent implements Agent<GenerateParams, {}, RLMAgentOutput> {
  readonly version = "agent-v1" as const;
  readonly id: string;
  readonly tools: ToolSet = {};

  private settings: Required<RLMAgentSettings>;

  constructor(settings: RLMAgentSettings) {
    this.settings = {
      model: settings.model,
      subModel: settings.subModel ?? settings.model,
      maxIterations: settings.maxIterations ?? 20,
      maxLLMCalls: settings.maxLLMCalls ?? 50,
      maxOutputChars: settings.maxOutputChars ?? 100000,
      maxHistoryPreview: settings.maxHistoryPreview ?? 500,
      maxDepth: settings.maxDepth ?? 1,
      verbose: settings.verbose ?? false,
    };
    this.id = "rlm-agent";
  }

  /**
   * Generate an answer by iteratively analyzing the context.
   * This is the primary method for using RLMAgent.
   * Implements the Agent interface with proper typing.
   */
  async generate(
    params: AgentCallParameters<GenerateParams, {}>
  ): Promise<GenerateTextResult<{}, RLMAgentOutput>> {
    // Extract parameters from Agent interface
    const { prompt, messages, abortSignal, timeout, options } = params;
    const { context: explicitContext, onStepResult } = options ?? {};

    // Determine context and query
    // Priority: explicit context param > system message content > fallback
    let context: RLMContext;
    let query: string;

    if (explicitContext) {
      // Explicit context provided via options — use prompt/last user message as query
      context = explicitContext;
      if (typeof prompt === "string") {
        query = prompt;
      } else if (messages && messages.length > 0) {
        const lastUserMsg = messages
          .filter((m: ModelMessage) => m.role === "user")
          .pop();
        query =
          lastUserMsg && typeof lastUserMsg.content === "string"
            ? lastUserMsg.content
            : "Please provide a query.";
      } else {
        query = "Please provide a query.";
      }
    } else if (messages && messages.length > 0) {
      // No explicit context — extract from messages
      // System messages become context, last user message becomes query
      const systemMsgs = messages.filter(
        (m: ModelMessage) => m.role === "system"
      );
      if (systemMsgs.length > 0) {
        context = systemMsgs
          .map((m: ModelMessage) =>
            typeof m.content === "string" ? m.content : "[complex content]"
          )
          .join("\n");
      } else {
        context = "No context provided. Answer based on the query.";
      }
      const lastUserMsg = messages
        .filter((m: ModelMessage) => m.role === "user")
        .pop();
      query =
        lastUserMsg && typeof lastUserMsg.content === "string"
          ? lastUserMsg.content
          : "Please provide a query.";
    } else {
      // Prompt-only fallback
      context = "No context provided. Answer based on the query.";
      query = typeof prompt === "string" ? prompt : "Please provide a query.";
    }

    // Call the internal implementation
    const result = await this._generate({
      context,
      query,
      abortSignal,
      timeout: typeof timeout === "number" ? timeout : undefined,
      onStepResult,
    });

    // Return a proper GenerateTextResult that matches the Agent interface
    // Using type assertion with 'as unknown as' to bypass strict type checking
    // while ensuring runtime compatibility
    return {
      text: result.text,
      content: [{ type: "text" as const, text: result.text }],
      reasoning: [],
      reasoningText: undefined,
      files: [],
      sources: [],
      toolCalls: [],
      toolResults: [],
      finishReason: "stop" as const,
      rawFinishReason: "stop",
      usage: result.response.usage,
      totalUsage: result.response.totalUsage,
      providerMetadata: undefined,
      request: {},
      response: result.response.response,
      warnings: undefined,
      steps: [],
      output: result,
      experimental_output: result,
      // Add missing required properties for GenerateTextResult
      staticToolCalls: [],
      dynamicToolCalls: [],
      staticToolResults: [],
      dynamicToolResults: [],
      // usage: {
      //   inputTokens: result.llmCallCount * 1000,
      //   outputTokens: Math.ceil(result.text.length / 4),
      //   outpuTokenDetails:
      //   totalTokens:
      //     result.llmCallCount * 1000 + Math.ceil(result.text.length / 4),
      // },
      // totalUsage: {
      //   promptTokens: result.llmCallCount * 1000,
      //   completionTokens: Math.ceil(result.text.length / 4),
      //   totalTokens:
      //     result.llmCallCount * 1000 + Math.ceil(result.text.length / 4),
      // },
      // [Symbol.asyncIterator]: async function* () {
      //   yield { type: "text", text: result.text };
      // },
    };
  }

  /**
   * Internal generate implementation
   * Public to allow recursive sub-RLM calls
   */
  async _generate({
    context,
    query,
    abortSignal,
    timeout,
    onStepResult,
  }: RLMAgentCallParameters): Promise<RLMGenerateResult> {
    const repl = new REPLEnvironment({
      model: this.settings.model,
      subModel: this.settings.subModel,
      maxLLMCalls: this.settings.maxLLMCalls,
      timeout: timeout ?? 30000,
      maxDepth: this.settings.maxDepth,
      currentDepth: 0,
      maxIterations: this.settings.maxIterations,
      maxOutputChars: this.settings.maxOutputChars,
      verbose: this.settings.verbose,
    });
    const steps: REPLStep[] = [];
    let mainLLMCallCount = 0; // Track main agent LLM calls

    try {
      repl.loadContext(context);

      // Build metadata about the context (as per Algorithm 1 in paper)
      let contextMeta: string;
      if (typeof context === "string") {
        const preview = context.substring(0, 200);
        contextMeta = `Type: string\nLength: ${
          context.length
        } characters\nPreview: "${preview}${
          context.length > 200 ? "..." : ""
        }"\nAccess: Use the 'context' variable to read data. Use string methods like context.substring(), context.indexOf(), context.split(), etc.`;
      } else if (Array.isArray(context)) {
        const preview = context.slice(0, 3).join("\n");
        contextMeta = `Type: array\nLength: ${
          context.length
        } items\nPreview: [\n${preview}${
          context.length > 3 ? "\n..." : ""
        }\n]\nAccess: Use the 'context' variable. Access items with context[index], iterate with context.forEach() or for...of.`;
      } else {
        const keys = Object.keys(context);
        const preview = keys.slice(0, 5).join(", ");
        contextMeta = `Type: object\nKeys: ${keys.length} (${preview}${
          keys.length > 5 ? ", ..." : ""
        })\nAccess: Use the 'context' variable. Access properties with context.property or context["key"].`;
      }

      const messages: ModelMessage[] = [
        { role: "system", content: RLM_SYSTEM_PROMPT },
        {
          role: "user",
          content: `The input context has been loaded into the REPL environment as a variable named 'context'.\n\nContext metadata:\n${contextMeta}\n\nYour task: ${query}\n\nBegin by exploring the context to understand its structure, then write JavaScript code to analyze it and answer the query.`,
        },
      ];

      for (
        let iteration = 0;
        iteration < this.settings.maxIterations;
        iteration++
      ) {
        if (this.settings.verbose) {
          console.log(
            `\n=== Iteration ${iteration + 1}/${
              this.settings.maxIterations
            } ===`
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
            response: result,
          };
        }

        // Execute code
        const codeBlocks = extractCodeBlocks(response);

        if (codeBlocks.length > 0 && codeBlocks[0]) {
          const code: string = codeBlocks[0];
          const executionResult = await repl.executeJavaScript(code);

          // Build full output (llm_query and sub_rlm now return values directly)
          let fullOutput = executionResult.stdout;
          if (
            executionResult.result !== undefined &&
            executionResult.result !== null
          ) {
            fullOutput += `\n[Return value]: ${JSON.stringify(
              executionResult.result
            )}`;
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

          // Build constant-size metadata about stdout for LLM history
          // Per Algorithm 1: only Metadata(stdout) is appended, not full output
          const previewLen = this.settings.maxHistoryPreview;
          const outputPreview = truncatedOutput.substring(0, previewLen);
          const hasError = !!executionResult.error;
          const outputMeta = [
            `Output metadata:`,
            `- Length: ${fullOutput.length} characters`,
            `- Preview:\n${outputPreview}${fullOutput.length > previewLen ? "\n..." : ""}`,
            hasError ? `- Error: ${executionResult.error}` : `- Errors: none`,
            `\nFull output is stored in the REPL environment. Use variables to access computed results. Continue with the next step.`,
          ].join("\n");

          // Add to messages
          messages.push(
            { role: "assistant", content: response },
            {
              role: "user",
              content: outputMeta,
            }
          );

          // Call onStepFinish callback if provided
          if (onStepResult) {
            try {
              await onStepResult(step);
            } catch (error) {
              console.error("Error in onStepFinish callback:", error);
            }
          }
        } else {
          messages.push(
            { role: "assistant", content: response },
            {
              role: "user",
              content:
                "Please write JavaScript code in a ```javascript block to explore the context and answer the query.",
            }
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
        response: finalResult,
      };
    } finally {
      repl.cleanup();
    }
  }

  /**
   * Stream the answer generation process.
   * Each step is yielded as it's completed.
   */
  async stream(
    options: AgentStreamParameters<GenerateParams, {}>
  ): Promise<StreamTextResult<{}, RLMAgentOutput>> {
    // For now, delegate to generate() and create a simple stream wrapper
    const result = await this.generate(options as any);

    // Create a simple text stream from the result
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(result.text);
        controller.close();
      },
    });

    return {
      textStream: stream,
      text: result.text,
      content: result.content,
      reasoning: result.reasoning,
      reasoningText: result.reasoningText,
      files: result.files,
      sources: result.sources,
      toolCalls: result.toolCalls,
      toolResults: result.toolResults,
      finishReason: result.finishReason,
      rawFinishReason: result.rawFinishReason,
      usage: {},
      providerMetadata: result.providerMetadata,
      request: result.request,
      response: result.response,
      warnings: result.warnings,
    } as unknown as StreamTextResult<{}, RLMAgentOutput>;
  }
}

export default RLMAgent;
