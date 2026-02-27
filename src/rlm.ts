/**
 * RLM (Recursive Language Model) - TypeScript Implementation
 *
 * Based on the paper "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
 * Uses the Vercel AI SDK for the implementation.
 *
 * REPL Environment: JavaScript with QuickJS sandbox (WebAssembly)
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
import {
  createQuickJSSandbox,
  type RLMSandboxFactory,
} from "./sandbox.js";
import {
  RLM_SYSTEM_PROMPT,
  addUsage,
  emptyUsageSummary,
  extractCodeBlocks,
  extractFinalAnswer,
  mergeUsage,
  usageFromGenerateResult,
} from "./rlm-utils.js";

export { createQuickJSSandbox };
export type {
  RLMSandbox,
  RLMSandboxExecutionResult,
  RLMSandboxFactory,
  RLMSandboxFactoryOptions,
} from "./sandbox.js";

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
  /** Optional hook to control/override each iteration before model call */
  prepareIteration?: (
    context: PrepareIterationContext
  ) => MaybePromise<PrepareIterationResult | void>;
  /** Optional hook to control recursive sub-agent behavior */
  prepareSubAgent?: (
    context: PrepareSubAgentContext
  ) => MaybePromise<PrepareSubAgentResult | void>;
  /** Enable verbose logging (default: false) */
  verbose?: boolean;
  /** Optional sandbox factory for custom code execution environments */
  sandboxFactory?: RLMSandboxFactory;
}

type MaybePromise<T> = T | Promise<T>;

export interface RLMUsageSummary {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  reasoningTokens: number;
  cachedInputTokens: number;
}

export interface PrepareIterationContext {
  iteration: number;
  maxIterations: number;
  depth: number;
  query: string;
  messages: ModelMessage[];
  llmCallCount: number;
  maxLLMCalls: number;
  usageSoFar: RLMUsageSummary;
}

export interface PrepareIterationResult {
  action?: "continue" | "finalize" | "abort";
  reason?: string;
  finalAnswer?: string;
  model?: LanguageModel;
  messages?: ModelMessage[];
  maxOutputChars?: number;
}

export interface PrepareSubAgentContext {
  parentDepth: number;
  nextDepth: number;
  maxDepth: number;
  prompt: string;
  subContext?: RLMContext;
  llmCallCount: number;
  maxLLMCalls: number;
  usageSoFar: RLMUsageSummary;
}

export interface PrepareSubAgentResult {
  action?: "continue" | "fallback_to_llm_query" | "abort";
  reason?: string;
  prompt?: string;
  subContext?: RLMContext;
  subAgentSettings?: Partial<RLMAgentSettings>;
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
  /** Internal recursion depth for sub-agents */
  currentDepth?: number;
  /** Called when iteration starts (before LLM call) */
  onIterationStart?: (event: RLMIterationStartEvent) => void;
  /** Called when iteration completes (after code execution) */
  onIterationComplete?: (event: RLMIterationCompleteEvent) => void;
  /** Called when LLM is invoked */
  onLLMCall?: (event: RLMCallEvent) => void;
  /** Called when errors occur */
  onError?: (event: RLMErrorEvent) => void;
  /** Enable debug logging */
  debug?: boolean;
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
  /** Aggregated usage across root and sub-calls */
  usage: RLMUsageSummary;

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
 * Context can be a string, array of strings, or structured data
 */
export type RLMContext = string | string[] | Record<string, unknown>;

/**
 * Event fired when an iteration starts (before LLM is called)
 */
export interface RLMIterationStartEvent {
  iteration: number;
  messages: ModelMessage[];
}

/**
 * Event fired when an iteration completes (after code execution)
 */
export interface RLMIterationCompleteEvent {
  iteration: number;
  step: REPLStep;
  llmResponse: string;
  executionTimeMs: number;
}

/**
 * Event fired when LLM is called
 */
export interface RLMCallEvent {
  prompt?: string;
  messages?: ModelMessage[];
  modelId: string;
  isSubCall: boolean;
}

/**
 * Event fired when execution errors occur
 */
export interface RLMErrorEvent {
  iteration: number;
  phase: "llm" | "execution" | "parse";
  error: Error;
  context: string;
}

type GenerateParams = {
  /** The large context to load into the REPL environment (not passed to the LLM) */
  context?: RLMContext;
  /** Callback when an iteration starts (before LLM call) */
  onIterationStart?: (event: RLMIterationStartEvent) => void;
  /** Callback when an iteration completes (after code execution) */
  onIterationComplete?: (event: RLMIterationCompleteEvent) => void;
  /** Callback when LLM is called */
  onLLMCall?: (event: RLMCallEvent) => void;
  /** Callback when errors occur */
  onError?: (event: RLMErrorEvent) => void;
  /** Enable detailed logging (default: false) */
  debug?: boolean;
};

interface RLMAgentOutput extends Output.Output<RLMGenerateResult, any, any> {}

interface RLMAgentResolvedSettings {
  model: LanguageModel;
  subModel: LanguageModel;
  maxIterations: number;
  maxLLMCalls: number;
  maxOutputChars: number;
  maxHistoryPreview: number;
  maxDepth: number;
  prepareIteration?: (
    context: PrepareIterationContext
  ) => MaybePromise<PrepareIterationResult | void>;
  prepareSubAgent?: (
    context: PrepareSubAgentContext
  ) => MaybePromise<PrepareSubAgentResult | void>;
  verbose: boolean;
  sandboxFactory: RLMSandboxFactory;
}

export class RLMAgent implements Agent<GenerateParams, {}, RLMAgentOutput> {
  readonly version = "agent-v1" as const;
  readonly id: string;
  readonly tools: ToolSet = {};

  private settings: RLMAgentResolvedSettings;

  constructor(settings: RLMAgentSettings) {
    this.settings = {
      model: settings.model,
      subModel: settings.subModel ?? settings.model,
      maxIterations: settings.maxIterations ?? 20,
      maxLLMCalls: settings.maxLLMCalls ?? 50,
      maxOutputChars: settings.maxOutputChars ?? 100000,
      maxHistoryPreview: settings.maxHistoryPreview ?? 500,
      maxDepth: settings.maxDepth ?? 1,
      prepareIteration: settings.prepareIteration,
      prepareSubAgent: settings.prepareSubAgent,
      verbose: settings.verbose ?? false,
      sandboxFactory: settings.sandboxFactory ?? createQuickJSSandbox,
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
    const {
      context: explicitContext,
      onIterationStart,
      onIterationComplete,
      onLLMCall,
      onError,
      debug,
    } = options ?? {};

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
      onIterationStart,
      onIterationComplete,
      onLLMCall,
      onError,
      debug,
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
      currentDepth,
      onIterationStart,
    onIterationComplete,
    onLLMCall,
    onError,
    debug = false,
  }: RLMAgentCallParameters): Promise<RLMGenerateResult> {
    const startTime = Date.now();
    const repl = this.settings.sandboxFactory({
      model: this.settings.model,
      subModel: this.settings.subModel,
      maxLLMCalls: this.settings.maxLLMCalls,
      timeout: timeout ?? 30000,
      maxDepth: this.settings.maxDepth,
      currentDepth: currentDepth ?? 0,
      maxIterations: this.settings.maxIterations,
      maxOutputChars: this.settings.maxOutputChars,
      prepareIteration: this.settings.prepareIteration,
      prepareSubAgent: this.settings.prepareSubAgent,
      verbose: this.settings.verbose,
      sandboxFactory: this.settings.sandboxFactory,
    });
    const steps: REPLStep[] = [];
    let mainLLMCallCount = 0; // Track main agent LLM calls
    const rootUsageSummary = emptyUsageSummary();

    const log = (msg: string, ...args: unknown[]) => {
      if (debug || this.settings.verbose) {
        console.log(`[RLM ${Date.now() - startTime}ms] ${msg}`, ...args);
      }
    };

    const emitError = async (
      phase: RLMErrorEvent["phase"],
      error: Error,
      ctx: string
    ) => {
      if (onError) {
        try {
          onError({
            iteration: steps.length,
            phase,
            error,
            context: ctx,
          });
        } catch (e) {
          log("Error in onError callback:", e);
        }
      }
    };

    try {
      await repl.loadContext(context);

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

        // Fire iteration start event
        const iterationStartTime = Date.now();
        if (onIterationStart) {
          try {
            onIterationStart({ iteration: iteration + 1, messages });
          } catch (e) {
            log("Error in onIterationStart callback:", e);
          }
        }

        // Generate next action
        let messagesForIteration = messages;
        let modelForIteration = this.settings.model;
        let maxOutputCharsForIteration = this.settings.maxOutputChars;

        if (this.settings.prepareIteration) {
          const prepareResult = await this.settings.prepareIteration({
            iteration: iteration + 1,
            maxIterations: this.settings.maxIterations,
            depth: currentDepth ?? 0,
            query,
            messages,
            llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
            maxLLMCalls: this.settings.maxLLMCalls,
            usageSoFar: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
          });

          if (prepareResult) {
            if (prepareResult.messages) {
              messages.length = 0;
              messages.push(...prepareResult.messages);
              messagesForIteration = messages;
            }
            if (prepareResult.model) {
              modelForIteration = prepareResult.model;
            }
            if (prepareResult.maxOutputChars !== undefined) {
              maxOutputCharsForIteration = prepareResult.maxOutputChars;
            }

            if (prepareResult.action === "finalize") {
              return {
                text: prepareResult.finalAnswer ?? "",
                steps,
                llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
                iterations: iteration,
                usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
                response: {
                  text: prepareResult.finalAnswer ?? "",
                  usage: {},
                  totalUsage: {},
                  response: {},
                } as GenerateTextResult<{}, any>,
              };
            }

            if (prepareResult.action === "abort") {
              throw new Error(prepareResult.reason ?? "prepareIteration aborted execution");
            }
          }
        }

        let result;
        try {
          result = await generateText({
            model: modelForIteration,
            messages: messagesForIteration,
            abortSignal,
          });
          mainLLMCallCount++; // Track main LLM call
          addUsage(rootUsageSummary, usageFromGenerateResult(result));

          // Fire LLM call event
          if (onLLMCall) {
            try {
              onLLMCall({
                messages,
                modelId: "rlm-model",
                isSubCall: false,
              });
            } catch (e) {
              log("Error in onLLMCall callback:", e);
            }
          }
        } catch (e) {
          const error = e instanceof Error ? e : new Error(String(e));
          await emitError("llm", error, "generateText failed");
          throw error;
        }

        const response = result.text;

        if (this.settings.verbose) {
          console.log("LLM Response:", response.substring(0, 500));
        }

        const codeBlocks = extractCodeBlocks(response);
        const finalAnswer = extractFinalAnswer(response);

        if (codeBlocks.length > 0 && codeBlocks[0]) {
          const code: string = codeBlocks[0];
          let executionResult;
          try {
            executionResult = await repl.executeJavaScript(code);
          } catch (e) {
            const error = e instanceof Error ? e : new Error(String(e));
            await emitError(
              "execution",
              error,
              `Code execution failed: ${code.substring(0, 100)}`
            );
            // Continue with error result
            executionResult = {
              stdout: "",
              stderr: error.message,
              error: error.message,
            };
          }

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
            fullOutput.length > maxOutputCharsForIteration
              ? fullOutput.substring(0, maxOutputCharsForIteration) +
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

          // Fire iteration complete event
          const iterationDuration = Date.now() - iterationStartTime;
          if (onIterationComplete) {
            try {
              onIterationComplete({
                iteration: iteration + 1,
                step,
                llmResponse: response,
                executionTimeMs: iterationDuration,
              });
            } catch (e) {
              log("Error in onIterationComplete callback:", e);
            }
          }

          // Build constant-size metadata about stdout for LLM history
          // Per Algorithm 1: only Metadata(stdout) is appended, not full output
          const previewLen = this.settings.maxHistoryPreview;
          const outputPreview = truncatedOutput.substring(0, previewLen);
          const hasError = !!executionResult.error;
          const outputMeta = [
            `Output metadata:`,
            `- Length: ${fullOutput.length} characters`,
            `- Preview:\n${outputPreview}${
              fullOutput.length > previewLen ? "\n..." : ""
            }`,
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

          // Finalize after code execution (supports responses that include both code and FINAL/FINAL_VAR)
          if (finalAnswer && finalAnswer.content) {
            let answer: string | undefined;

            if (finalAnswer.type === "direct") {
              answer = finalAnswer.content;
            } else {
              const varValue = repl.getVariable(finalAnswer.content);
              if (varValue !== undefined) {
                answer =
                  typeof varValue === "object"
                    ? JSON.stringify(varValue)
                    : String(varValue);
              }
            }

            if (answer !== undefined) {
              return {
                text: answer,
                steps: steps,
                llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
                iterations: iteration + 1,
                usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
                response: result,
              };
            }

            // Variable requested by FINAL_VAR was not created; continue loop with guidance.
            messages.push({
              role: "user",
              content: `FINAL_VAR(${finalAnswer.content}) referenced a variable that does not exist in REPL state. Define the variable in a code block first, verify with console.log, then call FINAL_VAR again.`,
            });
          }
        } else {
          // Final answer without code block (valid when value was computed in previous iterations)
          if (finalAnswer && finalAnswer.content) {
            let answer: string | undefined;

            if (finalAnswer.type === "direct") {
              answer = finalAnswer.content;
            } else {
              const varValue = repl.getVariable(finalAnswer.content);
              if (varValue !== undefined) {
                answer =
                  typeof varValue === "object"
                    ? JSON.stringify(varValue)
                    : String(varValue);
              }
            }

            if (answer !== undefined) {
              // Fire iteration complete for final answer (no code executed)
              if (onIterationComplete) {
                try {
                  onIterationComplete({
                    iteration: iteration + 1,
                    step: {
                      iteration: iteration + 1,
                      reasoning: response.substring(0, 200),
                      code: `FINAL${
                        finalAnswer.type === "variable" ? "_VAR" : ""
                      }(${finalAnswer.content})`,
                      output: `Final answer: ${answer}`,
                    },
                    llmResponse: response,
                    executionTimeMs: Date.now() - iterationStartTime,
                  });
                } catch (e) {
                  log("Error in onIterationComplete callback for final answer:", e);
                }
              }

              return {
                text: answer,
                steps: steps,
                llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
                iterations: iteration + 1,
                usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
                response: result,
              };
            }

            messages.push({
              role: "assistant",
              content: response,
            });
            messages.push({
              role: "user",
              content: `FINAL_VAR(${finalAnswer.content}) referenced a variable that does not exist in REPL state. Define the variable in a code block first, verify with console.log, then call FINAL_VAR again.`,
            });
            continue;
          }

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
      addUsage(rootUsageSummary, usageFromGenerateResult(finalResult));

      const finalAnswer = extractFinalAnswer(finalResult.text);
      let answer: string;

      if (finalAnswer && finalAnswer.content) {
        if (finalAnswer.type === "direct") {
          answer = finalAnswer.content;
        } else {
          const varValue = repl.getVariable(finalAnswer.content);
          if (varValue !== undefined) {
            answer =
              typeof varValue === "object"
                ? JSON.stringify(varValue)
                : String(varValue);
          } else {
            answer = `[Variable ${finalAnswer.content} not found]`;
          }
        }
      } else {
        answer = finalResult.text;
      }

      return {
        text: answer,
        steps: steps,
        llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
        iterations: this.settings.maxIterations,
        usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
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
