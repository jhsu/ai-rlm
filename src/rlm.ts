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
import { createQuickJSSandbox } from "./quickjs-sandbox.js";
import { createLogger, resolveLogLevel } from "./logger.js";
import type { RLMLogLevel, RLMLogger } from "./logger.js";
import {
  extractRequestedFinalVariable,
  resolveFinalAnswer,
} from "./rlm-final-answer.js";
import { normalizeGenerateInput } from "./rlm-input.js";
import {
  buildContextMetadata,
  buildExecutionOutput,
  buildOutputMetadata,
  createInitialMessages,
  extractReasoning,
  truncateOutput,
} from "./rlm-run-helpers.js";
import type { RLMSandboxFactory } from "./sandbox.js";
import type {
  MaybePromise,
  PrepareIterationContext,
  PrepareIterationResult,
  PrepareSubAgentContext,
  PrepareSubAgentResult,
  RLMContext,
  RLMSubAgentSettings,
  RLMUsageSummary,
} from "./rlm-types.js";
import {
  RLM_SYSTEM_PROMPT,
  addUsage,
  emptyUsageSummary,
  extractCodeBlocks,
  mergeUsage,
  usageFromGenerateResult,
} from "./rlm-utils.js";

export { createQuickJSSandbox };
export type { RLMLogLevel, RLMLogger } from "./logger.js";
export type {
  RLMSandbox,
  RLMSandboxExecutionResult,
  RLMSandboxFactory,
  RLMSandboxFactoryOptions,
} from "./sandbox.js";
export type {
  MaybePromise,
  PrepareIterationContext,
  PrepareIterationResult,
  PrepareSubAgentContext,
  PrepareSubAgentResult,
  RLMContext,
  RLMSubAgentSettings,
  RLMUsageSummary,
} from "./rlm-types.js";

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
  /** Optional injected logger for library diagnostics */
  logger?: RLMLogger;
  /** Log level for library diagnostics (default: silent) */
  logLevel?: RLMLogLevel;
  /** @deprecated Use logLevel="debug" */
  verbose?: boolean;
  /** Optional sandbox factory for custom code execution environments */
  sandboxFactory?: RLMSandboxFactory;
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
  /** @deprecated Use logLevel="debug" */
  debug?: boolean;
  /** Per-call logger override */
  logger?: RLMLogger;
  /** Per-call log level override */
  logLevel?: RLMLogLevel;
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
  /** @deprecated Use logLevel="debug" */
  debug?: boolean;
  /** Per-call logger override */
  logger?: RLMLogger;
  /** Per-call log level override */
  logLevel?: RLMLogLevel;
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
  logger?: RLMLogger;
  logLevel: RLMLogLevel;
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
      logger: settings.logger,
      logLevel: resolveLogLevel(settings),
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
    const { abortSignal, timeout, options } = params;
    const {
      onIterationStart,
      onIterationComplete,
      onLLMCall,
      onError,
      debug,
      logger,
      logLevel,
    } = options ?? {};
    const { context, query } = normalizeGenerateInput(params);

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
      logger,
      logLevel,
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
    logger,
    logLevel,
  }: RLMAgentCallParameters): Promise<RLMGenerateResult> {
    const startTime = Date.now();
    const internalLogger = createLogger({
      logger: logger ?? this.settings.logger,
      logLevel: logLevel ?? this.settings.logLevel,
      debug,
    });
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
      createSubAgent: (settings: RLMSubAgentSettings) =>
        new RLMAgent({
          ...settings,
          sandboxFactory: this.settings.sandboxFactory,
        }),
      logger: logger ?? this.settings.logger,
      logLevel: logLevel ?? this.settings.logLevel,
      sandboxFactory: this.settings.sandboxFactory,
    });
    const steps: REPLStep[] = [];
    let mainLLMCallCount = 0; // Track main agent LLM calls
    const rootUsageSummary = emptyUsageSummary();

    const logContext = (meta?: Record<string, unknown>) => ({
      elapsedMs: Date.now() - startTime,
      ...meta,
    });

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
          internalLogger.warn("onError callback failed", logContext({ error: e }));
        }
      }
    };

    try {
      await repl.loadContext(context);

      const contextMeta = buildContextMetadata(context);
      const messages: ModelMessage[] = createInitialMessages(
        RLM_SYSTEM_PROMPT,
        contextMeta,
        query
      );

      for (
        let iteration = 0;
        iteration < this.settings.maxIterations;
        iteration++
      ) {
        internalLogger.debug("Iteration started", logContext({
          iteration: iteration + 1,
          maxIterations: this.settings.maxIterations,
          depth: currentDepth ?? 0,
        }));

        // Fire iteration start event
        const iterationStartTime = Date.now();
        if (onIterationStart) {
          try {
            onIterationStart({ iteration: iteration + 1, messages });
          } catch (e) {
            internalLogger.warn(
              "onIterationStart callback failed",
              logContext({ iteration: iteration + 1, error: e })
            );
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
              throw new Error(
                prepareResult.reason ?? "prepareIteration aborted execution"
              );
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
              internalLogger.warn(
                "onLLMCall callback failed",
                logContext({ iteration: iteration + 1, error: e })
              );
            }
          }
        } catch (e) {
          const error = e instanceof Error ? e : new Error(String(e));
          await emitError("llm", error, "generateText failed");
          throw error;
        }

        const response = result.text;

        internalLogger.trace("LLM response received", logContext({
          iteration: iteration + 1,
          preview: response.substring(0, 500),
        }));

        const codeBlocks = extractCodeBlocks(response);
        const finalVariable = extractRequestedFinalVariable(response);

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

          const fullOutput = buildExecutionOutput(executionResult);
          const truncatedOutput = truncateOutput(
            fullOutput,
            maxOutputCharsForIteration
          );
          const reasoning = extractReasoning(response);

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
              internalLogger.warn(
                "onIterationComplete callback failed",
                logContext({ iteration: iteration + 1, error: e })
              );
            }
          }

          const outputMeta = buildOutputMetadata(
            fullOutput,
            truncatedOutput,
            this.settings.maxHistoryPreview,
            executionResult.error
          );

          // Add to messages
          messages.push(
            { role: "assistant", content: response },
            {
              role: "user",
              content: outputMeta,
            }
          );

          const answer = await resolveFinalAnswer(response, repl);
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

          if (finalVariable) {
            messages.push({
              role: "user",
              content: `FINAL_VAR(${finalVariable}) referenced a variable that does not exist in REPL state. Define the variable in a code block first, verify with console.log, then call FINAL_VAR again.`,
            });
          }
        } else {
          const answer = await resolveFinalAnswer(response, repl);
          if (answer !== undefined) {
            if (onIterationComplete) {
              try {
                onIterationComplete({
                  iteration: iteration + 1,
                  step: {
                    iteration: iteration + 1,
                    reasoning: response.substring(0, 200),
                    code: finalVariable
                      ? `FINAL_VAR(${finalVariable})`
                      : "FINAL(...)",
                    output: `Final answer: ${answer}`,
                  },
                  llmResponse: response,
                  executionTimeMs: Date.now() - iterationStartTime,
                });
              } catch (e) {
                internalLogger.warn(
                  "onIterationComplete callback failed for final answer",
                  logContext({ iteration: iteration + 1, error: e })
                );
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

          if (finalVariable) {
            messages.push({
              role: "assistant",
              content: response,
            });
            messages.push({
              role: "user",
              content: `FINAL_VAR(${finalVariable}) referenced a variable that does not exist in REPL state. Define the variable in a code block first, verify with console.log, then call FINAL_VAR again.`,
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

      const finalVariableName = extractRequestedFinalVariable(finalResult.text);
      const answer =
        (await resolveFinalAnswer(finalResult.text, repl)) ??
        (finalVariableName
          ? `[Variable ${finalVariableName} not found]`
          : finalResult.text);

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
