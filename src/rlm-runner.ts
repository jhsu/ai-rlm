import { generateText } from "ai";
import type { GenerateTextResult, LanguageModel, ModelMessage } from "ai";
import { createLogger } from "./logger.js";
import type { RLMLogLevel, RLMLogger } from "./logger.js";
import {
  extractRequestedFinalVariable,
  resolveExecutionFinalAnswer,
  resolveFinalAnswer,
} from "./rlm-final-answer.js";
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

export interface REPLStep {
  iteration: number;
  reasoning: string;
  code: string;
  output: string;
}

export interface RLMGenerateResult {
  text: string;
  steps: REPLStep[];
  llmCallCount: number;
  iterations: number;
  usage: RLMUsageSummary;
  response: GenerateTextResult<{}, any>;
}

export interface RLMIterationStartEvent {
  iteration: number;
  messages: ModelMessage[];
}

export interface RLMIterationCompleteEvent {
  iteration: number;
  step: REPLStep;
  llmResponse: string;
  executionTimeMs: number;
}

export interface RLMCallEvent {
  prompt?: string;
  messages?: ModelMessage[];
  modelId: string;
  isSubCall: boolean;
}

export interface RLMErrorEvent {
  iteration: number;
  phase: "llm" | "execution" | "parse";
  error: Error;
  context: string;
}

export interface RLMRunSettings {
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

export interface RLMRunParameters {
  context: RLMContext;
  query: string;
  abortSignal?: AbortSignal;
  timeout?: number;
  currentDepth?: number;
  onIterationStart?: (event: RLMIterationStartEvent) => void;
  onIterationComplete?: (event: RLMIterationCompleteEvent) => void;
  onLLMCall?: (event: RLMCallEvent) => void;
  onError?: (event: RLMErrorEvent) => void;
  logger?: RLMLogger;
  logLevel?: RLMLogLevel;
}

export interface RLMSubAgentFactory {
  (settings: RLMSubAgentSettings): { _generate(params: RLMRunParameters): Promise<RLMGenerateResult> };
}

export async function runRLMGenerate({
  settings,
  params,
  createSubAgent,
}: {
  settings: RLMRunSettings;
  params: RLMRunParameters;
  createSubAgent: RLMSubAgentFactory;
}): Promise<RLMGenerateResult> {
  const {
    context,
    query,
    abortSignal,
    timeout,
    currentDepth,
    onIterationStart,
    onIterationComplete,
    onLLMCall,
    onError,
    logger,
    logLevel,
  } = params;
  const startTime = Date.now();
  const internalLogger = createLogger({
    logger: logger ?? settings.logger,
    logLevel: logLevel ?? settings.logLevel,
  });
  const repl = settings.sandboxFactory({
    model: settings.model,
    subModel: settings.subModel,
    maxLLMCalls: settings.maxLLMCalls,
    timeout: timeout ?? 30000,
    maxDepth: settings.maxDepth,
    currentDepth: currentDepth ?? 0,
    maxIterations: settings.maxIterations,
    maxOutputChars: settings.maxOutputChars,
    prepareIteration: settings.prepareIteration,
    prepareSubAgent: settings.prepareSubAgent,
    createSubAgent,
    logger: logger ?? settings.logger,
    logLevel: logLevel ?? settings.logLevel,
    sandboxFactory: settings.sandboxFactory,
  });
  const steps: REPLStep[] = [];
  let mainLLMCallCount = 0;
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
        onError({ iteration: steps.length, phase, error, context: ctx });
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

    for (let iteration = 0; iteration < settings.maxIterations; iteration++) {
      internalLogger.debug(
        "Iteration started",
        logContext({
          iteration: iteration + 1,
          maxIterations: settings.maxIterations,
          depth: currentDepth ?? 0,
        })
      );

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

      let messagesForIteration = messages;
      let modelForIteration = settings.model;
      let maxOutputCharsForIteration = settings.maxOutputChars;

      if (settings.prepareIteration) {
        const prepareResult = await settings.prepareIteration({
          iteration: iteration + 1,
          maxIterations: settings.maxIterations,
          depth: currentDepth ?? 0,
          query,
          messages,
          llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
          maxLLMCalls: settings.maxLLMCalls,
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
              response: createSyntheticGenerateTextResult(
                prepareResult.finalAnswer ?? ""
              ),
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
        if (onLLMCall) {
          try {
            onLLMCall({
              messages: messagesForIteration,
              modelId: modelForIteration.toString(),
              isSubCall: false,
            });
          } catch (e) {
            internalLogger.warn(
              "onLLMCall callback failed",
              logContext({ iteration: iteration + 1, error: e })
            );
          }
        }

        result = await generateText({
          model: modelForIteration,
          messages: messagesForIteration,
          abortSignal,
        });
        mainLLMCallCount++;
        addUsage(rootUsageSummary, usageFromGenerateResult(result));
      } catch (e) {
        const error = e instanceof Error ? e : new Error(String(e));
        await emitError("llm", error, "generateText failed");
        throw error;
      }

      const response = result.text;
      internalLogger.trace(
        "LLM response received",
        logContext({ iteration: iteration + 1, preview: response.substring(0, 500) })
      );

      const codeBlocks = extractCodeBlocks(response);
      const finalVariable = extractRequestedFinalVariable(response);

      if (codeBlocks.length > 0 && codeBlocks[0]) {
        const code = codeBlocks[0];
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
          executionResult = { stdout: "", stderr: error.message, error: error.message };
        }

        const fullOutput = buildExecutionOutput(executionResult);
        const truncatedOutput = truncateOutput(fullOutput, maxOutputCharsForIteration);
        const reasoning = extractReasoning(response);
        const step: REPLStep = {
          iteration: iteration + 1,
          reasoning,
          code,
          output: truncatedOutput,
        };
        steps.push(step);

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

        const executionFinalAnswer = await resolveExecutionFinalAnswer(
          executionResult.result,
          repl
        );
        if (executionFinalAnswer !== undefined) {
          return {
            text: executionFinalAnswer,
            steps,
            llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
            iterations: iteration + 1,
            usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
            response: result,
          };
        }

        const outputMeta = buildOutputMetadata(
          fullOutput,
          truncatedOutput,
          settings.maxHistoryPreview,
          executionResult.error
        );
        messages.push(
          { role: "assistant", content: response },
          { role: "user", content: outputMeta }
        );

        const answer = await resolveFinalAnswer(response, repl);
        if (answer !== undefined) {
          return {
            text: answer,
            steps,
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
                  code: finalVariable ? `FINAL_VAR(${finalVariable})` : "FINAL(...)" ,
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
            steps,
            llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
            iterations: iteration + 1,
            usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
            response: result,
          };
        }

        if (finalVariable) {
          messages.push({ role: "assistant", content: response });
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

    messages.push({
      role: "user",
      content:
        "Maximum iterations reached. Based on all the information gathered, provide your final answer using FINAL(your_answer).",
    });

    const finalResult = await generateText({
      model: settings.model,
      messages,
      abortSignal,
    });
    mainLLMCallCount++;
    addUsage(rootUsageSummary, usageFromGenerateResult(finalResult));

    const finalVariableName = extractRequestedFinalVariable(finalResult.text);
    const answer =
      (await resolveFinalAnswer(finalResult.text, repl)) ??
      (finalVariableName ? `[Variable ${finalVariableName} not found]` : finalResult.text);

    return {
      text: answer,
      steps,
      llmCallCount: mainLLMCallCount + repl.getLLMCallCount(),
      iterations: settings.maxIterations,
      usage: mergeUsage(rootUsageSummary, repl.getUsageSummary()),
      response: finalResult,
    };
  } finally {
    repl.cleanup();
  }
}

function createSyntheticGenerateTextResult(text: string): GenerateTextResult<{}, any> {
  return {
    text,
    content: [{ type: "text" as const, text }],
    reasoning: [],
    reasoningText: undefined,
    files: [],
    sources: [],
    toolCalls: [],
    toolResults: [],
    finishReason: "stop" as const,
    rawFinishReason: "stop",
    usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    totalUsage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    providerMetadata: undefined,
    request: {},
    response: {
      id: "synthetic-response",
      timestamp: new Date(),
      modelId: "synthetic-response",
      messages: [],
    },
    warnings: undefined,
    steps: [],
    output: undefined as any,
    experimental_output: undefined as any,
    staticToolCalls: [],
    dynamicToolCalls: [],
    staticToolResults: [],
    dynamicToolResults: [],
  } as unknown as GenerateTextResult<{}, any>;
}
