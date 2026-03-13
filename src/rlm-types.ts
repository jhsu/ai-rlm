import type { LanguageModel, ModelMessage } from "ai";
import type { RLMLogLevel, RLMLogger } from "./logger.js";

export type MaybePromise<T> = T | Promise<T>;

export interface RLMUsageSummary {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  reasoningTokens: number;
  cachedInputTokens: number;
}

export type RLMContext = string | string[] | Record<string, unknown>;

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

export interface RLMSubAgentSettings {
  model: LanguageModel;
  subModel: LanguageModel;
  maxIterations: number;
  maxLLMCalls: number;
  maxOutputChars: number;
  maxDepth: number;
  prepareIteration?: (
    context: PrepareIterationContext
  ) => MaybePromise<PrepareIterationResult | void>;
  prepareSubAgent?: (
    context: PrepareSubAgentContext
  ) => MaybePromise<PrepareSubAgentResult | void>;
  logger?: RLMLogger;
  logLevel?: RLMLogLevel;
}

export interface PrepareSubAgentResult {
  action?: "continue" | "fallback_to_llm_query" | "abort";
  reason?: string;
  prompt?: string;
  subContext?: RLMContext;
  subAgentSettings?: Partial<RLMSubAgentSettings>;
}

export interface RLMSubAgentRunParameters {
  context: RLMContext;
  query: string;
  currentDepth?: number;
}

export interface RLMSubAgentRunResult {
  text: string;
  llmCallCount: number;
  usage: RLMUsageSummary;
}

export interface RLMSubAgentRunner {
  _generate(params: RLMSubAgentRunParameters): Promise<RLMSubAgentRunResult>;
}
