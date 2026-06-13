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

export interface RLMContextPlanningSettings {
  /** Maximum prompt characters to send directly to llm_query before chunking or delegation is preferred. */
  maxDirectLLMQueryChars?: number;
  /** Prompt/context size where sub_rlm should be preferred for complex subtasks when recursion is available. */
  preferSubRLMChars?: number;
  /** Recommended chunk size when generated code needs to split large text. */
  chunkSizeChars?: number;
  /** Recommended overlap between adjacent chunks when splitting large text. */
  chunkOverlapChars?: number;
  /** Maximum nested object/array depth included in first-iteration context metadata. */
  metadataMaxDepth?: number;
  /** Maximum number of nested metadata entries included in the prompt. */
  metadataMaxEntries?: number;
  /** Maximum preview characters shown per string field in metadata. */
  metadataMaxPreviewChars?: number;
}

export interface RLMToolDescriptor {
  description: string;
  inputSchema?: unknown;
  execute: (input: unknown) => MaybePromise<unknown>;
}

export type RLMToolSet = Record<string, RLMToolDescriptor>;

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
  rlmTools?: RLMToolSet;
  contextPlanning?: RLMContextPlanningSettings;
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
