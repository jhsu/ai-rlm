import type { LanguageModel } from "ai";
import type {
  MaybePromise,
  PrepareIterationContext,
  PrepareIterationResult,
  PrepareSubAgentContext,
  PrepareSubAgentResult,
  RLMSubAgentRunner,
  RLMSubAgentSettings,
  RLMContext,
  RLMUsageSummary,
} from "./rlm-types.js";

export interface RLMSandboxExecutionResult {
  stdout: string;
  stderr: string;
  error?: string;
  result?: unknown;
}

export interface RLMSandbox {
  loadContext(context: RLMContext): Promise<void>;
  executeJavaScript(code: string): Promise<RLMSandboxExecutionResult>;
  getVariable(name: string): unknown;
  getLLMCallCount(): number;
  getUsageSummary(): RLMUsageSummary;
  cleanup(): void;
}

export interface RLMSandboxFactoryOptions {
  model: LanguageModel;
  subModel: LanguageModel;
  maxLLMCalls: number;
  timeout?: number;
  maxDepth?: number;
  currentDepth?: number;
  maxIterations?: number;
  maxOutputChars?: number;
  prepareIteration?: (
    context: PrepareIterationContext
  ) => MaybePromise<PrepareIterationResult | void>;
  prepareSubAgent?: (
    context: PrepareSubAgentContext
  ) => MaybePromise<PrepareSubAgentResult | void>;
  createSubAgent?: (settings: RLMSubAgentSettings) => RLMSubAgentRunner;
  verbose?: boolean;
  sandboxFactory?: RLMSandboxFactory;
}

export type RLMSandboxFactory = (options: RLMSandboxFactoryOptions) => RLMSandbox;
