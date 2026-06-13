import { generateText } from "ai";
import type { LanguageModel } from "ai";
import {
  addUsage,
  emptyUsageSummary,
  usageFromGenerateResult,
} from "./rlm-utils.js";
import type { RLMLogLevel, RLMLogger } from "./logger.js";
import type {
  MaybePromise,
  PrepareIterationContext,
  PrepareIterationResult,
  PrepareSubAgentContext,
  PrepareSubAgentResult,
  RLMSubAgentSettings,
  RLMContext,
  RLMContextPlanningSettings,
  RLMToolSet,
  RLMUsageSummary,
} from "./rlm-types.js";
import type {
  RLMSandbox,
  RLMSandboxExecutionResult,
  RLMSandboxFactory,
  RLMSandboxFactoryOptions,
} from "./sandbox.js";

type CloudflareSandboxFunction = (...args: unknown[]) => Promise<unknown>;

interface CloudflareSandboxExecuteResult {
  result: unknown;
  error?: string;
  logs?: string[];
}

export interface CloudflareSandboxExecutor {
  execute(
    code: string,
    providersOrFns:
      | Record<string, CloudflareSandboxFunction>
      | Array<{
          name: string;
          fns: Record<string, CloudflareSandboxFunction>;
          positionalArgs?: boolean;
        }>
  ): Promise<CloudflareSandboxExecuteResult>;
}

export interface CloudflareSandboxFactoryOptions
  extends RLMSandboxFactoryOptions {
  executor: CloudflareSandboxExecutor;
}

interface CloudflareREPLEnvironmentOptions
  extends CloudflareSandboxFactoryOptions {
  model: LanguageModel;
  subModel: LanguageModel;
  maxLLMCalls: number;
}

function formatThrownValue(value: unknown): string {
  if (value instanceof Error) {
    return value.message;
  }
  if (typeof value === "string") {
    return value;
  }
  if (value && typeof value === "object") {
    const maybeError = value as { name?: unknown; message?: unknown };
    if (typeof maybeError.message === "string") {
      return typeof maybeError.name === "string"
        ? `${maybeError.name}: ${maybeError.message}`
        : maybeError.message;
    }
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function returnTrailingExpression(code: string): string {
  const lines = code.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]?.trim();
    if (!line || line.startsWith("//")) {
      continue;
    }
    const expression = line.endsWith(";") ? line.slice(0, -1).trimEnd() : line;
    if (
      expression.endsWith("}") ||
      /^(?:return|throw|const|let|var|if|for|while|switch|try|catch|finally|function|class)\b/.test(
        expression
      )
    ) {
      return code;
    }

    const originalLine = lines[i];
    if (originalLine === undefined) {
      continue;
    }
    lines[i] = `${originalLine.slice(0, originalLine.length - line.length)}return (${expression});`;
    return lines.join("\n");
  }
  return code;
}

function removeTrailingFinalExpression(code: string): string {
  const lines = code.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]?.trim();
    if (!line || line.startsWith("//")) {
      continue;
    }
    if (/^FINAL(?:_VAR)?\s*\(/.test(line)) {
      lines.splice(i, 1);
    }
    break;
  }
  return lines.join("\n");
}

function isSimpleVariableReference(name: string): boolean {
  return /^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*|\[["'][^"']+["']\]|\[\d+\])*$/.test(
    name
  );
}

function assertValidToolNames(rlmTools: RLMToolSet | undefined): void {
  for (const name of Object.keys(rlmTools ?? {})) {
    if (!/^[A-Za-z_$][\w$]*$/.test(name)) {
      throw new Error(
        `Invalid RLM tool name "${name}". Tool names must be valid JavaScript identifiers.`
      );
    }
  }
}

class CloudflareREPLEnvironment implements RLMSandbox {
  private executor: CloudflareSandboxExecutor;
  private context: RLMContext | undefined;
  private llmCallCount: number;
  private maxLLMCalls: number;
  private subModel: LanguageModel;
  private model: LanguageModel;
  private contextLoaded: boolean = false;
  private timeout: number;
  private maxDepth: number;
  private currentDepth: number;
  private maxIterations: number;
  private maxOutputChars: number;
  private prepareIteration?: (
    context: PrepareIterationContext
  ) => MaybePromise<PrepareIterationResult | void>;
  private prepareSubAgent?: (
    context: PrepareSubAgentContext
  ) => MaybePromise<PrepareSubAgentResult | void>;
  private usageSummary: RLMUsageSummary;
  private logger?: RLMLogger;
  private logLevel?: RLMLogLevel;
  private sandboxFactory: RLMSandboxFactory;
  private createSubAgent?: RLMSandboxFactoryOptions["createSubAgent"];
  private successfulSnippets: string[] = [];
  private rlmTools?: RLMToolSet;
  private contextPlanning?: RLMContextPlanningSettings;

  constructor(options: CloudflareREPLEnvironmentOptions) {
    const {
      executor,
      model,
      subModel,
      maxLLMCalls,
      timeout = 30000,
      maxDepth = 1,
      currentDepth = 0,
      maxIterations = 20,
      maxOutputChars = 100000,
      prepareIteration,
      prepareSubAgent,
      logger,
      logLevel,
      sandboxFactory = createCloudflareSandbox as RLMSandboxFactory,
      rlmTools,
      contextPlanning,
    } = options;
    assertValidToolNames(rlmTools);
    this.executor = executor;
    this.llmCallCount = 0;
    this.maxLLMCalls = maxLLMCalls;
    this.model = model;
    this.subModel = subModel;
    this.timeout = timeout;
    this.maxDepth = maxDepth;
    this.currentDepth = currentDepth;
    this.maxIterations = maxIterations;
    this.maxOutputChars = maxOutputChars;
    this.prepareIteration = prepareIteration;
    this.prepareSubAgent = prepareSubAgent;
    this.usageSummary = emptyUsageSummary();
    this.logger = logger;
    this.logLevel = logLevel;
    this.sandboxFactory = sandboxFactory;
    this.createSubAgent = options.createSubAgent;
    this.rlmTools = rlmTools;
    this.contextPlanning = contextPlanning;
  }

  async loadContext(context: RLMContext): Promise<void> {
    if (this.contextLoaded) {
      throw new Error("Context already loaded");
    }
    this.context = context;
    this.contextLoaded = true;
  }

  async llmQuery(prompt: string): Promise<string> {
    if (this.llmCallCount >= this.maxLLMCalls) {
      throw new Error(
        `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`
      );
    }

    this.llmCallCount++;

    try {
      const result = await generateText({
        model: this.subModel,
        prompt,
      });
      addUsage(this.usageSummary, usageFromGenerateResult(result));
      return result.text;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`llm_query failed: ${msg}`);
    }
  }

  async subRlmQuery(prompt: string, subContext?: RLMContext): Promise<string> {
    if (this.llmCallCount >= this.maxLLMCalls) {
      throw new Error(
        `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`
      );
    }

    try {
      let nextPrompt = prompt;
      let nextSubContext = subContext;
      let hookSubAgentSettings: Partial<RLMSubAgentSettings> | undefined;

      if (this.prepareSubAgent) {
        const hookResult = await this.prepareSubAgent({
          parentDepth: this.currentDepth,
          nextDepth: this.currentDepth + 1,
          maxDepth: this.maxDepth,
          prompt,
          subContext,
          llmCallCount: this.llmCallCount,
          maxLLMCalls: this.maxLLMCalls,
          usageSoFar: { ...this.usageSummary },
        });

        if (hookResult) {
          if (hookResult.prompt !== undefined) {
            nextPrompt = hookResult.prompt;
          }
          if (hookResult.subContext !== undefined) {
            nextSubContext = hookResult.subContext;
          }
          if (hookResult.subAgentSettings) {
            hookSubAgentSettings = hookResult.subAgentSettings;
          }

          if (hookResult.action === "abort") {
            throw new Error(
              hookResult.reason ?? "prepareSubAgent aborted sub-agent execution"
            );
          }

          if (hookResult.action === "fallback_to_llm_query") {
            return await this.llmQuery(nextPrompt);
          }
        }
      }

      const remainingCalls = this.maxLLMCalls - this.llmCallCount;
      if (remainingCalls <= 0) {
        throw new Error(
          `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`
        );
      }

      const defaultSubAgentSettings: RLMSubAgentSettings = {
        model: this.model,
        subModel: this.subModel,
        maxIterations: Math.max(5, Math.floor(this.maxIterations / 2)),
        maxLLMCalls: Math.max(
          1,
          Math.min(remainingCalls, Math.floor(this.maxLLMCalls / 2))
        ),
        maxOutputChars: this.maxOutputChars,
        maxDepth: this.maxDepth,
        prepareIteration: this.prepareIteration,
        prepareSubAgent: this.prepareSubAgent,
        logger: this.logger,
        logLevel: this.logLevel,
        rlmTools: this.rlmTools,
        contextPlanning: this.contextPlanning,
      };

      if (!this.createSubAgent) {
        return await this.llmQuery(nextPrompt);
      }

      const subAgent = this.createSubAgent({
        ...defaultSubAgentSettings,
        ...hookSubAgentSettings,
      });

      const result = await subAgent._generate({
        context: nextSubContext || "No context provided",
        query: nextPrompt,
        currentDepth: this.currentDepth + 1,
      });

      if (result.llmCallCount > remainingCalls) {
        throw new Error(
          `LLM call limit exceeded by sub-agent: used ${result.llmCallCount}, only ${remainingCalls} remaining`
        );
      }
      this.llmCallCount += result.llmCallCount;
      addUsage(this.usageSummary, result.usage);

      return result.text;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`sub_rlm failed: ${msg}`);
    }
  }

  async executeJavaScript(code: string): Promise<RLMSandboxExecutionResult> {
    if (!this.contextLoaded) {
      return {
        stdout: "",
        stderr: "Cloudflare sandbox context not initialized",
        error: "Cloudflare sandbox context not initialized",
      };
    }

    try {
      const result = await this.executor.execute(
        this.wrapCode(code),
        this.createHostFunctions()
      );
      const stdout = result.logs?.join("\n") ?? "";

      if (result.error) {
        return {
          stdout,
          stderr: result.error,
          error: result.error,
        };
      }

      this.successfulSnippets.push(code);
      return {
        stdout,
        stderr: "",
        result: result.result,
      };
    } catch (error) {
      const errorMessage = formatThrownValue(error);
      return {
        stdout: "",
        stderr: errorMessage,
        error: errorMessage,
      };
    }
  }

  async getVariable(name: string): Promise<unknown> {
    if (!this.contextLoaded) {
      return undefined;
    }
    if (!isSimpleVariableReference(name)) {
      return undefined;
    }

    try {
      const result = await this.executor.execute(
        this.wrapCode(`return ${name};`, false),
        this.createHostFunctions()
      );
      return result.error ? undefined : result.result;
    } catch {
      return undefined;
    }
  }

  getLLMCallCount(): number {
    return this.llmCallCount;
  }

  getUsageSummary(): RLMUsageSummary {
    return { ...this.usageSummary };
  }

  cleanup(): void {
    this.context = undefined;
    this.contextLoaded = false;
    this.successfulSnippets = [];
  }

  private createHostFunctions(): Record<string, CloudflareSandboxFunction> {
    return {
      llm_query: async (prompt) => this.llmQuery(String(prompt)),
      llm_query_batched: async (prompts) => {
        if (
          !Array.isArray(prompts) ||
          !prompts.every((prompt) => typeof prompt === "string")
        ) {
          throw new Error("llm_query_batched expects an array of strings");
        }
        const remaining = this.maxLLMCalls - this.llmCallCount;
        if (prompts.length > remaining) {
          throw new Error(
            `LLM call limit exceeded: batch requires ${prompts.length} calls, only ${remaining} remaining`
          );
        }
        return Promise.all(prompts.map((prompt) => this.llmQuery(prompt)));
      },
      sub_rlm: async (prompt, subContext) =>
        this.currentDepth >= this.maxDepth - 1
          ? this.llmQuery(String(prompt))
          : this.subRlmQuery(String(prompt), subContext as RLMContext | undefined),
      ...Object.fromEntries(
        Object.entries(this.rlmTools ?? {}).map(([name, tool]) => [
          `rlm_tool_${name}`,
          async (input: unknown) => tool.execute(input),
        ])
      ),
    };
  }

  private wrapCode(code: string, includeCurrentResult: boolean = true): string {
    const contextJson = JSON.stringify(this.context);
    const replay = this.successfulSnippets
      .map((snippet) => `\n/* replay */\n${removeTrailingFinalExpression(snippet)}`)
      .join("\n");
    const currentCode = includeCurrentResult ? returnTrailingExpression(code) : code;
    const toolBindings = Object.keys(this.rlmTools ?? {})
      .map((name) => `${JSON.stringify(name)}: (input) => rlm_tool_${name}(input)`)
      .join(", ");
    const current = includeCurrentResult
      ? `return await (async () => {\n${currentCode}\n})();`
      : code;

    return `async () => {
const context = ${contextJson};
const FINAL = (answer) => ({ type: "final", value: answer });
const FINAL_VAR = (varName) => ({ type: "final_var", value: varName });
const tools = {${toolBindings}};
${replay}
${current}
}`;
  }
}

export const createCloudflareSandbox = (
  options: CloudflareSandboxFactoryOptions
): RLMSandbox => new CloudflareREPLEnvironment(options);
