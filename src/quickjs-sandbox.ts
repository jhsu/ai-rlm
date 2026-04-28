import { generateText } from "ai";
import type { LanguageModel } from "ai";
import {
  newQuickJSAsyncWASMModuleFromVariant,
  shouldInterruptAfterDeadline,
  type QuickJSAsyncContext,
  type QuickJSAsyncRuntime,
} from "quickjs-emscripten-core";
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
  RLMUsageSummary,
} from "./rlm-types.js";
import type {
  RLMSandbox,
  RLMSandboxExecutionResult,
  RLMSandboxFactory,
  RLMSandboxFactoryOptions,
} from "./sandbox.js";

let quickJSModule: Awaited<
  ReturnType<typeof newQuickJSAsyncWASMModuleFromVariant>
> | null = null;

async function getQuickJS() {
  if (!quickJSModule) {
    quickJSModule = await newQuickJSAsyncWASMModuleFromVariant(
      import("@jitl/quickjs-wasmfile-release-asyncify")
    );
  }
  return quickJSModule;
}

interface REPLEnvironmentOptions extends RLMSandboxFactoryOptions {
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

class REPLEnvironment implements RLMSandbox {
  private ctx: QuickJSAsyncContext | undefined;
  private runtime: QuickJSAsyncRuntime | undefined;
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
      prepareIteration,
      prepareSubAgent,
      logger,
      logLevel,
      sandboxFactory = createQuickJSSandbox,
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
    this.prepareIteration = prepareIteration;
    this.prepareSubAgent = prepareSubAgent;
    this.usageSummary = emptyUsageSummary();
    this.logger = logger;
    this.logLevel = logLevel;
    this.sandboxFactory = sandboxFactory;
    this.createSubAgent = options.createSubAgent;
  }

  async loadContext(context: RLMContext): Promise<void> {
    if (this.contextLoaded) {
      throw new Error("Context already loaded");
    }

    const QuickJS = await getQuickJS();
    this.runtime = QuickJS.newRuntime();
    this.ctx = this.runtime.newContext();

    const contextJson = JSON.stringify(context);

    const consoleObj = this.ctx.newObject();

    const formatArg = (a: unknown): string => {
      if (a === null) return "null";
      if (a === undefined) return "undefined";
      if (typeof a === "object") {
        try {
          return JSON.stringify(a);
        } catch {
          return String(a);
        }
      }
      return String(a);
    };

    const logFn = this.ctx.newFunction("log", (...args) => {
      const nativeArgs = args.map((h) => {
        try {
          return this.ctx!.dump(h);
        } catch {
          return String(h);
        }
      });
      this.consoleOutput.push(nativeArgs.map(formatArg).join(" "));
    });
    this.ctx.setProp(consoleObj, "log", logFn);
    logFn.dispose();

    const errorFn = this.ctx.newFunction("error", (...args) => {
      const nativeArgs = args.map((h) => {
        try {
          return this.ctx!.dump(h);
        } catch {
          return String(h);
        }
      });
      this.consoleOutput.push("ERROR: " + nativeArgs.map(formatArg).join(" "));
    });
    this.ctx.setProp(consoleObj, "error", errorFn);
    errorFn.dispose();

    this.ctx.setProp(this.ctx.global, "console", consoleObj);
    consoleObj.dispose();

    const contextHandle = this.ctx.evalCode(`(${contextJson})`);
    const contextResult = this.ctx.unwrapResult(contextHandle);
    this.ctx.setProp(this.ctx.global, "context", contextResult);
    contextResult.dispose();

    const llmQueryFn = this.ctx.newAsyncifiedFunction(
      "llm_query",
      async (promptHandle) => {
        const prompt = this.ctx!.getString(promptHandle);
        const result = await this.llmQuery(prompt);
        return this.ctx!.newString(result);
      }
    );
    this.ctx.setProp(this.ctx.global, "llm_query", llmQueryFn);
    llmQueryFn.dispose();

    const llmQueryBatchedFn = this.ctx.newAsyncifiedFunction(
      "llm_query_batched",
      async (promptsHandle) => {
        const prompts = this.ctx!.dump(promptsHandle) as string[];
        if (
          !Array.isArray(prompts) ||
          !prompts.every((p) => typeof p === "string")
        ) {
          throw new Error("llm_query_batched expects an array of strings");
        }
        const remaining = this.maxLLMCalls - this.llmCallCount;
        if (prompts.length > remaining) {
          throw new Error(
            `LLM call limit exceeded: batch requires ${prompts.length} calls, only ${remaining} remaining`
          );
        }
        const results = await Promise.all(prompts.map((p) => this.llmQuery(p)));
        const resultHandle = this.ctx!.evalCode(`(${JSON.stringify(results)})`);
        return this.ctx!.unwrapResult(resultHandle);
      }
    );
    this.ctx.setProp(this.ctx.global, "llm_query_batched", llmQueryBatchedFn);
    llmQueryBatchedFn.dispose();

    const subRlmFn = this.ctx.newAsyncifiedFunction(
      "sub_rlm",
      async (promptHandle, subContextHandle) => {
        const prompt = this.ctx!.getString(promptHandle);
        let subContext: RLMContext | undefined;
        try {
          subContext = this.ctx!.dump(subContextHandle) as RLMContext;
        } catch {
          subContext = undefined;
        }

        const result =
          this.currentDepth >= this.maxDepth - 1
            ? await this.llmQuery(prompt)
            : await this.subRlmQuery(prompt, subContext);
        return this.ctx!.newString(result);
      }
    );
    this.ctx.setProp(this.ctx.global, "sub_rlm", subRlmFn);
    subRlmFn.dispose();

    const finalFn = this.ctx.newFunction("FINAL", (answerHandle) => {
      const answer = this.ctx!.dump(answerHandle);
      const obj = this.ctx!.newObject();
      const typeHandle = this.ctx!.newString("final");
      this.ctx!.setProp(obj, "type", typeHandle);
      typeHandle.dispose();
      const valueHandle = this.ctx!.evalCode(`(${JSON.stringify(answer)})`);
      const valueResult = this.ctx!.unwrapResult(valueHandle);
      this.ctx!.setProp(obj, "value", valueResult);
      valueResult.dispose();
      return obj;
    });
    this.ctx.setProp(this.ctx.global, "FINAL", finalFn);
    finalFn.dispose();

    const finalVarFn = this.ctx.newFunction("FINAL_VAR", (varNameHandle) => {
      const varName = this.ctx!.getString(varNameHandle);
      const obj = this.ctx!.newObject();
      const typeHandle = this.ctx!.newString("final_var");
      this.ctx!.setProp(obj, "type", typeHandle);
      typeHandle.dispose();
      const valueHandle = this.ctx!.newString(varName);
      this.ctx!.setProp(obj, "value", valueHandle);
      valueHandle.dispose();
      return obj;
    });
    this.ctx.setProp(this.ctx.global, "FINAL_VAR", finalVarFn);
    finalVarFn.dispose();

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
        prompt: prompt,
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
    this.consoleOutput = [];
    if (!this.ctx) {
      return {
        stdout: "",
        stderr: "",
        error: "QuickJS context not initialized",
      };
    }

    try {
      const deadline = Date.now() + this.timeout;
      this.runtime?.setInterruptHandler(shouldInterruptAfterDeadline(deadline));
      const result = await this.ctx.evalCodeAsync(code);

      if (result.error) {
        const errorMessage = formatThrownValue(this.ctx.dump(result.error));
        result.error.dispose();
        return {
          stdout: this.consoleOutput.join("\n"),
          stderr: errorMessage,
          error: errorMessage,
        };
      }

      const value = result.value;
      let dumpedValue: unknown;

      const state = this.ctx.getPromiseState(value);
      if (state.type === "pending") {
        try {
          const resolved = await this.ctx.resolvePromise(value);
          value.dispose();
          if (resolved.error) {
            const errorMessage = formatThrownValue(
              this.ctx.dump(resolved.error)
            );
            resolved.error.dispose();
            return {
              stdout: this.consoleOutput.join("\n"),
              stderr: errorMessage,
              error: errorMessage,
            };
          }
          dumpedValue = this.ctx.dump(resolved.value);
          resolved.value.dispose();
        } catch (resolveError) {
          value.dispose();
          throw resolveError;
        }
      } else {
        dumpedValue = this.ctx.dump(value);
        value.dispose();
      }

      return {
        stdout: this.consoleOutput.join("\n"),
        stderr: "",
        result: dumpedValue !== undefined ? dumpedValue : undefined,
      };
    } catch (error) {
      if (error instanceof Error) {
        return {
          stdout: this.consoleOutput.join("\n"),
          stderr: error.message,
          error: error.message,
        };
      }
      const errorMessage = formatThrownValue(error);
      return {
        stdout: this.consoleOutput.join("\n"),
        stderr: errorMessage,
        error: errorMessage,
      };
    } finally {
      this.runtime?.removeInterruptHandler();
    }
  }

  getVariable(name: string): unknown {
    try {
      if (this.ctx) {
        const result = this.ctx.evalCode(name);
        if (result.error) {
          result.error.dispose();
          return undefined;
        }
        const value = this.ctx.dump(result.value);
        result.value.dispose();
        return value;
      }
      return undefined;
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
    this.consoleOutput = [];
    if (this.ctx) {
      try {
        this.ctx.dispose();
      } catch {
        // Suppress dispose errors from asyncified functions
      }
      this.ctx = undefined;
    }
    if (this.runtime) {
      try {
        this.runtime.dispose();
      } catch {
        // Suppress dispose errors from asyncified functions
      }
      this.runtime = undefined;
    }
  }
}

export const createQuickJSSandbox: RLMSandboxFactory = (
  options: RLMSandboxFactoryOptions
) => new REPLEnvironment(options);
