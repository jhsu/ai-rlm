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
  newQuickJSAsyncWASMModuleFromVariant,
  type QuickJSAsyncContext,
  type QuickJSAsyncRuntime,
} from "quickjs-emscripten-core";

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

interface REPLEnvironmentOptions {
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
  verbose?: boolean;
}

const emptyUsageSummary = (): RLMUsageSummary => ({
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  reasoningTokens: 0,
  cachedInputTokens: 0,
});

const toNumber = (value: unknown): number => {
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) ? num : 0;
};

function usageFromGenerateResult(result: unknown): RLMUsageSummary {
  const raw = (result as any)?.usage ?? {};
  const inputTokens = toNumber(raw.inputTokens ?? raw.promptTokens ?? raw.prompt_tokens);
  const outputTokens = toNumber(
    raw.outputTokens ?? raw.completionTokens ?? raw.completion_tokens
  );
  const totalTokens = toNumber(raw.totalTokens ?? raw.total_tokens ?? inputTokens + outputTokens);
  const reasoningTokens = toNumber(
    raw.reasoningTokens ??
      raw.reasoning_tokens ??
      raw.completionTokensDetails?.reasoningTokens ??
      raw.completion_tokens_details?.reasoning_tokens
  );
  const cachedInputTokens = toNumber(
    raw.cachedInputTokens ??
      raw.cached_tokens ??
      raw.promptTokensDetails?.cachedTokens ??
      raw.prompt_tokens_details?.cached_tokens
  );

  return {
    inputTokens,
    outputTokens,
    totalTokens,
    reasoningTokens,
    cachedInputTokens,
  };
}

function addUsage(target: RLMUsageSummary, delta: RLMUsageSummary): void {
  target.inputTokens += delta.inputTokens;
  target.outputTokens += delta.outputTokens;
  target.totalTokens += delta.totalTokens;
  target.reasoningTokens += delta.reasoningTokens;
  target.cachedInputTokens += delta.cachedInputTokens;
}

function mergeUsage(a: RLMUsageSummary, b: RLMUsageSummary): RLMUsageSummary {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    totalTokens: a.totalTokens + b.totalTokens,
    reasoningTokens: a.reasoningTokens + b.reasoningTokens,
    cachedInputTokens: a.cachedInputTokens + b.cachedInputTokens,
  };
}

/**
 * Sandbox environment for executing JavaScript code safely
 * Uses QuickJS WebAssembly sandbox
 */
class REPLEnvironment {
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
      prepareIteration,
      prepareSubAgent,
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
    this.prepareIteration = prepareIteration;
    this.prepareSubAgent = prepareSubAgent;
    this.usageSummary = emptyUsageSummary();
    this.verbose = verbose;
  }

  /**
   * Load context into the REPL environment
   */
  async loadContext(context: RLMContext): Promise<void> {
    if (this.contextLoaded) {
      throw new Error("Context already loaded");
    }

    const QuickJS = await getQuickJS();
    this.runtime = QuickJS.newRuntime();
    this.ctx = this.runtime.newContext();

    let contextJson: string;
    if (typeof context === "string") {
      contextJson = JSON.stringify(context);
    } else if (Array.isArray(context)) {
      contextJson = JSON.stringify(context);
    } else {
      contextJson = JSON.stringify(context);
    }

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
        const results = await Promise.all(prompts.map((p) => this.llmQuery(p)));
        return this.ctx!.newString(JSON.stringify(results));
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
      const answer = this.ctx!.getString(answerHandle);
      const obj = this.ctx!.newObject();
      const typeHandle = this.ctx!.newString("final");
      this.ctx!.setProp(obj, "type", typeHandle);
      typeHandle.dispose();
      const valueHandle = this.ctx!.newString(answer);
      this.ctx!.setProp(obj, "value", valueHandle);
      valueHandle.dispose();
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

  /**
   * Query a recursive sub-RLM agent
   */
  async subRlmQuery(prompt: string, subContext?: RLMContext): Promise<string> {
    if (this.llmCallCount >= this.maxLLMCalls) {
      throw new Error(
        `LLM call limit exceeded: ${this.llmCallCount}/${this.maxLLMCalls}`
      );
    }

    try {
      let nextPrompt = prompt;
      let nextSubContext = subContext;
      let hookSubAgentSettings: Partial<RLMAgentSettings> | undefined;

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
            throw new Error(hookResult.reason ?? "prepareSubAgent aborted sub-agent execution");
          }

          if (hookResult.action === "fallback_to_llm_query") {
            return await this.llmQuery(nextPrompt);
          }
        }
      }

      // Create a sub-RLM agent with decreased depth
      const defaultSubAgentSettings: RLMAgentSettings = {
        model: this.model,
        subModel: this.subModel,
        maxIterations: Math.max(5, Math.floor(this.maxIterations / 2)),
        maxLLMCalls: Math.max(10, Math.floor(this.maxLLMCalls / 2)),
        maxOutputChars: this.maxOutputChars,
        maxDepth: this.maxDepth,
        prepareIteration: this.prepareIteration,
        prepareSubAgent: this.prepareSubAgent,
        verbose: this.verbose,
      };

      const subAgent = new RLMAgent({
        ...defaultSubAgentSettings,
        ...hookSubAgentSettings,
      });

      // The sub-agent inherits our call count budget
      const result = await subAgent._generate({
        context: nextSubContext || "No context provided",
        query: nextPrompt,
        currentDepth: this.currentDepth + 1,
      });

      // Track calls used by sub-agent
      this.llmCallCount += result.llmCallCount;
      addUsage(this.usageSummary, result.usage);

      return result.text;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`sub_rlm failed: ${msg}`);
    }
  }

  /**
   * Execute JavaScript code in the sandbox with timeout protection
   * With ASYNCIFY, async host functions can be called synchronously from QuickJS,
   * so we don't need to wrap code in async IIFE - await works at top level.
   */
  async executeJavaScript(code: string): Promise<{
    stdout: string;
    stderr: string;
    error?: string;
    result?: unknown;
  }> {
    this.consoleOutput = [];
    if (!this.ctx) {
      return {
        stdout: "",
        stderr: "",
        error: "QuickJS context not initialized",
      };
    }

    try {
      const result = await this.ctx.evalCodeAsync(code);

      if (result.error) {
        const errorVal = this.ctx.dump(result.error);
        result.error.dispose();
        return {
          stdout: this.consoleOutput.join("\n"),
          stderr: String(errorVal),
          error: String(errorVal),
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
            const errorVal = this.ctx.dump(resolved.error);
            resolved.error.dispose();
            return {
              stdout: this.consoleOutput.join("\n"),
              stderr: String(errorVal),
              error: String(errorVal),
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

      const stdout = this.consoleOutput.join("\n");

      return {
        stdout,
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
      const errorStr = String(error);
      return {
        stdout: this.consoleOutput.join("\n"),
        stderr: errorStr,
        error: errorStr,
      };
    }
  }

  /**
   * Get a variable value from the VM
   */
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

  /**
   * Get current LLM call count
   */
  getLLMCallCount(): number {
    return this.llmCallCount;
  }

  getUsageSummary(): RLMUsageSummary {
    return { ...this.usageSummary };
  }

  /**
   * Clean up
   * Note: Asyncified functions may cause dispose errors due to lingering host refs.
   * This is a known issue in quickjs-emscripten. We catch and suppress these errors
   * since they occur after all actual work is complete.
   */
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
  // First, remove code blocks to only search in reasoning text
  const textWithoutCode = text.replace(/```[\s\S]*?```/g, "");

  const finalVarMatch = textWithoutCode.match(
    /FINAL_VAR\s*\(\s*["']?([^"')\s]+)["']?\s*\)/i
  );
  if (finalVarMatch) {
    const content = finalVarMatch[1];
    if (content) {
      return { type: "variable", content };
    }
  }

  const finalMatch = textWithoutCode.match(/FINAL\s*\(\s*["']?([^"')]+)["']?\s*\)/i);
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
- llm_query(prompt): Query a sub-LLM (~500K char capacity) for semantic analysis. Returns the LLM response string directly (synchronous call).
- llm_query_batched(prompts[]): Query multiple prompts in parallel. Returns array of response strings.
- sub_rlm(prompt, subContext?): Launch a recursive sub-RLM agent for complex sub-tasks. Returns the final answer string.
- console.log(): ALWAYS log to see results
- Standard JavaScript: JSON, Array methods, String methods, Math, etc.

IMPORTANT: llm_query, llm_query_batched, and sub_rlm return values directly - do NOT use await. They are synchronous in this environment.
Example: const sentiment = llm_query("Analyze sentiment");  // No await needed

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
5. FINALIZE EARLY - Once you have successfully extracted the answer into a variable and confirmed it looks correct (one quick console.log), immediately return it with FINAL_VAR(variable_name). Do NOT keep exploring after you already have the answer.

OUTPUT VISIBILITY: After each code execution, you will only see a short preview of the output (first ~500 characters) and its total length. The full output exists in the REPL but is NOT included in your conversation history. To retain information across iterations:
- Store results in variables: \`const results = ...\`
- Use console.log() for short summaries only
- Access previously stored variables in later iterations

CORRECT WORKFLOW (Simple extraction):
✓ Step 1: console.log(context.slice(0,200));  // Quick peek
✓ Step 2: const answer = context.match(/codename:\s*(\w+)/i)?.[1];  // Extract
           console.log(answer);  // Verify: should show "PHOENIX"
✓ Step 3: FINAL_VAR(answer);  // IMMEDIATE - do not write more code

EXAMPLE - Finding a codename:
  // Step 1: Explore (check what we're working with)
  console.log(context.length);  // 290

  // Step 2: Extract (assign to variable!)
  const codename = context.match(/codename is:\s*(\w+)/i)?.[1];
  console.log(codename);  // Must print to verify: "PHOENIX"

  // Step 3: Finalize (immediately, no more code!)
  FINAL_VAR(codename);  // Returns: "PHOENIX"

INCORRECT (wastes iterations):
✗ Extract answer, then explore more "just to be sure"
✗ Extract answer, then extract it 3 different ways to verify
✗ Extract answer, then write a summary instead of FINALIZING

FINAL vs FINAL_VAR - WHEN TO USE EACH:

Use FINAL(answer) for simple, short answers (< 100 chars):
  Step 1: const total = data.reduce((sum, x) => sum + x.value, 0);
           console.log("Total:", total);  // Output: Total: 42
  Step 2: FINAL(42);  // Direct value, not a variable name

Use FINAL_VAR(variableName) for computed results (ALWAYS prefer this):
  Step 1: const extracted = context.match(/code: (\w+)/)?.[1];
           console.log("Found:", extracted);  // Output: Found: PHOENIX
  Step 2: FINAL_VAR(extracted);  // variableName WITHOUT quotes, NOT the value

COMMON MISTAKE - DO NOT DO THIS:
  // WRONG:
  FINAL_VAR("extracted");  // Putting variable name in quotes
  FINAL("extracted");      // Using string instead of actual value
  FINAL(extracted);        // Passing variable to FINAL instead of FINAL_VAR

CORRECT:
  FINAL_VAR(extracted);    // Pass variable name (no quotes), extracts value from REPL
  FINAL(42);              // Pass actual value directly

CRITICAL - FINAL AND FINAL_VAR MUST CONTAIN ONLY THE CLEAN ANSWER:
  // WRONG - includes descriptive text:
  FINAL("The secret project codename is: " + answer);  // ❌ Output: "The secret project codename is: PHOENIX"
  FINAL("Codename: " + codename);                    // ❌ Output: "Codename: PHOENIX"

  // CORRECT - clean value only:
  FINAL(answer);           // ✓ Output: "PHOENIX" (just the value)
  FINAL_VAR(codename);     // ✓ Output: "PHOENIX" (value from variable)

Put any explanation in your REASONING TEXT before the code block, NOT inside FINAL() or FINAL_VAR().

CRITICAL: FINAL_VAR must be placed OUTSIDE code blocks, in your reasoning text AFTER the code block.
- WRONG: \`\`\`javascript const x = 1; FINAL_VAR(x); \`\`\`  ← Code won't execute
- CORRECT: \`\`\`javascript const x = 1; \`\`\` FINAL_VAR(x)  ← Code executes, then variable is retrieved

When done, provide your final answer using:
- FINAL(your_answer) - to submit directly (use for simple answers under 100 chars, value ONLY)
- FINAL_VAR(variable_name) - to submit a variable from the REPL (preferred for computed results)

Think step-by-step and show your reasoning before each code block.`;

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
    const repl = new REPLEnvironment({
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
