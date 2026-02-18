/**
 * RLM Tool - Tool Factory for Recursive Language Model
 *
 * This module provides a factory function to create RLM tools that can be used
 * with the Vercel AI SDK's generateText, ToolLoopAgent, or any tool-compatible API.
 */

import { tool } from "ai";
import type { LanguageModel } from "ai";
import { z } from "zod";
import { RLMAgent, type RLMAgentSettings } from "./rlm.js";

/**
 * Configuration options for the RLM tool
 */
export interface RLMToolConfig extends Partial<RLMAgentSettings> {
  /** Model for the root agent */
  model?: LanguageModel;
  /** Model for sub-LLM queries (defaults to model) */
  subModel?: LanguageModel;
  /** Maximum iterations for the REPL loop (default: 20) */
  maxIterations?: number;
  /** Maximum sub-LLM calls per execution (default: 50) */
  maxLLMCalls?: number;
  /** Maximum characters in REPL output (default: 100000) */
  maxOutputChars?: number;
}

/**
 * Create an RLM tool for use with AI SDK functions.
 *
 * This tool allows models to analyze large contexts iteratively using JavaScript
 * code execution and sub-LLM queries. It's particularly useful for:
 * - Searching through large documents
 * - Analyzing datasets that exceed context limits
 * - Extracting structured information from unstructured data
 * - Performing complex multi-step analysis
 *
 * @example
 * ```typescript
 * import { createRLMTool } from './rlm-tool.js';
 * import { generateText } from 'ai';
 *
 * const rlmTool = createRLMTool({
 *   model: openai('gpt-4.1'),
 *   subModel: openai('gpt-4.1-mini'),
 * });
 *
 * const result = await generateText({
 *   model: openai('gpt-4.1'),
 *   tools: { deepAnalyze: rlmTool },
 *   prompt: 'Find all security issues in this codebase',
 * });
 * ```
 */
export function createRLMTool(config: RLMToolConfig = {}) {
  return tool({
    description: `Analyze large contexts iteratively using JavaScript code execution.

Use this tool when you need to:
- Search through or analyze documents/datasets too large for direct processing
- Extract structured information from unstructured text
- Perform complex multi-step analysis requiring code execution
- Answer questions about large codebases, logs, or datasets

The tool will:
1. Write JavaScript code to explore the context
2. Execute the code in a secure sandbox (node:vm)
3. Use sub-LLM calls for semantic understanding when needed
4. Iterate until it finds the answer

Provide the context (string, array of strings, or JSON object) and your query/question.`,

    inputSchema: z.object({
      context: z
        .union([
          z.string().describe("Text document or content to analyze"),
          z.array(z.string()).describe("Array of text lines or documents"),
          z.any().describe("JSON object or structured data"),
        ])
        .describe("The large context, document, or dataset to analyze"),

      query: z
        .string()
        .describe(
          "The specific question, task, or instruction to perform on the context. Be clear and specific."
        ),

      maxIterations: z
        .number()
        .min(1)
        .max(100)
        .optional()
        .describe("Maximum number of REPL iterations (default: 20)"),

      maxLLMCalls: z
        .number()
        .min(1)
        .max(200)
        .optional()
        .describe("Maximum sub-LLM calls allowed (default: 50)"),
    }),

    execute: async (
      { context, query, maxIterations, maxLLMCalls },
      { abortSignal }
    ) => {
      // Create RLMAgent with merged config
      const agent = new RLMAgent({
        model: config.model!,
        subModel: config.subModel,
        maxIterations: maxIterations ?? config.maxIterations ?? 20,
        maxLLMCalls: maxLLMCalls ?? config.maxLLMCalls ?? 50,
        maxOutputChars: config.maxOutputChars ?? 100000,
        verbose: false,
      });

      // Execute the analysis
      // Convert context to string if needed
      const contextStr =
        typeof context === "string"
          ? context
          : Array.isArray(context)
          ? context.join("\n")
          : JSON.stringify(context, null, 2);

      const result = await agent.generate({
        messages: [
          {
            role: "user",
            content: `Context:\n${contextStr}\n\nQuery: ${query}`,
          },
        ],
        abortSignal,
      });

      // Return just the essential information
      return {
        answer: result.text,
        iterations: result.iterations,
        stepsTaken: result.steps.length,
      };
    },
  });
}

/**
 * Create a pre-configured RLM tool for specific use cases.
 *
 * @example
 * ```typescript
 * // Create a tool optimized for code analysis
 * const codeAnalyzer = createRLMTool({
 *   model: openai('gpt-4.1'),
 *   subModel: openai('gpt-4.1-mini'),
 *   maxIterations: 30,
 *   maxLLMCalls: 50,
 * });
 * ```
 */
export function createRLMToolForCodeAnalysis(config: RLMToolConfig = {}) {
  return createRLMTool({
    model: config.model,
    subModel: config.subModel,
    maxIterations: config.maxIterations ?? 30,
    maxLLMCalls: config.maxLLMCalls ?? 50,
    maxOutputChars: config.maxOutputChars ?? 100000,
  });
}

/**
 * Create a pre-configured RLM tool optimized for log analysis.
 *
 * @example
 * ```typescript
 * const logAnalyzer = createRLMToolForLogAnalysis({
 *   model: openai('gpt-4.1'),
 *   maxIterations: 25,
 * });
 * ```
 */
export function createRLMToolForLogAnalysis(config: RLMToolConfig = {}) {
  return createRLMTool({
    model: config.model,
    subModel: config.subModel,
    maxIterations: config.maxIterations ?? 25,
    maxLLMCalls: config.maxLLMCalls ?? 30,
    maxOutputChars: config.maxOutputChars ?? 100000,
  });
}

/**
 * Create a pre-configured RLM tool optimized for document search.
 *
 * @example
 * ```typescript
 * const documentSearcher = createRLMToolForDocumentSearch({
 *   model: openai('gpt-4.1'),
 *   maxIterations: 20,
 * });
 * ```
 */
export function createRLMToolForDocumentSearch(config: RLMToolConfig = {}) {
  return createRLMTool({
    model: config.model,
    subModel: config.subModel,
    maxIterations: config.maxIterations ?? 20,
    maxLLMCalls: config.maxLLMCalls ?? 20,
    maxOutputChars: config.maxOutputChars ?? 100000,
  });
}

// ============================================================================
// Export
// ============================================================================

export default createRLMTool;
