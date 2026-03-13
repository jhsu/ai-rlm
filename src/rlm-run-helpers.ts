import type { ModelMessage } from "ai";
import type { RLMSandboxExecutionResult } from "./sandbox.js";
import type { RLMContext } from "./rlm-types.js";

export function buildContextMetadata(context: RLMContext): string {
  if (typeof context === "string") {
    const preview = context.substring(0, 200);
    return `Type: string\nLength: ${context.length} characters\nPreview: "${preview}${
      context.length > 200 ? "..." : ""
    }"\nAccess: Use the 'context' variable to read data. Use string methods like context.substring(), context.indexOf(), context.split(), etc.`;
  }

  if (Array.isArray(context)) {
    const preview = context.slice(0, 3).join("\n");
    return `Type: array\nLength: ${context.length} items\nPreview: [\n${preview}${
      context.length > 3 ? "\n..." : ""
    }\n]\nAccess: Use the 'context' variable. Access items with context[index], iterate with context.forEach() or for...of.`;
  }

  const keys = Object.keys(context);
  const preview = keys.slice(0, 5).join(", ");
  return `Type: object\nKeys: ${keys.length} (${preview}${
    keys.length > 5 ? ", ..." : ""
  })\nAccess: Use the 'context' variable. Access properties with context.property or context["key"].`;
}

export function createInitialMessages(
  systemPrompt: string,
  contextMeta: string,
  query: string
): ModelMessage[] {
  return [
    { role: "system", content: systemPrompt },
    {
      role: "user",
      content: `\
The input context has been loaded into the REPL environment as a variable named 'context'.

Context metadata:
${contextMeta}

Your task: ${query}

Begin by exploring the context to understand its structure, then write JavaScript code to analyze it and answer the query.`,
    },
  ];
}

export function buildExecutionOutput(
  executionResult: RLMSandboxExecutionResult
): string {
  let fullOutput = executionResult.stdout;

  if (executionResult.result !== undefined && executionResult.result !== null) {
    fullOutput += `\n[Return value]: ${JSON.stringify(executionResult.result)}`;
  }

  if (executionResult.error) {
    fullOutput += `\n[Error]: ${executionResult.error}`;
  }

  return fullOutput;
}

export function truncateOutput(output: string, maxOutputChars: number): string {
  return output.length > maxOutputChars
    ? output.substring(0, maxOutputChars) + "\n...[truncated]"
    : output;
}

export function extractReasoning(response: string): string {
  const reasoningParts = response.split("```");
  return reasoningParts.length > 0 ? (reasoningParts[0] ?? "").trim() : "";
}

export function buildOutputMetadata(
  fullOutput: string,
  truncatedOutput: string,
  previewLength: number,
  error?: string
): string {
  const outputPreview = truncatedOutput.substring(0, previewLength);

  return [
    `Output metadata:`,
    `- Length: ${fullOutput.length} characters`,
    `- Preview:\n${outputPreview}${fullOutput.length > previewLength ? "\n..." : ""}`,
    error ? `- Error: ${error}` : `- Errors: none`,
    `\nFull output is stored in the REPL environment. Use variables to access computed results. Continue with the next step.`,
  ].join("\n");
}
