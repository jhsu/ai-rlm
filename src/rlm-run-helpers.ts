import type { ModelMessage } from "ai";
import type { RLMSandboxExecutionResult } from "./sandbox.js";
import type {
  RLMContext,
  RLMContextPlanningSettings,
  RLMToolSet,
} from "./rlm-types.js";

export type ResolvedRLMContextPlanningSettings = Required<RLMContextPlanningSettings>;

const DEFAULT_CONTEXT_PLANNING: ResolvedRLMContextPlanningSettings = {
  maxDirectLLMQueryChars: 200000,
  preferSubRLMChars: 500000,
  chunkSizeChars: 120000,
  chunkOverlapChars: 4000,
  metadataMaxDepth: 3,
  metadataMaxEntries: 40,
  metadataMaxPreviewChars: 160,
};

type MetadataEntry = {
  path: string;
  type: string;
  detail: string;
  chars?: number;
  preview?: string;
};

export type RLMInitialPromptContext = {
  currentDepth: number;
  maxDepth: number;
};

export function resolveContextPlanningSettings(
  settings: RLMContextPlanningSettings | undefined
): ResolvedRLMContextPlanningSettings {
  return {
    ...DEFAULT_CONTEXT_PLANNING,
    ...settings,
  };
}

export function buildContextMetadata(
  context: RLMContext,
  planning: ResolvedRLMContextPlanningSettings = DEFAULT_CONTEXT_PLANNING
): string {
  const entries = collectMetadataEntries(context, planning);
  const textFields = collectTextFieldEntries(context, planning);
  const totalTextChars = estimateTextChars(context);
  const totalJsonChars = estimateSerializedChars(context);
  const largestTextFields = textFields
    .sort((a, b) => (b.chars ?? 0) - (a.chars ?? 0))
    .slice(0, 8);

  const nestedMetadata = entries
    .slice(0, planning.metadataMaxEntries)
    .map((entry) => {
      const size = entry.chars !== undefined ? `, ${entry.chars} chars` : "";
      const preview = entry.preview ? `, preview: "${entry.preview}"` : "";
      return `- ${entry.path}: ${entry.type}${size}, ${entry.detail}${preview}`;
    })
    .join("\n");

  const largestTextMetadata =
    largestTextFields.length === 0
      ? "- none"
      : largestTextFields
          .map((entry) => {
            const approxTokens = approximateTokens(entry.chars ?? 0);
            return `- ${entry.path}: ${entry.chars} chars (~${approxTokens} tokens)`;
          })
          .join("\n");

  if (typeof context === "string") {
    const previewLength = context.length <= 2000 ? context.length : 500;
    const preview = context.substring(0, previewLength);
    return `Type: string\nLength: ${context.length} characters (~${approximateTokens(
      context.length
    )} tokens)\nEstimated JSON characters: ${totalJsonChars}\nPreview: "${preview}${
      context.length > previewLength ? "..." : ""
    }"\nLargest text fields:\n${largestTextMetadata}\nNested metadata:\n${nestedMetadata}\nAccess: Use the 'context' variable to read data. Use string methods like context.substring(), context.indexOf(), context.split(), etc.`;
  }

  if (Array.isArray(context)) {
    return `Type: array\nLength: ${context.length} items\nEstimated nested text characters: ${totalTextChars} (~${approximateTokens(
      totalTextChars
    )} tokens)\nEstimated JSON characters: ${totalJsonChars}\nLargest text fields:\n${largestTextMetadata}\nNested metadata:\n${nestedMetadata}\nAccess: Use the 'context' variable. Access items with context[index], iterate with context.forEach() or for...of.`;
  }

  const keys = Object.keys(context);
  const preview = keys.slice(0, 5).join(", ");
  return `Type: object\nKeys: ${keys.length} (${preview}${
    keys.length > 5 ? ", ..." : ""
  })\nEstimated nested text characters: ${totalTextChars} (~${approximateTokens(
    totalTextChars
  )} tokens)\nEstimated JSON characters: ${totalJsonChars}\nLargest text fields:\n${largestTextMetadata}\nNested metadata:\n${nestedMetadata}\nAccess: Use the 'context' variable. Access properties with context.property or context["key"].`;
}

export function createInitialMessages(
  systemPrompt: string,
  contextMeta: string,
  query: string,
  rlmTools?: RLMToolSet,
  planning: ResolvedRLMContextPlanningSettings = DEFAULT_CONTEXT_PLANNING,
  promptContext: RLMInitialPromptContext = {
    currentDepth: 0,
    maxDepth: 1,
  }
): ModelMessage[] {
  const toolEntries = Object.entries(rlmTools ?? {});
  const toolsText =
    toolEntries.length === 0
      ? "No custom tools are available."
      : toolEntries
          .map(([name, tool]) => {
            const schema = tool.inputSchema
              ? `\n  inputSchema: ${JSON.stringify(tool.inputSchema)}`
              : "";
            return `- tools.${name}(input): ${tool.description}${schema}`;
          })
          .join("\n");
  const recursionAvailable = promptContext.currentDepth < promptContext.maxDepth - 1;
  const largeSubtaskGuidance = recursionAvailable
    ? `If an individual field or subtask is above ${planning.preferSubRLMChars} characters and needs multi-step semantic work, prefer sub_rlm(prompt, subContext) over a single llm_query.`
    : `Recursive sub_rlm is not available at this depth (current depth ${promptContext.currentDepth}, maxDepth ${promptContext.maxDepth}); for large semantic work, split into chunks and use llm_query_batched instead of passing one oversized prompt to llm_query.`;

  return [
    {
      role: "system",
      content: `${systemPrompt}\n\nCustom tools available in the REPL:\n${toolsText}\n\nCustom tools are asynchronous. Use await, for example: const results = await tools.search({ query: "..." });`,
    },
    {
      role: "user",
      content: `\
The input context has been loaded into the REPL environment as a variable named 'context'.

Context metadata:
${contextMeta}

Context planning guidance:
- Before calling llm_query, estimate the prompt size using string lengths from the metadata or by inspecting context in code.
- If one semantic prompt would exceed ${planning.maxDirectLLMQueryChars} characters, do not pass it directly to llm_query. Split it into chunks of about ${planning.chunkSizeChars} chars with about ${planning.chunkOverlapChars} chars of overlap${recursionAvailable ? ", or delegate the large subtask with sub_rlm." : "."}
- ${largeSubtaskGuidance}
- If there are many independent items under the direct-query limit, prefer llm_query_batched with one prompt per item or chunk.
- Keep the final synthesis separate from per-item or per-chunk analysis.

Your task: ${query}

Begin by exploring the context to understand its structure, then write JavaScript code to analyze it and answer the query.`,
    },
  ];
}

function collectMetadataEntries(
  value: unknown,
  planning: ResolvedRLMContextPlanningSettings,
  path = "context",
  depth = 0,
  entries: MetadataEntry[] = [],
  seen = new WeakSet<object>()
): MetadataEntry[] {
  if (entries.length >= planning.metadataMaxEntries) {
    return entries;
  }

  if (typeof value === "string") {
    entries.push({
      path,
      type: "string",
      detail: "text field",
      chars: value.length,
      preview: previewString(value, planning.metadataMaxPreviewChars),
    });
    return entries;
  }

  if (value === null) {
    entries.push({ path, type: "null", detail: "null value" });
    return entries;
  }

  if (typeof value !== "object") {
    entries.push({ path, type: typeof value, detail: String(value) });
    return entries;
  }

  if (seen.has(value)) {
    entries.push({ path, type: "object", detail: "circular reference" });
    return entries;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    entries.push({
      path,
      type: "array",
      detail: `${value.length} items`,
    });
    if (depth >= planning.metadataMaxDepth) {
      return entries;
    }

    const sampleCount = Math.min(value.length, 5);
    for (let i = 0; i < sampleCount; i++) {
      collectMetadataEntries(
        value[i],
        planning,
        `${path}[${i}]`,
        depth + 1,
        entries,
        seen
      );
    }
    if (value.length > sampleCount && entries.length < planning.metadataMaxEntries) {
      entries.push({
        path: `${path}[${sampleCount}...]`,
        type: "array",
        detail: `${value.length - sampleCount} more items not shown`,
      });
    }
    return entries;
  }

  const record = value as Record<string, unknown>;
  const keys = Object.keys(record);
  entries.push({
    path,
    type: "object",
    detail: `${keys.length} keys (${keys.slice(0, 8).join(", ")}${
      keys.length > 8 ? ", ..." : ""
    })`,
  });
  if (depth >= planning.metadataMaxDepth) {
    return entries;
  }

  for (const key of keys.slice(0, 12)) {
    collectMetadataEntries(
      record[key],
      planning,
      `${path}.${safePathKey(key)}`,
      depth + 1,
      entries,
      seen
    );
    if (entries.length >= planning.metadataMaxEntries) {
      break;
    }
  }
  return entries;
}

function collectTextFieldEntries(
  value: unknown,
  planning: ResolvedRLMContextPlanningSettings,
  path = "context",
  depth = 0,
  entries: MetadataEntry[] = [],
  seen = new WeakSet<object>()
): MetadataEntry[] {
  if (entries.length >= planning.metadataMaxEntries * 20) {
    return entries;
  }

  if (typeof value === "string") {
    entries.push({
      path,
      type: "string",
      detail: "text field",
      chars: value.length,
      preview: previewString(value, planning.metadataMaxPreviewChars),
    });
    return entries;
  }

  if (value === null || typeof value !== "object") {
    return entries;
  }

  if (seen.has(value)) {
    return entries;
  }
  seen.add(value);

  if (depth > Math.max(planning.metadataMaxDepth + 3, 6)) {
    return entries;
  }

  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      collectTextFieldEntries(
        value[i],
        planning,
        `${path}[${i}]`,
        depth + 1,
        entries,
        seen
      );
    }
    return entries;
  }

  const record = value as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    collectTextFieldEntries(
      record[key],
      planning,
      `${path}.${safePathKey(key)}`,
      depth + 1,
      entries,
      seen
    );
  }
  return entries;
}

function estimateTextChars(value: unknown, seen = new WeakSet<object>()): number {
  if (typeof value === "string") {
    return value.length;
  }
  if (value === null || typeof value !== "object") {
    return 0;
  }
  if (seen.has(value)) {
    return 0;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    return value.reduce<number>(
      (total, item) => total + estimateTextChars(item, seen),
      0
    );
  }

  return Object.values(value as Record<string, unknown>).reduce<number>(
    (total, item) => total + estimateTextChars(item, seen),
    0
  );
}

function estimateSerializedChars(
  value: unknown,
  seen = new WeakSet<object>()
): number {
  if (typeof value === "string") {
    return value.length + 2;
  }
  if (
    value === null ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value).length;
  }
  if (value === undefined || typeof value === "function") {
    return 0;
  }
  if (typeof value !== "object") {
    return String(value).length;
  }
  if (seen.has(value)) {
    return 0;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return 2;
    }
    return (
      2 +
      Math.max(0, value.length - 1) +
      value.reduce<number>(
        (total, item) => total + estimateSerializedChars(item, seen),
        0
      )
    );
  }

  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) {
    return 2;
  }
  return (
    2 +
    Math.max(0, entries.length - 1) +
    entries.reduce<number>(
      (total, [key, item]) =>
        total + key.length + 3 + estimateSerializedChars(item, seen),
      0
    )
  );
}

function approximateTokens(chars: number): number {
  return Math.ceil(chars / 4);
}

function previewString(value: string, maxChars: number): string {
  return value
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxChars)
    .replace(/"/g, '\\"');
}

function safePathKey(key: string): string {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(key) ? key : JSON.stringify(key);
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
