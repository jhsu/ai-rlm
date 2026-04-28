import { extractFinalAnswer } from "./rlm-utils.js";
import type { RLMSandbox } from "./sandbox.js";

function stringifyFinalValue(value: unknown): string {
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

async function resolveFinalDescriptor(
  finalAnswer: { type: "direct" | "variable"; content: unknown } | null,
  repl: Pick<RLMSandbox, "getVariable">
): Promise<string | undefined> {
  if (!finalAnswer) {
    return undefined;
  }

  if (finalAnswer.type === "direct") {
    return stringifyFinalValue(finalAnswer.content);
  }

  if (typeof finalAnswer.content !== "string" || !finalAnswer.content) {
    return undefined;
  }

  const variableValue = await repl.getVariable(finalAnswer.content);
  if (variableValue === undefined) {
    return undefined;
  }

  return stringifyFinalValue(variableValue);
}

export async function resolveFinalAnswer(
  response: string,
  repl: Pick<RLMSandbox, "getVariable">
): Promise<string | undefined> {
  return resolveFinalDescriptor(extractFinalAnswer(response), repl);
}

export async function resolveExecutionFinalAnswer(
  result: unknown,
  repl: Pick<RLMSandbox, "getVariable">
): Promise<string | undefined> {
  if (!result || typeof result !== "object") {
    return undefined;
  }

  const descriptor = result as { type?: unknown; value?: unknown };
  if (descriptor.type === "final") {
    return resolveFinalDescriptor(
      { type: "direct", content: descriptor.value },
      repl
    );
  }
  if (descriptor.type === "final_var") {
    return resolveFinalDescriptor(
      { type: "variable", content: descriptor.value },
      repl
    );
  }

  return undefined;
}

export function extractRequestedFinalVariable(
  response: string
): string | undefined {
  const finalAnswer = extractFinalAnswer(response);
  return finalAnswer?.type === "variable" ? finalAnswer.content : undefined;
}
