import { extractFinalAnswer } from "./rlm-utils.js";
import type { RLMSandbox } from "./sandbox.js";

export async function resolveFinalAnswer(
  response: string,
  repl: Pick<RLMSandbox, "getVariable">
): Promise<string | undefined> {
  const finalAnswer = extractFinalAnswer(response);

  if (!finalAnswer?.content) {
    return undefined;
  }

  if (finalAnswer.type === "direct") {
    return finalAnswer.content;
  }

  const variableValue = await repl.getVariable(finalAnswer.content);
  if (variableValue === undefined) {
    return undefined;
  }

  return typeof variableValue === "object"
    ? JSON.stringify(variableValue)
    : String(variableValue);
}

export function extractRequestedFinalVariable(
  response: string
): string | undefined {
  const finalAnswer = extractFinalAnswer(response);
  return finalAnswer?.type === "variable" ? finalAnswer.content : undefined;
}
