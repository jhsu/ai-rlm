import type { ModelMessage } from "ai";
import type { RLMContext } from "./rlm-types.js";

export interface NormalizedGenerateInput {
  context: RLMContext;
  query: string;
}

interface NormalizeGenerateInputParams {
  prompt?: unknown;
  messages?: ModelMessage[];
  options?: {
    context?: RLMContext;
  };
}

export function normalizeGenerateInput(
  params: NormalizeGenerateInputParams
): NormalizedGenerateInput {
  const { prompt, messages, options } = params;
  const explicitContext = options?.context;

  if (explicitContext !== undefined) {
    return {
      context: explicitContext,
      query: extractQueryFromPromptOrMessages(prompt, messages),
    };
  }

  if (messages && messages.length > 0) {
    const systemMessages = messages.filter(
      (message: ModelMessage) => message.role === "system"
    );

    return {
      context:
        systemMessages.length > 0
          ? systemMessages
              .map((message: ModelMessage) =>
                typeof message.content === "string"
                  ? message.content
                  : "[complex content]"
              )
              .join("\n")
          : "No context provided. Answer based on the query.",
      query: extractLastUserMessage(messages),
    };
  }

  return {
    context: "No context provided. Answer based on the query.",
    query: typeof prompt === "string" ? prompt : "Please provide a query.",
  };
}

function extractQueryFromPromptOrMessages(
  prompt: unknown,
  messages: ModelMessage[] | undefined
): string {
  if (typeof prompt === "string") {
    return prompt;
  }

  return extractLastUserMessage(messages);
}

function extractLastUserMessage(messages: ModelMessage[] | undefined): string {
  if (!messages || messages.length === 0) {
    return "Please provide a query.";
  }

  const lastUserMessage = messages
    .filter((message: ModelMessage) => message.role === "user")
    .pop();

  return lastUserMessage && typeof lastUserMessage.content === "string"
    ? lastUserMessage.content
    : "Please provide a query.";
}
