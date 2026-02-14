import { createOpenAI } from "@ai-sdk/openai";

const openai = createOpenAI();

export const model = openai("gpt-5.2");
export const subModel = openai("gpt-5-mini");
