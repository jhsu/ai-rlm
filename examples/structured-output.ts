/**
 * RLM - Structured Output Example
 *
 * This example demonstrates validating FINAL / FINAL_VAR against a Zod schema.
 */

import { z } from "zod";
import { RLMAgent } from "../src/rlm";
import { model, subModel } from "./model";

const feedbackSchema = z.object({
  customers: z.array(
    z.object({
      name: z.string(),
      email: z.string(),
      rating: z.number(),
      sentiment: z.enum(["positive", "neutral", "negative"]),
      issue: z.string().nullable(),
    })
  ),
  averageRating: z.number(),
  positiveCount: z.number(),
  negativeCount: z.number(),
});

export async function structuredOutputExample() {
  console.log("\n" + "=".repeat(60));
  console.log("Structured Output Validation Example");
  console.log("=".repeat(60) + "\n");

  const context = `
Customer feedback export:

Alice Nguyen <alice@example.com> gave 5/5 stars.
Comment: "Fast setup, clean docs, exactly what our team needed."

Marcus Lee <marcus@example.net> gave 2/5 stars.
Comment: "Powerful, but the dashboard timed out twice during onboarding."

Priya Shah <priya@example.org> gave 4/5 stars.
Comment: "Works well after configuration. Pricing is fair."
`;

  const query = [
    "Extract every customer feedback record.",
    "Return an object with customers, averageRating, positiveCount, and negativeCount.",
    "Use issue: null when the customer did not report a concrete problem.",
  ].join(" ");

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 8,
    maxLLMCalls: 4,
  });

  const result = await agent.generate({
    prompt: query,
    options: {
      context,
      outputSchema: feedbackSchema,
    },
  });

  const output = feedbackSchema.parse(result.output.structuredOutput);

  console.log("Validated structured output:");
  console.log(JSON.stringify(output, null, 2));
  console.log("\nText output:", result.text);
  console.log("\nIterations:", result.output.iterations);
  console.log("LLM calls:", result.output.llmCallCount);

  return result;
}

if (import.meta.main) {
  structuredOutputExample().catch(console.error);
}

export default structuredOutputExample;
