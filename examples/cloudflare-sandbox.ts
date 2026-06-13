/**
 * Cloudflare sandbox example for RLM.
 *
 * This example is intended to run inside a Cloudflare Worker with a
 * `worker_loaders` binding named `LOADER`. The sandbox itself is provided by
 * `@cloudflare/codemode`'s DynamicWorkerExecutor.
 *
 * Required packages for a Worker project using this example:
 *   npm install ai ai-rlm @ai-sdk/openai @cloudflare/codemode zod
 *
 * wrangler.jsonc:
 *   {
 *     "worker_loaders": [{ "binding": "LOADER" }],
 *     "compatibility_flags": ["nodejs_compat"]
 *   }
 */

import { createOpenAI } from "@ai-sdk/openai";
import { DynamicWorkerExecutor } from "@cloudflare/codemode";
import {
  RLMAgent,
  createCloudflareSandbox,
  type RLMSandboxFactory,
} from "../src/index";

interface Env {
  LOADER: unknown;
  OPENAI_API_KEY: string;
}

export default {
  async fetch(_request: Request, env: Env): Promise<Response> {
    const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY });
    const executor = new DynamicWorkerExecutor({
      loader: env.LOADER,
      // Cloudflare codemode blocks outbound network access by default.
      globalOutbound: null,
    });

    const sandboxFactory: RLMSandboxFactory = (options) =>
      createCloudflareSandbox({
        ...options,
        executor,
      });

    const agent = new RLMAgent({
      model: openai("gpt-5.4"),
      subModel: openai("gpt-5.4-mini"),
      maxIterations: 5,
      maxLLMCalls: 5,
      sandboxFactory,
    });

    const context = `
Project status:
- API migration is complete.
- Billing migration is in progress and blocked on invoice export validation.
- The mobile launch is scheduled for Friday.
- Customer support needs the billing migration status by Wednesday.
`;

    const result = await agent.generate({
      prompt: "What is blocked, and who needs to be notified?",
      options: { context },
    });

    return Response.json({
      answer: result.text,
      iterations: result.output?.iterations,
      llmCallCount: result.output?.llmCallCount,
      steps: result.output?.steps.map((step) => ({
        iteration: step.iteration,
        code: step.code,
        output: step.output,
      })),
    });
  },
};
