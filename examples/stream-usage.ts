/**
 * RLM - Streaming Example
 *
 * This example demonstrates how to consume RLMAgent.stream() using both
 * fullStream (AI SDK text stream parts) and textStream (text deltas only).
 */

import { RLMAgent } from "../src/rlm";
import { model, subModel } from "./model";

export async function streamingExample() {
  console.log("\n" + "=".repeat(60));
  console.log("RLM Streaming Example");
  console.log("=".repeat(60) + "\n");

  const context = `
    Incident Report Summary:

    Incident INC-1001: API latency spike in us-east-1, duration 12 minutes.
    Root cause: cache invalidation storm after deployment.

    Incident INC-1002: Payment retries increased by 18%.
    Root cause: downstream provider timeout.

    Incident INC-1003: Elevated 500 errors on /checkout for 7 minutes.
    Root cause: database connection pool exhaustion.
  `;

  const query =
    "Summarize the incidents and identify the most severe issue based on impact and duration.";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 8,
    maxLLMCalls: 5,
  });

  console.log("Query:", query);
  console.log("Streaming progress events...\n");

  const streamResult = await agent.stream({
    prompt: query,
    options: { context },
  });

  const fullStreamReader = streamResult.fullStream.getReader();
  const textStreamReader = streamResult.textStream.getReader();

  const progressTask = (async () => {
    while (true) {
      const { done, value } = await fullStreamReader.read();
      if (done) {
        break;
      }

      switch (value.type) {
        case "start":
          console.log("[fullStream] start");
          break;
        case "start-step":
          console.log("[fullStream] start-step");
          break;
        case "finish-step":
          console.log(
            `[fullStream] finish-step reason=${value.finishReason} model=${value.response.modelId}`
          );
          break;
        case "text-start":
          console.log(`[fullStream] text-start id=${value.id}`);
          break;
        case "text-delta":
          console.log(`[fullStream] text-delta length=${value.text.length}`);
          break;
        case "text-end":
          console.log(`[fullStream] text-end id=${value.id}`);
          break;
        case "finish":
          console.log(
            `[fullStream] finish reason=${value.finishReason} totalTokens=${value.totalUsage.totalTokens ?? "unknown"}`
          );
          break;
        case "error":
          console.error("[fullStream] error:", value.error);
          break;
      }
    }
  })();

  const textTask = (async () => {
    let finalText = "";

    while (true) {
      const { done, value } = await textStreamReader.read();
      if (done) {
        break;
      }

      finalText += value;
      process.stdout.write(value);
    }

    return finalText;
  })();

  const [_, finalText] = await Promise.all([progressTask, textTask]);
  const finalResult = await streamResult.output;

  console.log("\n\n--- Final Result ---");
  console.log("Answer:", finalText);
  console.log("Iterations:", finalResult.iterations);
  console.log("LLM Calls:", finalResult.llmCallCount);
  console.log("Steps:", finalResult.steps.length);

  return streamResult;
}

if (import.meta.main) {
  streamingExample().catch(console.error);
}
