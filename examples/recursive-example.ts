/**
 * Recursive RLM Example - Demonstrating depth tracking
 *
 * This example shows how the RLM can recursively call itself
 * when using llm_query() with maxDepth > 1
 */

import { RLMAgent } from "../src/rlm.ts";
import { model, subModel } from "./model.ts";

async function recursiveExample() {
  console.log("\n=== Recursive RLM Example ===\n");

  // Context that requires multi-level analysis
  const context = `\
    Company Structure:

    Engineering Department:
    - 50 engineers
    - 5 teams: Frontend, Backend, DevOps, Mobile, QA
    - Average salary: $120,000

    Sales Department:
    - 30 sales reps
    - 3 regions: North America, Europe, Asia
    - Average salary: $80,000 + commission

    Marketing Department:
    - 20 marketers
    - 4 channels: Social, Content, Email, Events
    - Average salary: $75,000

    Total company budget for salaries: $10,000,000
  `;

  const query =
    "Calculate the total salary cost and determine if we're within budget. For each department breakdown, analyze the team composition using sub-queries.";

  // Create agent with recursion depth = 2
  // This means:
  // - Depth 0: Main agent analyzes overall structure
  // - Depth 1: Sub-agents analyze each department
  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 15,
    maxLLMCalls: 30,
    maxDepth: 3, // Enable recursive calls
    verbose: true,
  });

  console.log("Query:", query);
  console.log("Max recursion depth: 3\n");

  try {
    const result = await agent.generate({
      options: {
        context,
        onIterationStart: (evt) => {
          console.log("---iteration start---");
        },
        onIterationComplete: (evt) => {
          console.log("---iteration complete---");
        },
        onLLMCall: (call) => {
          console.log("- LLM called");
        },
        onError: (error) => {
          console.error("- Error:", error);
        },
      },
      prompt: query,
    });

    // Access RLM-specific data via output
    const rlmData = result.output;

    console.log("\n=== Result ===");
    console.log("Answer:", result.text);

    if (rlmData) {
      console.log("\n=== Execution Stats ===");
      console.log("Iterations:", rlmData.iterations);
      console.log("Total LLM Calls:", rlmData.llmCallCount);
      console.log("Max Depth:", 2);
      console.log("Total Steps:", rlmData.steps.length);

      // Show execution trajectory
      console.log("\n=== Execution Trajectory ===");
      rlmData.steps.forEach((step: any, index: number) => {
        console.log(`\nStep ${index + 1} (Iteration ${step.iteration}):`);
        console.log("Reasoning:", step.reasoning.substring(0, 100) + "...");
        console.log("Code:", step.code.substring(0, 80) + "...");
        console.log("Output preview:", step.output.substring(0, 100) + "...");
      });
    }
  } catch (error) {
    console.error("Error:", error);
  }
}

// Run if executed directly
if (import.meta.main) {
  recursiveExample();
}

export { recursiveExample };
