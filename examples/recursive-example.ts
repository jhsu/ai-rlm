/**
 * Recursive RLM Example - Demonstrating depth tracking
 *
 * This example shows how the RLM can recursively call itself
 * when using llm_query() with maxDepth > 1
 */

import { RLMAgent } from "../src/rlm.js";
import { model, subModel } from "./model";

async function recursiveExample() {
  console.log("\n=== Recursive RLM Example ===\n");

  // Context that requires multi-level analysis
  const context = `
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
    maxDepth: 2, // Enable recursive calls
    verbose: true,
  });

  console.log("Query:", query);
  console.log("Max recursion depth: 2\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("\n=== Result ===");
    console.log("Answer:", result.text);
    console.log("\n=== Execution Stats ===");
    console.log("Iterations:", result.iterations);
    console.log("Total LLM Calls:", result.llmCallCount);
    console.log("Max Depth Reached:", result.maxDepthReached);
    console.log("Total Steps:", result.steps.length);

    // Show depth breakdown
    console.log("\n=== Depth Breakdown ===");
    const depthCounts: Record<number, number> = {};
    result.steps.forEach((step) => {
      depthCounts[step.depth] = (depthCounts[step.depth] || 0) + 1;
      if (step.subSteps) {
        step.subSteps.forEach((subStep) => {
          depthCounts[subStep.depth] = (depthCounts[subStep.depth] || 0) + 1;
        });
      }
    });

    Object.entries(depthCounts).forEach(([depth, count]) => {
      console.log(`  Depth ${depth}: ${count} steps`);
    });

    // Show recursive step example
    const recursiveStep = result.steps.find(
      (s) => s.subSteps && s.subSteps.length > 0
    );
    if (recursiveStep?.subSteps && recursiveStep.subSteps.length > 0) {
      const firstSubStep = recursiveStep.subSteps[0];
      if (firstSubStep) {
        console.log("\n=== Recursive Call Example ===");
        console.log(`Parent step (depth ${recursiveStep.depth}):`);
        console.log("  Reasoning:", recursiveStep.reasoning);
        console.log("  Code:", recursiveStep.code);
        console.log(`\n  Sub-step (depth ${firstSubStep.depth}):`);
        console.log("    Reasoning:", firstSubStep);
      }
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
