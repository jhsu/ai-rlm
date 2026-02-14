/**
 * RLM (Recursive Language Model) - Comprehensive Examples
 *
 * This file demonstrates various use cases for the RLM implementation
 * using JavaScript with vm2 sandbox and the RLMAgent API.
 */

import { RLMAgent } from "../src/rlm";
import { model, subModel } from "./model";

// ============================================================================
// Example 1: Simple Text Search
// ============================================================================

/**
 * Example: Find specific information in a document
 * Demonstrates basic text searching with JavaScript string methods
 */
export async function example1SimpleTextSearch() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 1: Simple Text Search");
  console.log("=".repeat(60) + "\n");

  const context = `
    Company: Acme Corporation
    Founded: 2020
    CEO: Jane Smith
    Employees: 500
    Revenue: $50 million
    Headquarters: San Francisco, CA

    Products:
    - Widget Pro ($99)
    - Widget Plus ($149)
    - Widget Enterprise ($499)

    The secret project codename is: PHOENIX
  `;

  const query = "What is the secret project codename?";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 5,
    maxLLMCalls: 3,
    verbose: false,
  });

  console.log("Query:", query);
  console.log("Context length:", context.length, "characters\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Answer:", result.text);
    console.log("✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    // Show the trajectory
    console.log("\n--- Execution Trajectory ---");
    result.steps.forEach((step) => {
      console.log(`\nStep ${step.iteration}:`);
      console.log("Reasoning:", step.reasoning.substring(0, 100) + "...");
      console.log("Code:", step.code.substring(0, 80) + "...");
      console.log("Output:", step.output.substring(0, 100) + "...");
    });

    return result;
  } catch (error) {
    console.error("Error in Example 1:", error);
    throw error;
  }
}

// ============================================================================
// Example 2: Data Analysis (JSON Processing)
// ============================================================================

/**
 * Example: Analyze structured JSON data
 * Demonstrates working with objects and arrays in JavaScript
 */
export async function example2DataAnalysis() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 2: Data Analysis (JSON Processing)");
  console.log("=".repeat(60) + "\n");

  const context = {
    sales: [
      { month: "January", revenue: 45000, units: 450, region: "North" },
      { month: "February", revenue: 52000, units: 520, region: "North" },
      { month: "March", revenue: 48000, units: 480, region: "North" },
      { month: "January", revenue: 38000, units: 380, region: "South" },
      { month: "February", revenue: 42000, units: 420, region: "South" },
      { month: "March", revenue: 51000, units: 510, region: "South" },
    ],
    targets: {
      q1: 250000,
      avgUnitsPerMonth: 450,
    },
  };

  const query =
    "Did we meet our Q1 revenue target? Calculate total Q1 revenue and compare.";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 8,
    maxLLMCalls: 5,
    verbose: false,
  });

  console.log("Query:", query);
  console.log(
    "Data:",
    JSON.stringify(context, null, 2).substring(0, 200) + "...\n",
  );

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Answer:", result.text);
    console.log("✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    console.log("\n--- Key Code Executed ---");
    result.steps.forEach((step) => {
      console.log(`\nStep ${step.iteration}:`);
      console.log("Code:\n", step.code);
    });

    return result;
  } catch (error) {
    console.error("Error in Example 2:", error);
    throw error;
  }
}

// ============================================================================
// Example 3: Needle in Haystack
// ============================================================================

/**
 * Example: Find a specific value in a large document
 * Demonstrates efficient chunking and searching strategies
 */
export async function example3NeedleInHaystack() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 3: Needle in Haystack");
  console.log("=".repeat(60) + "\n");

  // Generate a large context with a hidden target
  const lines: string[] = [];
  const targetLine = 5000;
  const targetValue = "PHOENIX-42";

  console.log("Generating context with 10,000 lines...");

  for (let i = 0; i < 10000; i++) {
    if (i === targetLine) {
      lines.push(`Line ${i}: SECRET_CODE=${targetValue} // TOP SECRET`);
    } else {
      lines.push(`Line ${i}: ${generateRandomSentence()}`);
    }
  }

  const context = lines.join("\n");
  const query = "Find the SECRET_CODE value hidden in this document.";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 15,
    maxLLMCalls: 10,
    verbose: true, // Enable verbose to see the search strategy
  });

  console.log("\nQuery:", query);
  console.log("Context size:", context.length, "characters");
  console.log("Target line:", targetLine);
  console.log("Expected value:", targetValue);
  console.log("\nSearching...\n");

  try {
    const startTime = Date.now();
    const result = await agent.generate({
      context,
      query,
    });
    const duration = (Date.now() - startTime) / 1000;

    console.log("\n✓ Answer:", result.text);
    console.log("✓ Expected:", targetValue);
    console.log(
      "✓ Match:",
      result.text.includes(targetValue) ? "✅ YES" : "❌ NO",
    );
    console.log("✓ Duration:", duration.toFixed(2), "seconds");
    console.log("✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    console.log("\n--- Search Strategy Used ---");
    result.steps.slice(0, 3).forEach((step) => {
      console.log(`\nStep ${step.iteration}:`);
      console.log("Reasoning:", step.reasoning.substring(0, 120) + "...");
      console.log("Code:", step.code.substring(0, 100) + "...");
    });

    return result;
  } catch (error) {
    console.error("Error in Example 3:", error);
    throw error;
  }
}

// ============================================================================
// Example 4: Multi-step Analysis with Sub-LLM Queries
// ============================================================================

/**
 * Example: Complex analysis requiring sub-LLM semantic understanding
 * Demonstrates llm_query() usage for semantic tasks
 */
export async function example4SemanticAnalysis() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 4: Semantic Analysis with Sub-LLM Queries");
  console.log("=".repeat(60) + "\n");

  const context = `
    Customer Feedback Log:

    1. "The product is amazing! Best purchase I've made all year."
    2. "Terrible experience. Waste of money."
    3. "Works as expected. Nothing special but does the job."
    4. "Absolutely love it! Fast shipping too."
    5. "Disappointed. Broke after two weeks."
    6. "Good value for the price. Would recommend."
    7. "Not what I expected. Returning it."
    8. "Fantastic quality. Exceeded my expectations!"
    9. "Average product. Meh."
    10. "Horrible customer service. Never again."
  `;

  const query =
    "Analyze customer sentiment. Categorize each review as positive, negative, or neutral, and count totals.";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 12,
    maxLLMCalls: 15, // More calls needed for semantic analysis
    verbose: false,
  });

  console.log("Query:", query);
  console.log("Reviews to analyze: 10\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Answer:", result.text);
    console.log("✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    console.log("\n--- Analysis Approach ---");
    console.log(`Total steps: ${result.steps.length}`);

    // Show all steps
    result.steps.forEach((step) => {
      console.log(`\nStep ${step.iteration}:`);
      console.log("Reasoning:", step.reasoning.substring(0, 100) + "...");
      console.log("Code:", step.code.substring(0, 150) + "...");
    });

    // Highlight sub-LLM queries if any
    const subLLMSteps = result.steps.filter((step) =>
      step.code.includes("llm_query"),
    );
    if (subLLMSteps.length > 0) {
      console.log(`\n--- Sub-LLM Queries (${subLLMSteps.length}) ---`);
      subLLMSteps.forEach((step) => {
        console.log(`\nStep ${step.iteration}:`);
        console.log("Code:", step.code.substring(0, 200) + "...");
      });
    } else {
      console.log(
        "\n(No sub-LLM queries used - analysis done purely with code)",
      );
    }

    return result;
  } catch (error) {
    console.error("Error in Example 4:", error);
    throw error;
  }
}

// ============================================================================
// Example 5: Pattern Extraction
// ============================================================================

/**
 * Example: Extract patterns and structured data from text
 * Demonstrates regex and string manipulation in JavaScript
 */
export async function example5PatternExtraction() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 5: Pattern Extraction");
  console.log("=".repeat(60) + "\n");

  const context = `
    Log File Analysis:

    [2024-01-15 10:23:45] ERROR: Connection failed to db://prod-server-01
    [2024-01-15 10:24:12] INFO: Retrying connection...
    [2024-01-15 10:24:15] SUCCESS: Connected to db://prod-server-01
    [2024-01-15 11:45:30] ERROR: Timeout on api://user-service
    [2024-01-15 11:46:00] INFO: Fallback to api://user-service-backup
    [2024-01-15 12:00:00] ERROR: Disk space low on storage://data-node-03
    [2024-01-15 12:05:00] WARNING: Cleanup initiated on storage://data-node-03
    [2024-01-15 13:30:00] SUCCESS: Cleanup completed, 50GB freed

    Server Status: db://prod-server-01 (healthy), api://user-service (degraded), storage://data-node-03 (critical)
  `;

  const query =
    "Extract all ERROR entries with timestamps and server names. Summarize the issues found.";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 8,
    maxLLMCalls: 5,
    verbose: false,
  });

  console.log("Query:", query);
  console.log("Log entries: Multiple lines with timestamps\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Answer:", result.text);
    console.log("✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    console.log("\n--- Pattern Matching Code ---");
    result.steps.forEach((step) => {
      if (
        step.code.includes("match") ||
        step.code.includes("RegExp") ||
        step.code.includes("extract")
      ) {
        console.log(`\nStep ${step.iteration}:`);
        console.log("Code:\n", step.code);
      }
    });

    return result;
  } catch (error) {
    console.error("Error in Example 5:", error);
    throw error;
  }
}

// ============================================================================
// Example 6: Streaming Example
// ============================================================================

/**
 * Example: Stream the generation process
 * Shows how to use the stream() method
 */
export async function example6Streaming() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 6: Streaming");
  console.log("=".repeat(60) + "\n");

  const context = `
    Quick brown fox jumps over the lazy dog.
    The secret password is: HUNTER2
    Lorem ipsum dolor sit amet.
  `;

  const query = "What is the secret password?";

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 5,
    maxLLMCalls: 3,
    verbose: false,
  });

  console.log("Query:", query);
  console.log("Using stream() method...\n");

  try {
    const result = await agent.stream({
      context,
      query,
    });

    console.log("Answer received via stream:");

    // Read from textStream
    const reader = result.textStream.getReader();
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fullText += value;
      process.stdout.write(value);
    }

    console.log("\n\n✓ Full Answer:", fullText);
    console.log("✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    return result;
  } catch (error) {
    console.error("Error in Example 6:", error);
    throw error;
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

function generateRandomSentence(): string {
  const subjects = [
    "The system",
    "A user",
    "The server",
    "The application",
    "The database",
    "An API",
    "The network",
    "The service",
  ];
  const verbs = [
    "processed",
    "handled",
    "managed",
    "processed",
    "completed",
    "executed",
    "performed",
    "ran",
  ];
  const objects = [
    "a request",
    "data",
    "an operation",
    "a task",
    "a query",
    "a transaction",
    "a job",
    "a command",
  ];

  const subject = subjects[Math.floor(Math.random() * subjects.length)];
  const verb = verbs[Math.floor(Math.random() * verbs.length)];
  const object = objects[Math.floor(Math.random() * objects.length)];

  return `${subject} ${verb} ${object} successfully at timestamp ${Date.now()}`;
}

// ============================================================================
// Main Runner
// ============================================================================

/**
 * Run all examples
 */
export async function runAllExamples() {
  console.log("\n" + "=".repeat(60));
  console.log("RLM (Recursive Language Model) - Example Suite");
  console.log("JavaScript + vm2 Sandbox Implementation");
  console.log("=".repeat(60));

  const results = {
    example1: null as Awaited<
      ReturnType<typeof example1SimpleTextSearch>
    > | null,
    example2: null as Awaited<ReturnType<typeof example2DataAnalysis>> | null,
    example3: null as Awaited<
      ReturnType<typeof example3NeedleInHaystack>
    > | null,
    example4: null as Awaited<
      ReturnType<typeof example4SemanticAnalysis>
    > | null,
    example5: null as Awaited<
      ReturnType<typeof example5PatternExtraction>
    > | null,
    example6: null as Awaited<ReturnType<typeof example6Streaming>> | null,
  };

  try {
    // Run examples sequentially
    results.example1 = await example1SimpleTextSearch();
    results.example2 = await example2DataAnalysis();
    // Example 3 is slower (large context), skip in quick runs
    // results.example3 = await example3NeedleInHaystack();
    results.example4 = await example4SemanticAnalysis();
    results.example5 = await example5PatternExtraction();
    results.example6 = await example6Streaming();

    // Summary
    console.log("\n" + "=".repeat(60));
    console.log("Summary");
    console.log("=".repeat(60));
    console.log("\nExamples completed successfully!");
    console.log(
      "Total LLM calls across examples:",
      Object.values(results).reduce(
        (sum, r) => sum + (r?.llmCallCount || 0),
        0,
      ),
    );

    return results;
  } catch (error) {
    console.error("\nExample suite failed:", error);
    throw error;
  }
}

// Run if executed directly
if (import.meta.main) {
  runAllExamples().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

export default {
  example1SimpleTextSearch,
  example2DataAnalysis,
  example3NeedleInHaystack,
  example4SemanticAnalysis,
  example5PatternExtraction,
  example6Streaming,
  runAllExamples,
};
