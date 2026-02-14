/**
 * RLM - Data Transformation Example
 *
 * This example demonstrates using RLMAgent to transform unstructured data
 * into structured formats (CSV, JSON, etc.)
 */

import { RLMAgent } from "../src/rlm";

/**
 * Extract and transform messy log data into structured format
 */
export async function dataTransformationExample() {
  console.log("\n" + "=".repeat(60));
  console.log("Data Transformation Example");
  console.log("=".repeat(60) + "\n");

  // Messy unstructured data
  const context = `
    Raw Customer Feedback Dump:

    John Doe john@email.com says: "Love the product! 5 stars"
    Rating: 5/5 Date: 2024-01-15

    Contact: Jane Smith jane.smith@company.org
    Comment: "Good but expensive" Score 3 out of 5 on Jan 20th

    From: Mike Johnson mike.j@yahoo.com
    Message: "Terrible experience, broken on arrival" Rating: 1/5 2024-02-01

    Sarah Williams (sarah.w@outlook.com) wrote on 2024-02-10:
    "Amazing quality, fast shipping!"
    Rating: 5/5

    Email: tom.brown@gmail.com
    Feedback: "Average product, nothing special"
    Score: 3/5 Date: Feb 15, 2024
  `;

  const query =
    "Transform this messy feedback data into a structured JSON array with fields: name, email, rating (number), comment, date (ISO format). Clean and normalize the data.";

  const agent = new RLMAgent({
    model: "gpt-4.1",
    subModel: "gpt-4.1-mini",
    maxIterations: 15,
    maxLLMCalls: 8,
    verbose: false,
  });

  console.log("Transforming unstructured feedback data...\n");
  console.log("Input sample:", context.substring(0, 200) + "...\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Transformation Complete!\n");
    console.log("Structured Output:\n", result.text);
    console.log("\n✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    // Show the extraction strategy
    console.log("\n--- Extraction Steps ---");
    result.steps.forEach((step, idx) => {
      if (
        step.code.includes("match") ||
        step.code.includes("split") ||
        step.code.includes("extract")
      ) {
        console.log(`\nStep ${step.iteration}:`);
        console.log("Extraction code:", step.code.substring(0, 120) + "...");
      }
    });

    return result;
  } catch (error) {
    console.error("Error in data transformation:", error);
    throw error;
  }
}

/**
 * Convert markdown notes to structured meeting minutes
 */
export async function meetingMinutesExample() {
  console.log("\n" + "=".repeat(60));
  console.log("Meeting Minutes Extraction Example");
  console.log("=".repeat(60) + "\n");

  const context = `
    Meeting Notes - March 15, 2024

    Attendees: Alice (PM), Bob (Dev), Carol (Design), Dave (QA)

    ## Action Items
    - Alice: Prepare roadmap for Q2 by March 22
    - Bob: Fix critical bug #1234 - payment processing error
    - Carol: Update design system colors for accessibility
    - Dave: Write test cases for new checkout flow

    ## Decisions Made
    1. Move release date from April 1 to April 15
    2. Drop feature: advanced analytics for v2.0
    3. Adopt new payment provider: Stripe
    4. Budget approved: $50K for marketing campaign

    ## Discussion Notes
    Alice said we need better user onboarding. Bob agreed and mentioned the
    current drop-off rate is 40%. Carol suggested adding a tutorial wizard.
    Everyone liked that idea.

    Dave raised concerns about testing time. Alice said we can extend sprint
    by one week. Bob worried about burnout but agreed if we get pizza.

    Next meeting: March 22, 2024 at 2pm
  `;

  const query =
    "Extract structured meeting minutes with: attendees list, action items (who/what/deadline), decisions made, and key discussion points. Format as clean JSON.";

  const agent = new RLMAgent({
    model: "gpt-4.1",
    subModel: "gpt-4.1-mini",
    maxIterations: 10,
    maxLLMCalls: 5,
    verbose: false,
  });

  console.log("Extracting structured meeting minutes...\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Extraction Complete!\n");
    console.log("Structured Minutes:\n", result.text);
    console.log("\n✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    return result;
  } catch (error) {
    console.error("Error in meeting minutes extraction:", error);
    throw error;
  }
}

/**
 * Run all transformation examples
 */
export async function runAllTransformationExamples() {
  console.log("\n" + "=".repeat(60));
  console.log("RLM Data Transformation Examples");
  console.log("=".repeat(60));

  await dataTransformationExample();
  await meetingMinutesExample();

  console.log("\n" + "=".repeat(60));
  console.log("All transformation examples completed!");
  console.log("=".repeat(60) + "\n");
}

// Run if executed directly
if (import.meta.main) {
  runAllTransformationExamples().catch(console.error);
}

export default {
  dataTransformationExample,
  meetingMinutesExample,
  runAllTransformationExamples,
};
