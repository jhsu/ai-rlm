/**
 * RLM - Document Comparison Example
 *
 * This example demonstrates using RLMAgent to compare two versions of a document
 * and identify changes, additions, and deletions.
 */

import { RLMAgent } from "../src/rlm";

/**
 * Compare two versions of a policy document
 */
export async function documentComparisonExample() {
  console.log("\n" + "=".repeat(60));
  console.log("Document Comparison Example");
  console.log("=".repeat(60) + "\n");

  const context = {
    version1: `
      Company Privacy Policy - v1.0 (January 2024)

      1. Data Collection
      We collect your name, email, and phone number during registration.

      2. Data Usage
      Your data is used to provide our services and send occasional updates.

      3. Data Sharing
      We do not share your data with third parties.

      4. Contact
      For questions, contact: support@example.com
    `,
    version2: `
      Company Privacy Policy - v2.0 (March 2024)

      1. Data Collection
      We collect your name, email, phone number, and location during registration.
      We also collect usage analytics to improve our service.

      2. Data Usage
      Your data is used to provide our services, send updates, and personalize content.
      Analytics data helps us understand user behavior.

      3. Data Sharing
      We may share anonymized data with analytics providers.
      We do not share your personal data with third parties for marketing.

      4. New Section: Data Retention
      We retain your data for 2 years after account deletion.

      5. Contact
      For questions, contact: privacy@example.com or support@example.com
      Phone: +1-800-123-4567
    `,
  };

  const query =
    "Compare these two policy versions. Identify all additions, deletions, and changes. Present as a structured diff.";

  const agent = new RLMAgent({
    model: "gpt-4.1",
    subModel: "gpt-4.1-mini",
    maxIterations: 12,
    maxLLMCalls: 8,
    verbose: false,
  });

  console.log("Comparing two versions of a privacy policy...\n");

  try {
    const result = await agent.generate({
      context,
      query,
    });

    console.log("✓ Comparison Complete!\n");
    console.log("Answer:\n", result.text);
    console.log("\n✓ Iterations:", result.iterations);
    console.log("✓ LLM Calls:", result.llmCallCount);

    // Show the strategy used
    console.log("\n--- Comparison Strategy ---");
    result.steps.forEach((step) => {
      console.log(`\nStep ${step.iteration}:`);
      if (step.code.includes("version1") || step.code.includes("version2")) {
        console.log("Processing code:", step.code.substring(0, 100) + "...");
      }
    });

    return result;
  } catch (error) {
    console.error("Error in document comparison:", error);
    throw error;
  }
}

/**
 * Run the example
 */
if (import.meta.main) {
  documentComparisonExample().catch(console.error);
}

export default documentComparisonExample;
