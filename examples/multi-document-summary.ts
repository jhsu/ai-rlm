/**
 * RLM - Multi-Document Summary Example
 *
 * This example demonstrates using RLMAgent to summarize several related
 * documents and produce a final cross-document summary.
 */

import { RLMAgent } from "../src/rlm";
import { model, subModel } from "./model";

type SourceDocument = {
  id: string;
  title: string;
  text: string;
};

const documents: SourceDocument[] = [
  {
    id: "support-q1",
    title: "Q1 Support Review",
    text: `
      Support volume rose 18% in Q1, mostly from onboarding questions after the
      self-serve launch. The top ticket categories were workspace setup,
      billing-plan changes, and CSV import errors. Median first response time
      improved from 7 hours to 3.5 hours after the team added a rotating triage
      shift. Customers continued to ask for clearer import error messages and a
      way to preview billing changes before confirming plan updates.
    `,
  },
  {
    id: "product-roadmap",
    title: "Product Roadmap Notes",
    text: `
      The product team plans to ship an import preview screen, richer CSV
      validation, and an account-level billing simulator in Q2. Engineering
      wants to keep the import work small by validating the first 1,000 rows
      before upload. The billing simulator is blocked on pricing-service API
      changes. Product also wants to run a guided onboarding experiment for new
      workspace admins.
    `,
  },
  {
    id: "sales-feedback",
    title: "Sales Feedback Digest",
    text: `
      Sales calls show strong demand from mid-market accounts, but prospects
      often hesitate when they cannot estimate seat expansion costs. Several
      deals were delayed because buyers wanted proof that large CSV imports
      would succeed before migration. Account executives asked for clearer
      onboarding collateral, especially for workspace administrators who are
      migrating teams from spreadsheets.
    `,
  },
];

/**
 * Summarize each source document, then produce a final synthesis that connects
 * the themes across all documents.
 */
export async function multiDocumentSummaryExample() {
  console.log("\n" + "=".repeat(60));
  console.log("Multi-Document Summary Example");
  console.log("=".repeat(60) + "\n");

  const query = [
    "Summarize each document separately in 2-3 bullets.",
    "Then provide a final summary at the end that synthesizes the recurring",
    "themes, strongest customer needs, and recommended next actions across all documents.",
    "Use document titles when referring to sources.",
  ].join(" ");

  const agent = new RLMAgent({
    model,
    subModel,
    maxIterations: 10,
    maxLLMCalls: 8,
  });

  console.log(`Summarizing ${documents.length} documents...\n`);
  documents.forEach((document) => {
    console.log(`- ${document.title} (${document.id})`);
  });
  console.log("\nQuery:", query, "\n");

  try {
    const result = await agent.generate({
      prompt: query,
      options: { context: { documents } },
    });

    const rlmData = result.output;

    console.log("Summary Complete!\n");
    console.log(result.text);
    console.log("\nIterations:", rlmData.iterations);
    console.log("LLM Calls:", rlmData.llmCallCount);

    console.log("\n--- Summary Strategy ---");
    rlmData.steps.forEach((step) => {
      const preview = step.code.replace(/\s+/g, " ").trim().slice(0, 140);
      console.log(`Step ${step.iteration}: ${preview}`);
    });

    return result;
  } catch (error) {
    console.error("Error in multi-document summary:", error);
    throw error;
  }
}

if (import.meta.main) {
  multiDocumentSummaryExample().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

export default multiDocumentSummaryExample;
