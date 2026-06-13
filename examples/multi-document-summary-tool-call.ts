/**
 * RLM - Multi-Document Summary Tool Call Example
 *
 * This example demonstrates using an RLM agent inside an AI SDK tool call. The
 * outer model passes a document query to the tool; the tool retrieves matching
 * documents from its own corpus, runs RLMAgent on those documents, and returns
 * the summary for the outer model to synthesize.
 */

import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { RLMAgent } from "../src/rlm";
import { model, subModel } from "./model";

type SourceDocument = {
  id: string;
  title: string;
  text: string;
};

const documentCorpus: SourceDocument[] = [
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

function retrieveDocuments(documentQuery: string): SourceDocument[] {
  const queryTerms = documentQuery
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((term) => term.length > 2);

  const scoredDocuments = documentCorpus.map((document) => {
    const searchableText =
      `${document.title} ${document.id} ${document.text}`.toLowerCase();
    const score = queryTerms.reduce(
      (total, term) => total + (searchableText.includes(term) ? 1 : 0),
      0,
    );

    return { document, score };
  });

  const matches = scoredDocuments
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .map(({ document }) => document);

  return matches.length > 0 ? matches : documentCorpus;
}

export async function multiDocumentSummaryToolCallExample() {
  console.log("\n" + "=".repeat(60));
  console.log("Multi-Document Summary Tool Call Example");
  console.log("=".repeat(60) + "\n");

  const summarizeDocuments = tool({
    description: [
      "Retrieve source documents by query and summarize them with an RLM agent.",
      "Use this when the caller knows what documents it wants but should not pass the documents directly.",
    ].join(" "),
    inputSchema: z.object({
      documentQuery: z
        .string()
        .describe(
          "A search query describing which documents or topics to retrieve.",
        ),
      summaryTask: z
        .string()
        .describe("The summarization and synthesis instructions to run."),
    }),
    execute: async ({ documentQuery, summaryTask }, { abortSignal }) => {
      const retrievedDocuments = retrieveDocuments(documentQuery);
      const rlmTask = [
        summaryTask,
        "",
        "Decompose this task before the final synthesis:",
        "- Summarize each document independently first.",
        "- Prefer llm_query_batched(...) with one prompt per document, or sub_rlm(...) per document when a document needs its own recursive analysis.",
        "- Then synthesize the per-document summaries into the final answer.",
        "- Do not pass the full retrieved corpus to a single llm_query unless only one document was retrieved.",
      ].join("\n");

      const agent = new RLMAgent({
        model,
        subModel,
        maxIterations: 12,
        maxLLMCalls: 20,
        maxDepth: 3,
      });

      const result = await agent.generate({
        prompt: rlmTask,
        options: {
          context: {
            documentQuery,
            summaryTask,
            documents: retrievedDocuments,
          },
        },
        abortSignal,
      });

      return {
        answer: result.text,
        documentCount: retrievedDocuments.length,
        documentTitles: retrievedDocuments.map((document) => document.title),
        iterations: result.output.iterations,
        stepsTaken: result.output.steps.length,
        maxDepth: 3,
        codeSteps: result.output.steps.map((step) => ({
          iteration: step.iteration,
          reasoning: step.reasoning,
          code: step.code,
        })),
      };
    },
  });

  const tools = { summarizeDocuments };

  const documentQuery =
    "customer onboarding, billing transparency, CSV import reliability, support trends, roadmap, and sales feedback";
  const summaryTask = [
    "Summarize each retrieved document separately in 2-3 bullets.",
    "Then provide a final synthesis of recurring themes, strongest customer needs, and recommended next actions.",
    "Use document titles when referring to sources.",
  ].join(" ");

  console.log("Delegating document retrieval and summarization to the RLM tool.");
  console.log("\nDocument query:", documentQuery);
  console.log("Summary task:", summaryTask, "\n");

  try {
    const result = await generateText({
      model,
      tools,
      stopWhen: stepCountIs(2),
      prepareStep: ({ stepNumber }) =>
        stepNumber === 0
          ? {
              toolChoice: {
                type: "tool",
                toolName: "summarizeDocuments",
              },
            }
          : {
              toolChoice: "none",
            },
      system:
        "You are a concise analyst. Use the document tool to retrieve and summarize source documents, then synthesize the tool result.",
      prompt: [
        "Use summarizeDocuments for this request.",
        `Document query: ${documentQuery}`,
        `Summary task: ${summaryTask}`,
        "After the tool returns, write a concise final answer for an executive reader.",
      ].join("\n"),
    });

    console.log("Final Answer:\n");
    console.log(result.text);

    console.log("\n--- Tool Calls ---");
    result.steps.forEach((step, index) => {
      step.toolCalls.forEach((call) => {
        console.log(`Step ${index + 1}: ${call.toolName}`);
        console.log("Input:", JSON.stringify(call.input, null, 2));
      });
    });

    console.log("\n--- RLM Tool Results ---");
    result.steps.forEach((step, index) => {
      step.toolResults.forEach((toolResult) => {
        const output = toolResult.output as {
          answer?: string;
          documentCount?: number;
          documentTitles?: string[];
          iterations?: number;
          stepsTaken?: number;
          maxDepth?: number;
          codeSteps?: Array<{
            iteration: number;
            reasoning: string;
            code: string;
          }>;
        };

        console.log(`Step ${index + 1}: ${toolResult.toolName}`);
        console.log("Retrieved Documents:", output.documentTitles?.join(", "));
        console.log("Document Count:", output.documentCount);
        console.log("RLM Answer:", output.answer);
        console.log("RLM Iterations:", output.iterations);
        console.log("RLM Steps Taken:", output.stepsTaken);
        console.log("RLM Max Depth:", output.maxDepth);

        console.log("\n--- RLM Code Executed ---");
        output.codeSteps?.forEach((codeStep) => {
          console.log(`\nRLM Iteration ${codeStep.iteration}`);
          console.log("Reasoning:", codeStep.reasoning);
          console.log("Code:\n" + codeStep.code);
        });
      });
    });

    return result;
  } catch (error) {
    console.error("Error in multi-document summary tool call:", error);
    throw error;
  }
}

if (import.meta.main) {
  multiDocumentSummaryToolCallExample().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

export default multiDocumentSummaryToolCallExample;
