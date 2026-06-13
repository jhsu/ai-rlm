/**
 * Book recommendation example using Exa search.
 *
 * This mirrors the fast-rlm `book_rec_with_exa.py` example for this package:
 * it exposes Exa and Goodreads search as sandbox tools, then lets RLM decide
 * how to fan out searches and rank recommendations from the gathered evidence.
 *
 * Required environment variables:
 *   EXA_API_KEY=...
 *   OPENAI_API_KEY=...
 *
 * Run:
 *   bun run examples/book-rec-with-exa.ts
 */

import { z } from "zod";
import { RLMAgent, type RLMToolSet } from "../src/rlm";
import { model, subModel } from "./model";

interface ExaSearchResult {
  title: string;
  url: string;
  highlights: string[];
  published_date: string;
}

interface GoodreadsReview {
  title: string;
  url: string;
  published_date: string;
  text: string;
}

const TopReadsSchema = z.object({
  picks: z
    .array(
      z.object({
        title: z.string(),
        why: z.string(),
      })
    )
    .length(10),
});

const TOPIC = `
Books released in 2026 that I would love. Make sure they are good books.
I don't like reading books that are anti-feminist, racist, or sexist.
I love reading good complex characters.
Last books I have loved are: The Palace of Illusions, Piranesi, Dark Matter, and Animal Farm.
Give me 10 book recommendations and state why.
`;

async function exaPost(path: "search" | "contents", body: unknown) {
  const apiKey = process.env.EXA_API_KEY;
  if (!apiKey) {
    throw new Error("Set EXA_API_KEY in your environment first");
  }

  const response = await fetch(`https://api.exa.ai/${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(
      `Exa ${path} request failed: ${response.status} ${await response.text()}`
    );
  }

  return response.json() as Promise<{ results?: Array<Record<string, unknown>> }>;
}

async function exaSearch({
  query,
  numResults = 5,
}: {
  query: string;
  numResults?: number;
}): Promise<ExaSearchResult[]> {
  const data = await exaPost("search", {
    query,
    type: "auto",
    numResults,
    contents: { highlights: true },
  });

  return (data.results ?? []).slice(0, numResults).map((result) => ({
    title: typeof result.title === "string" ? result.title : "",
    url: typeof result.url === "string" ? result.url : "",
    highlights: Array.isArray(result.highlights)
      ? result.highlights.filter((value): value is string => typeof value === "string")
      : [],
    published_date:
      typeof result.publishedDate === "string" ? result.publishedDate : "",
  }));
}

async function goodreadsReviews({
  query,
  numReviews = 5,
}: {
  query: string;
  numReviews?: number;
}): Promise<GoodreadsReview[]> {
  const searchData = await exaPost("search", {
    query,
    type: "auto",
    numResults: numReviews,
    includeDomains: ["goodreads.com"],
  });
  const results = (searchData.results ?? []).slice(0, numReviews);
  const urls = results
    .map((result) => result.url)
    .filter((url): url is string => typeof url === "string" && url.length > 0);

  if (urls.length === 0) {
    return [];
  }

  const contentsData = await exaPost("contents", {
    urls,
    text: true,
  });
  const textByUrl = new Map(
    (contentsData.results ?? []).map((result) => [
      typeof result.url === "string" ? result.url : "",
      typeof result.text === "string" ? result.text : "",
    ])
  );

  return results
    .filter((result): result is Record<string, unknown> & { url: string } =>
      typeof result.url === "string"
    )
    .map((result) => ({
      title: typeof result.title === "string" ? result.title : "",
      url: result.url,
      published_date:
        typeof result.publishedDate === "string" ? result.publishedDate : "",
      text: (textByUrl.get(result.url) ?? "").slice(0, 8000),
    }));
}

export async function bookRecWithExa() {
  if (!process.env.EXA_API_KEY) {
    throw new Error("Set EXA_API_KEY in your environment first");
  }

  console.log("Running RLM with Exa and Goodreads tools...");

  const rlmTools: RLMToolSet = {
    exa_search: {
      description:
        "Search Exa for high-signal articles or pages matching a query. Returns title, url, highlights, and published_date.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
          numResults: { type: "number", default: 5 },
        },
        required: ["query"],
      },
      execute: (input) =>
        exaSearch(input as { query: string; numResults?: number }),
    },
    goodreads_reviews: {
      description:
        "Search Goodreads via Exa and return review/page text. Returns title, url, published_date, and text.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
          numReviews: { type: "number", default: 5 },
        },
        required: ["query"],
      },
      execute: (input) =>
        goodreadsReviews(input as { query: string; numReviews?: number }),
    },
  };

  const agent = new RLMAgent({
    model,
    subModel,
    maxDepth: 2,
    maxIterations: 8,
    maxLLMCalls: 15,
    maxOutputChars: 200000,
    rlmTools,
  });

  const result = await agent.generate({
    options: { context: TOPIC },
    prompt: `
Launch tool calls to search and explore. Divide up the work between searches.
Use Promise.all where useful to run independent searches concurrently.
Use tools.exa_search and tools.goodreads_reviews to gather evidence.

Recommend exactly 10 books for the user's preferences in context.
Prefer 2026 releases when supported by evidence. Avoid books that evidence suggests are anti-feminist, racist, or sexist.
Favor books with complex characters and explain fit relative to The Palace of Illusions, Piranesi, Dark Matter, and Animal Farm.

Return only JSON matching this schema:
{
  "picks": [
    { "title": "Book title", "why": "One evidence-grounded sentence." }
  ]
}
`,
  });

  const parsed = TopReadsSchema.safeParse(JSON.parse(result.text));
  if (!parsed.success) {
    console.log("Raw result:", result.text);
    throw new Error(`Result did not match expected schema: ${parsed.error.message}`);
  }

  console.log("\n=== TOP READS ===\n");
  parsed.data.picks.forEach((pick, index) => {
    console.log(`${index + 1}. ${pick.title}`);
    console.log(`   why: ${pick.why}\n`);
  });

  console.log("USAGE:", result.usage);
  return parsed.data;
}

if (import.meta.main) {
  bookRecWithExa().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
