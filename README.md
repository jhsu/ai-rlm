# ai-rlm

[![npm](https://img.shields.io/npm/v/ai-rlm?style=for-the-bdage)](https://www.npmjs.com/package/ai-rlm)

RLM (Recursive Language Model) provided via ai-sdk Agent or tool.

Based on the paper "Recursive Language Models" by Zhang, Kraska, and Khattab (2025).

## Overview

RLM is an inference strategy where LLMs treat long contexts as part of an external environment rather than feeding them directly to the model. The LLM writes JavaScript code to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

### Key Features

- **Iterative Code Execution**: The model writes JavaScript code, sees output, then writes more code
- **Sub-LLM Queries**: Access to `llm_query()` and `llm_query_batched()` for semantic analysis
- **Context Management**: Efficient handling of large contexts through chunking
- **Sandboxed REPL**: JavaScript execution in a sandboxed QuickJS WebAssembly context
- **AI SDK Integration**: Works as an Agent or Tool with the Vercel AI SDK
- **Multiple Usage Patterns**: Use as standalone agent or as a tool in larger workflows

## Installation

```bash
npm install ai-rlm ai zod @ai-sdk/openai
```

`ai` and `zod` are peer dependencies and must be installed in your project.

The `model` and `subModel` settings accept any AI SDK `LanguageModel` — use any provider ([OpenAI](https://sdk.vercel.ai/providers/ai-sdk-providers/openai), [Anthropic](https://sdk.vercel.ai/providers/ai-sdk-providers/anthropic), [Google](https://sdk.vercel.ai/providers/ai-sdk-providers/google-generative-ai), etc.).

## Usage

### As Agent (Recommended)

The **RLMAgent** class provides a clean, agent-based API that integrates seamlessly with the AI SDK:

```typescript
import { RLMAgent } from 'ai-rlm';
import { openai } from '@ai-sdk/openai';

// Create agent
const agent = new RLMAgent({
  model: openai('gpt-4.1'),              // Root agent model
  subModel: openai('gpt-4.1-mini'),      // Sub-LLM model for queries
  maxIterations: 20,                      // Max REPL iterations
  maxLLMCalls: 50,                        // Max sub-LLM calls
});

// Process a context
const context = `
  The quick brown fox jumps over the lazy dog.
  The magic number is 42.
`;

const query = 'What is the magic number?';

const result = await agent.generate({
  context,
  query,
});

console.log('Answer:', result.text);
console.log('Iterations:', result.iterations);
console.log('LLM Calls:', result.llmCallCount);
console.log('Steps:', result.steps); // Full trajectory
```

### As Tool

Use **createRLMTool** to create an AI SDK-compatible tool for use with `generateText` or `ToolLoopAgent`:

```typescript
import { createRLMTool } from 'ai-rlm';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

// Create the tool
const rlmTool = createRLMTool({
  model: openai('gpt-4.1'),
  subModel: openai('gpt-4.1-mini'),
});

// Use in generateText
const result = await generateText({
  model: openai('gpt-4.1'),
  tools: { analyzeLargeContext: rlmTool },
  prompt: 'Analyze this large codebase for security vulnerabilities',
});
```

### With ToolLoopAgent

```typescript
import { ToolLoopAgent } from 'ai';
import { createRLMTool } from 'ai-rlm';
import { openai } from '@ai-sdk/openai';

const agent = new ToolLoopAgent({
  model: openai('gpt-4.1'),
  tools: {
    analyzeLargeContext: createRLMTool({
      model: openai('gpt-4.1'),
      subModel: openai('gpt-4.1-mini'),
    }),
    // ... other tools
  },
});

const result = await agent.generate({
  prompt: 'Check this document for compliance issues',
});
```

### Streaming Support

```typescript
const stream = await agent.stream({
  context: largeDocument,
  query: 'Analyze this',
});

// Read from the stream
const reader = stream.textStream.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  process.stdout.write(value);
}
```

## How It Works

The RLM agent writes JavaScript code to explore the context in an iterative loop:

```javascript
// First, explore the context
console.log('Context length:', context.length);
console.log('First 200 chars:', context.substring(0, 200));

// Search for specific patterns
const lines = context.split('\n');
const targetLine = lines.find(line => line.includes('magic number'));
console.log('Found:', targetLine);

// Store result for later
const answer = targetLine?.match(/magic number is (\d+)/)?.[1];

// Submit answer
FINAL_VAR(answer)
```

1. **Context Loading**: The context is loaded into a sandboxed JavaScript REPL environment
2. **Iterative Reasoning**: The root LLM writes JavaScript code to explore the context
3. **Code Execution**: Code is executed in a QuickJS WebAssembly sandbox with a 30s timeout
4. **Sub-LLM Queries**: For semantic analysis, `llm_query()` delegates to a sub-model
5. **Result Accumulation**: The model iterates until it finds an answer
6. **Final Answer**: The model submits an answer using `FINAL(answer)` or `FINAL_VAR(variable_name)`

### System Prompt

The RLM system prompt instructs the model to:
- EXPLORE FIRST - Look at data before processing
- ITERATE - Write small code snippets, observe outputs
- VERIFY BEFORE SUBMITTING - Check results are correct
- USE llm_query FOR SEMANTICS - Code finds WHERE; LLM understands WHAT
- CHUNK SMARTLY - Feed substantial chunks to sub-LLMs (~500K chars)

## REPL Sandbox

The JavaScript REPL runs code in a QuickJS WebAssembly sandboxed context:

### Available in the Sandbox:

- **`context`**: The input context (string or object)
- **`console.log()` / `console.error()`**: Output logging
- **`llm_query(prompt)`**: Query a sub-LLM for semantic analysis
- **`llm_query_batched(prompts)`**: Query multiple sub-LLMs
- **`FINAL(answer)`**: Submit final answer directly
- **`FINAL_VAR(varName)`**: Submit a variable from the REPL
- **Standard JavaScript**: All ES6+ features, Array methods, String methods, Math, JSON, etc.

### Security Features:

- 30-second timeout on code execution
- No access to Node.js built-in modules or file system
- No network access
- Sandboxed console output capture

## API Reference

### RLMAgent

The primary class for using RLM as an agent.

#### `constructor(settings: RLMAgentSettings)`

```typescript
import type { LanguageModel } from 'ai';

interface RLMAgentSettings {
  model: LanguageModel;     // Required: Root agent model
  subModel?: LanguageModel; // Optional: Sub-LLM model (defaults to model)
  maxIterations?: number;   // Max REPL iterations (default: 20)
  maxLLMCalls?: number;     // Max sub-LLM calls (default: 50)
  maxOutputChars?: number;  // Max REPL output chars (default: 100000)
  verbose?: boolean;        // Enable verbose logging (default: false)
}
```

#### `async generate(options): Promise<RLMGenerateResult>`

Generate an answer by iteratively analyzing the context.

**Parameters:**
```typescript
interface RLMAgentCallParameters {
  context: RLMContext;                    // The large context to analyze
  query: string;                          // The question or task
  abortSignal?: AbortSignal;              // Optional abort signal
  timeout?: number;                       // Optional timeout in ms
  onStepFinish?: (step: REPLStep) => void; // Callback for each step
}
```

**Returns:**
```typescript
interface RLMGenerateResult {
  text: string;             // The generated answer
  steps: REPLStep[];        // Array of REPL steps taken
  llmCallCount: number;     // Total LLM calls made
  iterations: number;       // Total iterations performed
}

interface REPLStep {
  iteration: number;
  reasoning: string;        // The model's reasoning before code
  code: string;             // JavaScript code executed
  output: string;           // Console output and results
}
```

#### `async stream(options): Promise<RLMStreamResult>`

Stream the answer generation process.

**Returns:**
```typescript
interface RLMStreamResult extends RLMGenerateResult {
  textStream: ReadableStream<string>;  // Readable stream of text
}
```

### createRLMTool

Factory function to create RLM as an AI SDK-compatible tool.

#### `createRLMTool(config?: RLMToolConfig)`

```typescript
import type { LanguageModel } from 'ai';

function createRLMTool(config?: {
  model?: LanguageModel;    // Root agent model
  subModel?: LanguageModel; // Sub-LLM model
  maxIterations?: number;   // Max iterations (default: 20)
  maxLLMCalls?: number;     // Max LLM calls (default: 50)
  maxOutputChars?: number;  // Max output chars (default: 100000)
}): Tool
```

**Tool Input Schema:**
```typescript
{
  context: string | string[] | Record<string, unknown>;
  query: string;
  maxIterations?: number;   // Optional override
  maxLLMCalls?: number;     // Optional override
}
```

**Tool Output:**
```typescript
{
  answer: string;           // The generated answer
  iterations: number;       // Number of iterations
  stepsTaken: number;       // Number of steps executed
}
```

### RLMContext

Context can be any of these formats:
```typescript
type RLMContext = string | string[] | Record<string, unknown>;
```

- `string`: Raw text document
- `string[]`: Array of lines or documents
- `Record<string, unknown>`: JSON/structured data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RLMAgent Class                         │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │              REPL Environment (QuickJS)               │  │
│  │  - Sandboxed JavaScript execution                     │  │
│  │  - llm_query() for sub-LLM semantic analysis          │  │
│  │  - 30s timeout protection                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              generate() Method                        │  │
│  │  1. Generate reasoning + JS code                      │  │
│  │  2. Execute in sandboxed context                      │  │
│  │  3. Process llm_query markers → real LLM calls        │  │
│  │  4. Check for FINAL() answer                          │  │
│  │  5. Repeat or return answer                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              stream() Method                          │  │
│  │  - Same as generate() with streaming                  │  │
│  │  - Returns ReadableStream for real-time output        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ createRLMTool()
                              ▼
                    ┌──────────────────────┐
                    │    AI SDK Tool        │
                    │ - Tool interface      │
                    │ - Input validation    │
                    │ - Auto-execution      │
                    └──────────────────────┘
```

## Examples

Run the examples:

```bash
# Basic agent examples
bun run examples/basic-usage.ts

# Tool integration examples
bun run examples/tool-usage.ts

# Individual examples
bun run -e "import { example1SimpleTextSearch } from './examples/basic-usage.ts'; example1SimpleTextSearch()"
```

### Example Files

- **`examples/basic-usage.ts`**: Agent API examples (generate, stream, callbacks)
- **`examples/tool-usage.ts`**: Tool API examples (with generateText, ToolLoopAgent)
- **`examples/document-comparison.ts`**: Document diffing example
- **`examples/data-transformation.ts`**: Data extraction and transformation

## License

MIT

## References

- Paper: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
- AI SDK Documentation: https://sdk.vercel.ai/docs
