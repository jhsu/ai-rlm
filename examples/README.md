# RLM Examples

This folder contains practical examples demonstrating how to use the Recursive Language Model (RLM) implementation with the **RLMAgent** and **Tool** APIs.

## Running Examples

### Run all basic agent examples
```bash
bun run examples/basic-usage.ts
```

### Run tool integration examples
```bash
bun run examples/tool-usage.ts
```

### Run individual examples
```bash
# Agent Examples
bun run -e "import { example1SimpleTextSearch } from './examples/basic-usage.ts'; example1SimpleTextSearch()"
bun run -e "import { example2DataAnalysis } from './examples/basic-usage.ts'; example2DataAnalysis()"
bun run -e "import { example3NeedleInHaystack } from './examples/basic-usage.ts'; example3NeedleInHaystack()"
bun run -e "import { example4SemanticAnalysis } from './examples/basic-usage.ts'; example4SemanticAnalysis()"
bun run -e "import { example5PatternExtraction } from './examples/basic-usage.ts'; example5PatternExtraction()"
bun run -e "import { example6Streaming } from './examples/basic-usage.ts'; example6Streaming()"

# Tool Examples
bun run -e "import { example1ToolWithGenerateText } from './examples/tool-usage.ts'; example1ToolWithGenerateText()"
bun run -e "import { example2ToolWithAgent } from './examples/tool-usage.ts'; example2ToolWithAgent()"

# Other Examples
bun run examples/document-comparison.ts
bun run examples/data-transformation.ts
```

## Examples Overview

### Agent Examples (`basic-usage.ts`)

#### Example 1: Simple Text Search
**Function**: `example1SimpleTextSearch()`

Demonstrates basic text searching within a document using JavaScript string methods.

**Key concepts**:
- String searching (`indexOf`, `includes`)
- Simple extraction
- RLMAgent basic workflow with `generate()`

#### Example 2: Data Analysis (JSON Processing)
**Function**: `example2DataAnalysis()`

Shows how to analyze structured JSON data with array methods.

**Key concepts**:
- Working with objects and arrays
- `Array.reduce()` for calculations
- Data aggregation
- Using structured context (JSON objects)

#### Example 3: Needle in Haystack
**Function**: `example3NeedleInHaystack()`

Finds a specific value hidden in a large document (10,000 lines).

**Key concepts**:
- Efficient chunking strategies
- Large context handling
- Search optimization
- Performance metrics

#### Example 4: Semantic Analysis with Sub-LLM Queries
**Function**: `example4SemanticAnalysis()`

Analyzes customer sentiment using `llm_query()` for semantic understanding.

**Key concepts**:
- `llm_query()` usage
- Semantic tasks requiring LLM understanding
- Multiple sub-LLM calls

#### Example 5: Pattern Extraction
**Function**: `example5PatternExtraction()`

Extracts structured data from log files using regex.

**Key concepts**:
- Regular expressions in JavaScript
- Pattern matching
- Data extraction and summarization

#### Example 6: Streaming
**Function**: `example6Streaming()`

Demonstrates the `stream()` method for real-time output.

**Key concepts**:
- Streaming API
- `textStream` consumption
- Real-time result processing

### Tool Examples (`tool-usage.ts`)

#### Example 1: Tool with generateText
**Function**: `example1ToolWithGenerateText()`

Shows how to use the RLM tool with AI SDK's `generateText` function.

**Key concepts**:
- `createRLMTool()` factory
- Tool integration with `generateText`
- Automatic tool invocation

#### Example 2: Tool with ToolLoopAgent
**Function**: `example2ToolWithAgent()`

Demonstrates using the RLM tool within a `ToolLoopAgent`.

**Key concepts**:
- `ToolLoopAgent` integration
- Multiple tools in an agent
- Agent instructions and context

### Document Comparison (`document-comparison.ts`)

Compares two versions of a policy document and identifies changes.

**Key concepts**:
- Document diffing
- Change detection (additions/deletions)
- Structured comparison output

### Data Transformation (`data-transformation.ts`)

Transforms unstructured data (messy logs, notes) into structured formats.

**Key concepts**:
- Pattern extraction with regex
- Data normalization
- JSON/CSV transformation
- Meeting minutes extraction

## Example Output

Each example prints:
- ✓ The query being asked
- ✓ The final answer (`result.text`)
- ✓ Number of iterations performed (`result.iterations`)
- ✓ Number of LLM calls made (`result.llmCallCount`)
- ✓ Execution trajectory (`result.steps`)

## Creating Your Own Examples

### Using RLMAgent (Recommended)

```typescript
import { RLMAgent } from '../src/rlm';

async function myExample() {
  const context = 'Your data here...';
  const query = 'Your question here?';
  
  const agent = new RLMAgent({
    model: 'gpt-4.1',
    subModel: 'gpt-4.1-mini',
    maxIterations: 10,
    maxLLMCalls: 10,
    verbose: false,
  });
  
  // Use generate()
  const result = await agent.generate({
    context,
    query,
  });
  
  console.log('Answer:', result.text);
  console.log('Iterations:', result.iterations);
  console.log('LLM Calls:', result.llmCallCount);
  
  // Or use stream()
  const stream = await agent.stream({ context, query });
  const reader = stream.textStream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    process.stdout.write(value);
  }
}

myExample();
```

### Using as a Tool

```typescript
import { createRLMTool } from '../src/rlm-tool';
import { generateText } from 'ai';

async function myToolExample() {
  const rlmTool = createRLMTool({
    model: 'gpt-4.1',
    subModel: 'gpt-4.1-mini',
  });
  
  const result = await generateText({
    model: 'gpt-4.1',
    tools: { analyzeLargeContext: rlmTool },
    prompt: 'Analyze this large codebase for issues...',
  });
  
  console.log('Result:', result.text);
}

myToolExample();
```

## Tips

1. **Start with `verbose: false`** for cleaner output, then enable for debugging
2. **Adjust `maxIterations`** based on expected complexity (5-20 is typical)
3. **Set `maxLLMCalls`** based on whether you need semantic analysis
4. **Use structured data** (JSON) when possible for easier JavaScript processing
5. **Test with small contexts first** before scaling to large documents
6. **Use `stream()`** for long-running operations to get real-time feedback
7. **Handle `abortSignal`** in production for proper cancellation support

## Migration from Old API

If you have examples using the old `RLM` class:

```typescript
// BEFORE (deprecated):
import { RLM } from '../src/rlm';
const rlm = new RLM({ model: 'gpt-4.1' });
const result = await rlm.completion(context, query);

// AFTER (new API):
import { RLMAgent } from '../src/rlm';
const agent = new RLMAgent({ model: 'gpt-4.1' });
const result = await agent.generate({ context, query });
```

## See Also

- [Main README](../README.md) - Full API documentation
- [RLM Source Code](../src/rlm.ts) - Agent implementation
- [RLM Tool Source](../src/rlm-tool.ts) - Tool factory implementation
- [AI SDK Documentation](https://sdk.vercel.ai/docs)
- [vm2 Documentation](https://github.com/patriksimek/vm2)
