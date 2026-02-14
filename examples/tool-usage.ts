/**
 * RLM Tool Usage Examples
 *
 * Demonstrates using RLM as a tool with AI SDK's generateText and ToolLoopAgent.
 */

import { generateText, ToolLoopAgent } from "ai";
import { createRLMTool } from "../src/rlm-tool";
import { model, subModel } from "./model";

// ============================================================================
// Example 1: Using RLM Tool with generateText
// ============================================================================

export async function example1ToolWithGenerateText() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 1: RLM Tool with generateText()");
  console.log("=".repeat(60) + "\n");

  // Create the RLM tool
  const rlmTool = createRLMTool({
    model,
    subModel,
    maxIterations: 10,
    maxLLMCalls: 10,
  });

  // Large context to analyze
  const largeContext = `
    System Architecture Document:

    Authentication Service (auth-service)
    - Port: 8080
    - Database: PostgreSQL
    - Cache: Redis
    - Issues: Password hashing uses MD5 (vulnerable)

    Payment Gateway (payment-service)
    - Port: 8081
    - Database: PostgreSQL
    - Issues: No input validation on card numbers

    User Service (user-service)
    - Port: 8082
    - Database: MongoDB
    - Issues: SQL injection vulnerability in search endpoint

    API Gateway (api-gateway)
    - Port: 80
    - Rate limiting: 100 req/min
    - SSL: Enabled
    - Issues: None identified
  `;

  console.log("Context: System architecture with security info\n");

  try {
    const result = await generateText({
      model: "gpt-4.1",
      tools: {
        analyzeArchitecture: rlmTool,
      },
      prompt: `Analyze this system architecture and identify all security vulnerabilities:

${largeContext}

List each service and its security issues, then provide recommendations for fixes.`,
    });

    console.log("✓ Result:", result.text);
    console.log("\n--- Tool Calls Made ---");
    if (result.toolCalls && result.toolCalls.length > 0) {
      result.toolCalls.forEach((call) => {
        console.log(`\nTool: ${call.toolName}`);
        console.log("Input:", JSON.stringify(call.input, null, 2));
      });
    }

    return result;
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
}

// ============================================================================
// Example 2: Using RLM Tool with ToolLoopAgent
// ============================================================================

export async function example2ToolWithAgent() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 2: RLM Tool with ToolLoopAgent");
  console.log("=".repeat(60) + "\n");

  // Create RLM tool
  const rlmTool = createRLMTool({
    model,
    subModel,
    maxIterations: 15,
    maxLLMCalls: 20,
  });

  // Create agent with RLM tool
  const agent = new ToolLoopAgent({
    model: "gpt-4.1",
    instructions: `You are a security auditor. Use the analyzeLargeContext tool to examine code and configuration files for security issues. Be thorough and document all findings.`,
    tools: {
      analyzeLargeContext: rlmTool,
    },
  });

  const configFile = `
    # Application Configuration
    DATABASE_URL=postgres://admin:password123@localhost:5432/mydb
    API_KEY=sk-1234567890abcdef
    DEBUG_MODE=true
    SECRET_KEY=hardcoded_secret_key_here
    JWT_SECRET=weak_jwt_secret

    # Feature flags
    ENABLE_PAYMENTS=true
    ENABLE_LOGGING=false
  `;

  console.log("Analyzing configuration file for secrets...\n");

  try {
    const result = await agent.generate({
      prompt: `Analyze this configuration file and identify all security issues, especially hardcoded secrets and weak credentials:

${configFile}`,
    });

    console.log("✓ Agent Result:", result.text);
    console.log("\n--- Execution Steps ---");
    if (result.steps) {
      result.steps.forEach((step, idx) => {
        const toolCalls = step.toolCalls || [];
        if (toolCalls.length > 0) {
          console.log(`\nStep ${idx + 1}:`);
          toolCalls.forEach((call: { toolName: string }) => {
            console.log(`  Tool called: ${call.toolName}`);
          });
        }
      });
    }

    return result;
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
}

// ============================================================================
// Example 3: Direct Tool Execution
// ============================================================================

export async function example3DirectToolExecution() {
  console.log("\n" + "=".repeat(60));
  console.log("Example 3: Direct Tool Execution");
  console.log("=".repeat(60) + "\n");

  const rlmTool = createRLMTool({
    model,
    subModel,
  });

  // Access the tool's execute function directly
  const context = `
    Sales Report Q1-Q4 2024:

    Q1 Revenue: $125,000
    Q2 Revenue: $142,000
    Q3 Revenue: $138,000
    Q4 Revenue: $165,000

    Target: $500,000 annual
  `;

  const query =
    "Did we meet our annual revenue target? Calculate totals and variance.";

  console.log("Query:", query);
  console.log("Executing tool directly...\n");

  try {
    // Note: In practice, the LLM would call the tool automatically
    // For this example, we'll just show that the tool is ready to use
    console.log("✓ Tool configured and ready");
    console.log(
      "  Input schema:",
      JSON.stringify(rlmTool.inputSchema, null, 2).substring(0, 200) + "...",
    );

    return { status: "Tool ready for LLM invocation" };
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
}

// ============================================================================
// Main Runner
// ============================================================================

export async function runAllToolExamples() {
  console.log("\n" + "=".repeat(60));
  console.log("RLM Tool Usage Examples");
  console.log("=".repeat(60));

  try {
    await example1ToolWithGenerateText();
    await example2ToolWithAgent();
    await example3DirectToolExecution();

    console.log("\n" + "=".repeat(60));
    console.log("All tool examples completed!");
    console.log("=".repeat(60) + "\n");
  } catch (error) {
    console.error("Examples failed:", error);
    throw error;
  }
}

// Run if executed directly
if (import.meta.main) {
  runAllToolExamples().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

export default {
  example1ToolWithGenerateText,
  example2ToolWithAgent,
  example3DirectToolExecution,
  runAllToolExamples,
};
