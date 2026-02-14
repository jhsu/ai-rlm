/**
 * AI RLM - Recursive Language Model for the Vercel AI SDK
 *
 * @module ai-rlm
 * @description Process long contexts through iterative code execution and sub-LLM queries
 */

// Main exports
export { RLMAgent, RLM } from './rlm.js';
export type {
  RLMAgentSettings,
  RLMAgentCallParameters,
  REPLStep,
  RLMGenerateResult,
  RLMStreamResult,
  RLMResult,
  RLMContext,
} from './rlm.js';

// Tool exports
export { createRLMTool } from './rlm-tool.js';
export type {
  RLMToolConfig,
} from './rlm-tool.js';

// Default export
export { RLMAgent as default } from './rlm.js';
