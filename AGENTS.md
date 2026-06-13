@/home/jhsu/.codex/RTK.md

# ai-rlm Workflow Notes

## Examples

- Examples live in `examples/`.
- Root dependencies are managed with npm, but examples are run directly with Bun:
  - `bun run examples/<example>.ts`
- Examples are excluded from the root `tsconfig`, so when adding or changing an example, run a targeted check:
  - `npx tsc --noEmit --module Preserve --moduleResolution bundler --allowImportingTsExtensions --target ESNext --lib ESNext --strict --skipLibCheck --types bun examples/<example>.ts`

## Verification

Before finishing code changes, run:

- `npm run typecheck`
- `npm run build`
- The targeted example typecheck above if an example changed

## Changesets

- For public API or behavior changes, add a changeset in `.changeset/`.
- Use `minor` for new public options or features.
- Use `patch` for fixes, docs, or internal behavior changes.

## RLM Design Notes

- Full context should stay in the sandbox as `context`; avoid putting raw large documents into outer prompts.
- For large-context behavior, prefer improving context metadata and planning guidance over task-specific prompt hacks.
- `sub_rlm()` only creates recursive RLM agents when `maxDepth > 1`; at terminal depth it falls back to `llm_query()`.
- If changing context planning behavior, verify both QuickJS and Cloudflare sandbox propagation paths.

## AI SDK

- This repo uses AI SDK v6 patterns:
  - `inputSchema`, not `parameters`
  - `stepCountIs(...)` for multi-step `generateText`
  - `prepareStep` and `toolChoice` for forced tool-call flows
