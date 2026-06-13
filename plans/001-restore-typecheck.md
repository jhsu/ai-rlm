# Plan 001: Restore Typecheck As A Reliable Gate

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report; do not improvise. When done, update the status row for this plan in `plans/README.md` unless a reviewer tells you they maintain the index.
>
> **Drift check (run first)**: `git diff --stat cdceb83..HEAD -- package.json tsconfig.json tsconfig.build.json src/cloudflare-sandbox.ts apps/rlm-monitor/package.json apps/rlm-monitor/tsconfig.json apps/rlm-monitor/src/index.tsx`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding. If the relevant code no longer matches, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `cdceb83`, 2026-06-10

## Why this matters

The repo currently has no reliable TypeScript safety gate. `rtk tsc --noEmit` fails before runtime behavior can be checked, mostly because the root typecheck includes the experimental OpenTUI monitor without JSX intrinsic types, plus one strict-nullability error in `src/cloudflare-sandbox.ts`. Restoring typecheck gives future executors a cheap, repeatable verification command before making higher-risk changes to sandboxing or agent orchestration.

## Current state

Relevant files and roles:

- `package.json` — root package scripts and dependency metadata.
- `tsconfig.json` — root no-emit TypeScript config; currently broad enough to include `apps/rlm-monitor/src/index.tsx`.
- `tsconfig.build.json` — package build config; already scoped to `src/**/*` and emits declarations/build output.
- `src/cloudflare-sandbox.ts` — Cloudflare sandbox adapter; has one strict nullability error.
- `apps/rlm-monitor/package.json` — experimental Bun/OpenTUI app manifest.
- `apps/rlm-monitor/tsconfig.json` — monitor-specific TypeScript config if present; if absent, create only if choosing a separate monitor typecheck.
- `apps/rlm-monitor/src/index.tsx` — monitor UI using OpenTUI JSX intrinsic elements.

Observed root script excerpt:

```json
package.json:53-62
"scripts": {
  "build": "tsc -p tsconfig.build.json",
  "build:watch": "tsc -p tsconfig.build.json --watch",
  "cf:dev": "wrangler dev",
  "cf:deploy": "wrangler deploy",
  "clean": "rm -rf dist",
  "prepublishOnly": "npm run clean && npm run build",
  "release:github": "node ./bin/create-github-release.js",
  "test": "echo \"Error: no test specified\" && exit 1"
}
```

Root TypeScript config excerpt:

```jsonc
{
  "compilerOptions": {
    "lib": ["ESNext"],
    "target": "ESNext",
    "module": "Preserve",
    "moduleDetection": "force",
    "jsx": "react-jsx",
    "allowJs": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "noEmit": true,
    "strict": true,
    "skipLibCheck": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true
  }
}
```

Build config excerpt showing the package source boundary:

```jsonc
"include": ["src/**/*"],
"exclude": ["node_modules", "dist", "examples", "**/*.test.ts"]
```

Cloudflare nullability hotspot:

```ts
src/cloudflare-sandbox.ts:83-101
function returnTrailingExpression(code: string): string {
  const lines = code.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]?.trim();
    if (!line || line.startsWith("//")) {
      continue;
    }
    const expression = line.endsWith(";") ? line.slice(0, -1).trimEnd() : line;
    if (
      expression.endsWith("}") ||
      /^(?:return|throw|const|let|var|if|for|while|switch|try|catch|finally|function|class)\b/.test(
        expression
      )
    ) {
      return code;
    }

    lines[i] = `${lines[i]?.slice(0, lines[i].length - line.length) ?? ""}return (${expression});`;
    return lines.join("\n");
  }
  return code;
}
```

Monitor JSX pattern excerpt:

```tsx
apps/rlm-monitor/src/index.tsx:209-223
return (
  <box flexDirection="column" padding={1}>
    <box flexDirection="row" gap={2} marginBottom={1}>
      <text>
        <strong>RLM Monitor</strong>
      </text>
      <text fg={connected ? "green" : "red"}>
        {connected ? "● Connected" : "○ Disconnected"}
      </text>
      {wsPort && (
        <text fg="gray">WebSocket Port: {wsPort}</text>
      )}
      <text fg="gray">Press 'q' to quit, 'r' to reset</text>
    </box>
```

Read-only verification observed during audit:

- `rtk tsc --noEmit` failed with 79 errors in 2 files.
- Most errors were `TS2339` / `TS2322` in `apps/rlm-monitor/src/index.tsx`, e.g. `Property 'box' does not exist on type 'JSX.IntrinsicElements'` and `Property 'fg' does not exist on type 'SVGTextElementAttributes<SVGTextElement>`.
- One package-source error was `src/cloudflare-sandbox.ts(100,38): error TS2532: Object is possibly 'undefined'.`

Repo conventions to match:

- Source files use ESM imports with explicit `.js` extensions for local imports, e.g. `import type { RLMLogLevel, RLMLogger } from "./logger.js";` in `src/cloudflare-sandbox.ts`.
- TypeScript is strict. Prefer narrowing and explicit local variables over casts.
- Recent commit messages are short imperative phrases, e.g. `Use latest npm for changeset publish`, `Fix RLM finalization and tool guidance`.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Inspect status | `git status --short` | Shows only intentional changes after you edit |
| Root typecheck | `rtk tsc --noEmit` | exit 0, no TypeScript errors |
| Package build check | `npm run build` | exit 0 and emits `dist/`; note this mutates ignored/generated output |
| Monitor typecheck, if you add a monitor-specific config/script | `npm run typecheck:monitor` or documented equivalent | exit 0, no TypeScript errors |

Do not run formatters unless you add a formatter config in a separate approved plan. Do not run `npm test`; it is still a placeholder until a test-baseline plan lands.

## Scope

**In scope**:

- `package.json`
- `tsconfig.json`
- `tsconfig.build.json` only if needed to keep build/typecheck boundaries explicit
- `src/cloudflare-sandbox.ts`
- `apps/rlm-monitor/package.json`
- `apps/rlm-monitor/tsconfig.json` if creating or fixing monitor-specific typecheck
- `apps/rlm-monitor/src/index.tsx` only if fixing JSX type imports/config requires a local type reference or import
- New small TypeScript declaration file for OpenTUI JSX types, only if required and scoped to the monitor app

**Out of scope**:

- Runtime behavior changes to sandbox execution semantics.
- Refactoring `src/rlm.ts` or monitor UI layout.
- Package manager lockfile cleanup; that is Plan 002.
- Adding a test runner; that belongs to a separate verification-baseline plan.
- Removing secrets or editing `.env`; that belongs to a separate security hygiene plan.

## Git workflow

- Branch suggestion: `advisor/001-restore-typecheck`.
- Commit message style: short imperative sentence, e.g. `Restore typecheck gate`.
- Do not push or open a PR unless instructed.

## Steps

### Step 1: Decide and encode the typecheck boundary

Decide whether root `rtk tsc --noEmit` should cover only the package source or the whole workspace including `apps/rlm-monitor`.

Recommended answer: keep root typecheck package-focused and add a separate monitor typecheck if you want the monitor checked. This matches `tsconfig.build.json`, where the published package build already includes only `src/**/*`.

Implementation shape if following the recommendation:

- Add an explicit `include` to root `tsconfig.json` for package source files, likely `src/**/*`.
- Exclude generated/vendor/workspace app directories that are not part of the package typecheck, especially `node_modules`, `dist`, `examples`, `apps`, `site`, `.wrangler`.
- Add a root script such as `"typecheck": "tsc --noEmit"` so contributors do not have to infer the command.
- Do not change `build`, `prepublishOnly`, or package exports in this step.

If you instead decide root typecheck must include `apps/rlm-monitor`, STOP and report unless you can find official OpenTUI React JSX typing guidance locally in dependencies or docs. Do not invent a broad `any` JSX namespace for the entire repo.

**Verify**: `rtk tsc --noEmit` → should now report only package-source TypeScript errors. At the time this plan was written, the expected remaining error is the `TS2532` in `src/cloudflare-sandbox.ts:100`.

### Step 2: Fix the Cloudflare strict-nullability error narrowly

In `src/cloudflare-sandbox.ts`, update `returnTrailingExpression()` so TypeScript can prove the line being rewritten is defined.

Preferred shape:

- After the existing `if (!line || line.startsWith("//")) continue;` guard, assign `const originalLine = lines[i];`.
- If `originalLine === undefined`, continue defensively.
- Use `originalLine.slice(0, originalLine.length - line.length)` when constructing the replacement line.
- Keep the function behavior the same: preserve indentation, return trailing expression, and return `code` unchanged for declarations/control-flow/trailing block expressions.

Do not change `returnTrailingExpression()` parsing semantics beyond the nullability proof.

**Verify**: `rtk tsc --noEmit` → exit 0, no TypeScript errors for package source.

### Step 3: Add monitor typecheck only if it can be made accurate

If Step 1 excluded `apps/rlm-monitor` from root typecheck, decide whether to add an explicit monitor typecheck in this same plan.

Only do this if you can make OpenTUI JSX types accurate without masking real errors. Acceptable approaches:

- Use official OpenTUI React type declarations if the dependency exposes them.
- Add a monitor-local declaration file that models the OpenTUI intrinsic elements used by `apps/rlm-monitor/src/index.tsx` with reasonably specific props, not `any` for all JSX.
- Add `apps/rlm-monitor/tsconfig.json` with the correct JSX settings and include only monitor source/type declaration files.

If accurate monitor typing requires researching external docs, broad `any` JSX declarations, or changing UI code structure, skip this step and document in `package.json`/README that monitor typecheck is not yet part of the root gate.

**Verify**: If you add a monitor command, run it and expect exit 0. If you skip it, `rtk tsc --noEmit` must still exit 0 and `package.json` must make the root `typecheck` command clear.

### Step 4: Document the verification command in package metadata

Ensure `package.json` includes a clear typecheck script. Recommended minimal script:

```json
"typecheck": "tsc --noEmit"
```

If you added a monitor-specific typecheck, add a second script with an explicit name such as `typecheck:monitor`.

Do not modify the placeholder `test` script in this plan.

**Verify**: `npm run typecheck` → exit 0, no TypeScript errors.

## Test plan

This plan restores typecheck; it does not add tests.

- New tests: none.
- Required checks:
  - `npm run typecheck` must exit 0.
  - `rtk tsc --noEmit` must exit 0.
  - `npm run build` should exit 0, but it emits `dist/`; run it only when generated output is acceptable in your executor environment.

## Done criteria

All must hold:

- [ ] `npm run typecheck` exits 0.
- [ ] `rtk tsc --noEmit` exits 0.
- [ ] No broad global `JSX.IntrinsicElements` declaration was added to hide monitor errors across the whole repo.
- [ ] `npm test` remains unchanged unless a separate test-baseline plan was also approved.
- [ ] No files outside the in-scope list are modified, except generated `dist/` if you ran `npm run build`.
- [ ] `plans/README.md` status row for Plan 001 is updated.

## STOP conditions

Stop and report back if:

- The current code no longer matches the excerpts above.
- Fixing typecheck requires changing runtime behavior in `src/rlm.ts`, sandbox interfaces, or package exports.
- OpenTUI JSX typing cannot be made accurate without a broad `any` declaration.
- `npm run typecheck` still fails after the Cloudflare nullability fix and root typecheck scoping.
- You discover a package-manager change is required to install or resolve type dependencies; defer that to Plan 002 instead of improvising here.

## Maintenance notes

- Plan 002 may change package-manager scripts or install commands. Preserve the `typecheck` script semantics when doing that work.
- If the monitor graduates from experimental app to supported package surface, add a dedicated monitor typecheck rather than silently expanding root `tsconfig.json` again.
- Reviewers should scrutinize whether TypeScript errors were fixed by correct scoping/narrowing rather than suppressed with casts or broad declarations.
