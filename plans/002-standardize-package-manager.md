# Plan 002: Standardize Package Manager And Lockfiles

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report; do not improvise. When done, update the status row for this plan in `plans/README.md` unless a reviewer tells you they maintain the index.
>
> **Drift check (run first)**: `git diff --stat cdceb83..HEAD -- package.json package-lock.json pnpm-lock.yaml bun.lock pnpm-workspace.yaml .github/workflows/release.yml apps/rlm-monitor/package.json apps/rlm-monitor/README.md apps/rlm-monitor/bun.lock README.md examples/README.md site/package.json`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding. If the relevant code no longer matches, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: `plans/001-restore-typecheck.md`
- **Category**: dx
- **Planned at**: commit `cdceb83`, 2026-06-10

## Why this matters

The repo currently presents npm, pnpm, and Bun as possible sources of dependency truth. That makes local verification, release automation, and dependency audit results hard to reproduce because different tools can resolve different dependency graphs. Standardizing package-manager ownership will reduce onboarding friction and give future dependency/security plans a single lockfile to update.

## Current state

Relevant files and roles:

- `package.json` — root package manifest; currently has no `packageManager` field.
- `package-lock.json` — npm lockfile used by the current release workflow.
- `pnpm-lock.yaml` — pnpm lockfile present at repo root.
- `bun.lock` — Bun lockfile present at repo root.
- `pnpm-workspace.yaml` — declares `apps/*` and `site` workspace packages.
- `.github/workflows/release.yml` — release workflow installs with npm.
- `apps/rlm-monitor/package.json` — monitor app manifest, likely intended for Bun runtime.
- `apps/rlm-monitor/README.md` — monitor setup docs instruct `bun install`.
- `site/package.json` — simple static site package.
- `README.md` and `examples/README.md` — user/contributor install and example docs.

Root package script excerpt:

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

Release workflow excerpt:

```yaml
.github/workflows/release.yml:25-31
- uses: actions/setup-node@v4
  with:
    node-version: "22"
    registry-url: "https://registry.npmjs.org"

- run: npm install
```

Workspace config currently present:

```yaml
pnpm-workspace.yaml:1-4
packages:
  - apps/*
  - site
```

Monitor install docs currently say:

```md
apps/rlm-monitor/README.md:15-18
```bash
cd apps/rlm-monitor
bun install
```
```

Site package is a minimal static app:

```json
site/package.json:1-9
{
  "name": "ai-rlm-site",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "npx serve .",
    "start": "npx serve ."
  }
}
```

Package manager files observed at plan time:

- Root `package-lock.json` exists.
- Root `pnpm-lock.yaml` exists.
- Root `bun.lock` exists.
- Root `pnpm-workspace.yaml` exists.
- `apps/rlm-monitor/bun.lock` may exist if the app was installed independently; verify before editing.

Repo conventions to match:

- Release workflow is npm-oriented today.
- Recent commit messages are short imperative phrases, e.g. `Use latest npm for changeset publish`, `Let changesets own release publishing`.
- Do not introduce broad tooling migrations without updating docs and CI in the same change.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Inspect status | `git status --short` | Shows only intentional changes after you edit |
| Typecheck from Plan 001 | `npm run typecheck` | exit 0, no TypeScript errors |
| Dependency audit after standardization | `npm audit --json` or chosen package-manager equivalent | Completes; remaining vulnerabilities are understood and documented |
| Build check | `npm run build` or chosen package-manager equivalent | exit 0 and emits `dist/`; note this mutates generated output |

If you choose a package manager other than npm, replace the npm commands above with exact equivalent commands and update this plan's row in `plans/README.md` with the final commands used.

## Scope

**In scope**:

- `package.json`
- One authoritative root lockfile for the chosen package manager
- Deleting stale root lockfiles for unchosen package managers
- `pnpm-workspace.yaml` if choosing or rejecting pnpm as the root package manager
- `.github/workflows/release.yml`
- `README.md` setup/development commands if present
- `examples/README.md` setup/development commands if present
- `apps/rlm-monitor/package.json`
- `apps/rlm-monitor/README.md`
- `apps/rlm-monitor/bun.lock` if present
- `site/package.json` only if needed to align workspace/package-manager scripts

**Out of scope**:

- Removing vulnerable dev dependencies such as `just-bash`; that should be a follow-up dependency-remediation plan after the package-manager decision.
- TypeScript fixes beyond preserving Plan 001's working `typecheck` command.
- Refactoring source code or examples.
- Changing publish semantics beyond install/tooling alignment.
- Adding tests or replacing the placeholder `test` script.

## Git workflow

- Branch suggestion: `advisor/002-standardize-package-manager`.
- Commit message style: short imperative sentence, e.g. `Standardize package manager`.
- Do not push or open a PR unless instructed.

## Steps

### Step 1: Choose the package manager explicitly

Make one explicit decision before editing lockfiles.

Recommended answer for least disruption: standardize the root package and release workflow on npm because `.github/workflows/release.yml` already uses `npm install`, `package-lock.json` exists, and package scripts are npm-oriented. Under this recommendation:

- Add a `packageManager` field to root `package.json`, using the npm major/version available in the release environment. If you cannot determine an exact patch version locally, use the installed local npm version and document it.
- Keep root `package-lock.json` as the authoritative root lockfile.
- Remove root `pnpm-lock.yaml`, root `bun.lock`, and `pnpm-workspace.yaml` unless there is an explicit, documented workspace reason to keep pnpm.
- Treat Bun as a runtime option for the monitor only if necessary, not as root dependency authority.

Alternative acceptable answer: standardize on pnpm because `pnpm-workspace.yaml` already defines workspaces. If choosing pnpm, update the release workflow to install/action setup pnpm, set `packageManager`, keep `pnpm-lock.yaml`, and remove npm/Bun root lockfiles. This is higher blast radius than npm because release automation changes more.

STOP if you cannot justify the choice from repo evidence. Do not keep multiple authoritative root lockfiles.

**Verify**: `git status --short` → no source-code changes yet, only metadata/lockfile edits once you proceed.

### Step 2: Align root manifest and lockfiles

Implement the package-manager decision from Step 1.

If choosing npm:

- Add `"packageManager": "npm@<version>"` near the top-level metadata in `package.json`.
- Keep `package-lock.json` and regenerate it only with npm if needed.
- Delete root `pnpm-lock.yaml`, root `bun.lock`, and `pnpm-workspace.yaml`.
- Do not delete `apps/rlm-monitor/bun.lock` in this step unless you also update monitor docs to stop using Bun.

If choosing pnpm:

- Add `"packageManager": "pnpm@<version>"` to `package.json`.
- Keep `pnpm-lock.yaml` and regenerate it only with pnpm if needed.
- Delete root `package-lock.json` and root `bun.lock`.
- Keep `pnpm-workspace.yaml`, but confirm it intentionally covers `apps/*` and `site`.

If choosing Bun for root, STOP and report. The current release workflow and npm package publishing setup make Bun root ownership a higher-risk migration that needs a separate design decision.

**Verify**: run the chosen package manager's install command only if required to refresh the authoritative lockfile. Expected result: exit 0 and exactly one authoritative root lockfile remains.

### Step 3: Align release workflow with the chosen tool

Update `.github/workflows/release.yml` to use the chosen package manager consistently.

If choosing npm:

- `npm install` can remain, but consider `npm ci` if the lockfile is authoritative and CI should be reproducible.
- Keep `changesets/action` publish command compatible with npm.
- Do not change secrets or provenance settings.

If choosing pnpm:

- Add package-manager setup using the standard GitHub Action pattern for pnpm.
- Replace install with a frozen-lockfile pnpm install command.
- Ensure the changeset publish command still publishes to npm registry with provenance.

**Verify**: read the final workflow and ensure there is exactly one install tool in the release job.

### Step 4: Update docs for install and monitor usage

Update user-facing docs so there is no contradictory install guidance.

Minimum docs updates:

- In root `README.md`, if development/setup commands exist, use the chosen root package manager consistently.
- In `examples/README.md`, if install/run commands exist, use the chosen root package manager consistently.
- In `apps/rlm-monitor/README.md`, make the relationship explicit:
  - If root standardizes on npm and the monitor still requires Bun runtime, say: root dependencies are managed with npm; the monitor is a Bun app and is run with Bun. Avoid implying Bun manages the root repo.
  - If root standardizes on pnpm, update monitor install/run commands to match pnpm workspace behavior unless Bun-specific APIs (`Bun.serve`, `Bun.write`) require Bun runtime. If Bun runtime remains required, document that split clearly.

Do not rewrite docs beyond install/tooling clarity.

**Verify**: search docs for the unchosen root package-manager commands and ensure any remaining mention is intentionally scoped, e.g. Bun runtime for the monitor.

### Step 5: Run verification and record remaining audit signal

Run the final command set for the chosen package manager.

Required checks:

- Typecheck from Plan 001: `npm run typecheck` or chosen equivalent → exit 0.
- Dependency audit: `npm audit --json` or chosen equivalent → command completes. It may still report the known `just-bash` / `fast-xml-parser` issue because removing vulnerable dev tooling is out of scope for this plan; record that as expected follow-up if still present.
- Optional build: `npm run build` or chosen equivalent → exit 0; this emits `dist/`, so do not include generated output unless the repo expects it.

**Verify**: `git status --short` → only in-scope manifest, lockfile, workflow, and docs changes are present.

## Test plan

This plan changes dependency/tooling metadata, not runtime code.

- New tests: none.
- Required verification:
  - Chosen install command exits 0 if run.
  - `npm run typecheck` or chosen equivalent exits 0.
  - Chosen audit command completes and remaining vulnerabilities are either cleared or explicitly deferred to the dependency-remediation plan.
  - Release workflow uses only the chosen root package manager.

## Done criteria

All must hold:

- [ ] Root `package.json` has a `packageManager` field.
- [ ] Exactly one authoritative root lockfile remains.
- [ ] Release workflow install step uses the chosen package manager consistently.
- [ ] Root docs and monitor docs no longer present conflicting root install instructions.
- [ ] Plan 001's `typecheck` command still exits 0.
- [ ] Any remaining audit vulnerabilities are documented as follow-up, not hidden.
- [ ] No source code files are modified.
- [ ] `plans/README.md` status row for Plan 002 is updated.

## STOP conditions

Stop and report back if:

- The current package-manager files no longer match the current-state summary.
- The chosen package manager requires changing package exports, source imports, or runtime code.
- The release workflow cannot be aligned without changing publish credentials, provenance, or changeset behavior.
- The monitor app cannot be represented cleanly under the chosen root package manager because it relies on Bun runtime APIs.
- Installing/regenerating the lockfile produces unexpected large dependency churn not explainable by the package-manager decision.

## Maintenance notes

- After this plan lands, run a separate dependency-remediation plan for `just-bash` / `fast-xml-parser` / `brace-expansion` if audit still reports them.
- Future contributors and agents should use the `packageManager` field and the remaining lockfile as source of truth.
- Reviewers should focus on whether the release workflow, docs, and lockfile all tell the same story.
