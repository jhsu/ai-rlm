#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const repoRoot = process.cwd();

function run(command, args, options = {}) {
  return execFileSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  }).trim();
}

function runQuiet(command, args) {
  try {
    run(command, args);
    return true;
  } catch {
    return false;
  }
}

function fail(message) {
  console.error(message);
  process.exit(1);
}

function getPackageInfo() {
  const packageJsonPath = join(repoRoot, "package.json");
  const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));

  if (!packageJson.name || !packageJson.version) {
    fail("package.json must contain both name and version.");
  }

  return {
    name: packageJson.name,
    version: packageJson.version,
  };
}

function getChangelogSection(version) {
  const changelogPath = join(repoRoot, "CHANGELOG.md");
  const changelog = readFileSync(changelogPath, "utf8");
  const heading = `## ${version}`;
  const start = changelog.indexOf(heading);

  if (start === -1) {
    fail(`Could not find changelog section "${heading}" in CHANGELOG.md.`);
  }

  const afterHeading = changelog.slice(start + heading.length);
  const nextHeadingMatch = afterHeading.match(/\n##\s+/);
  const end =
    nextHeadingMatch === null
      ? changelog.length
      : start + heading.length + nextHeadingMatch.index;

  const section = changelog.slice(start, end).trim();

  if (!section) {
    fail(`Changelog section for version ${version} is empty.`);
  }

  return section;
}

function ensureVersionFilesAreCommitted() {
  const diff = run("git", ["status", "--short", "--", "package.json", "CHANGELOG.md"]);

  if (diff) {
    fail(
      "package.json or CHANGELOG.md has uncommitted changes. Commit the version/changelog changes before creating a release tag.",
    );
  }
}

function ensureRemoteExists(remoteName) {
  if (!runQuiet("git", ["remote", "get-url", remoteName])) {
    fail(`Git remote "${remoteName}" does not exist.`);
  }
}

function ensureTag(tagName, tagMessage) {
  if (runQuiet("git", ["rev-parse", "-q", "--verify", `refs/tags/${tagName}`])) {
    console.log(`Using existing local tag ${tagName}.`);
    return;
  }

  run("git", ["tag", "-a", tagName, "-m", tagMessage]);
  console.log(`Created annotated tag ${tagName}.`);
}

function pushTagIfNeeded(remoteName, tagName) {
  const remoteHasTag = runQuiet("git", ["ls-remote", "--exit-code", "--tags", remoteName, `refs/tags/${tagName}`]);

  if (remoteHasTag) {
    console.log(`Remote ${remoteName} already has tag ${tagName}.`);
    return;
  }

  run("git", ["push", remoteName, tagName], { stdio: "inherit" });
}

function ensureReleaseDoesNotExist(tagName) {
  if (runQuiet("gh", ["release", "view", tagName])) {
    fail(`GitHub release ${tagName} already exists.`);
  }
}

function createRelease(tagName, title, notes) {
  const notesFile = join(tmpdir(), `gh-release-notes-${process.pid}.md`);

  try {
    writeFileSync(notesFile, notes);
    run("gh", ["release", "create", tagName, "--title", title, "--notes-file", notesFile], {
      stdio: "inherit",
    });
  } finally {
    unlinkSync(notesFile);
  }
}

const remoteName = process.argv[2] || "origin";
const { name, version } = getPackageInfo();
const tagName = version;
const releaseTitle = `${name}@${version}`;
const releaseNotes = getChangelogSection(version);

ensureVersionFilesAreCommitted();
ensureRemoteExists(remoteName);
ensureTag(tagName, releaseNotes);
pushTagIfNeeded(remoteName, tagName);
ensureReleaseDoesNotExist(tagName);
createRelease(tagName, releaseTitle, releaseNotes);
