# ai-rlm

## 2.0.0

### Major Changes

- d58f9ba: - BREAKING: changed sandbox implementation, switched from vm2 to quickjs
  - Adds sandbox provider interface

### Minor Changes

- ee8d5d9: Add a pluggable sandbox interface to `RLMAgent` via `sandboxFactory`, including exported `RLMSandbox` types and a `createQuickJSSandbox` default factory.
- 51c4403: update stream to emit steps and events
- d58f9ba: Add logger interface, deprecate verbose option

## 1.3.0

### Minor Changes

- c2a0f4c: add iteration/sub-agent hooks and usage summary

## 1.2.0

### Minor Changes

- c6189cd: switch to using quickjs for REPL

## 1.1.0

### Minor Changes

- 5eba4f8: Fix recursion functions

## 1.0.0

### Major Changes

- a23dcd1: initial release of ai-rlm
