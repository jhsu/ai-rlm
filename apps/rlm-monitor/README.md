# RLM Monitor

A terminal UI application for real-time monitoring of RLMAgent execution.

## Features

- **Live Metrics Dashboard**: Track iterations, LLM calls, execution time, and errors
- **Step-by-Step Trajectory**: View each execution step with code and output
- **Real-Time Event Log**: Monitor all instrumentation events as they happen
- **WebSocket Server**: Receives instrumentation data from RLMAgent
- **Keyboard Navigation**: Interactive step selection and controls

## Installation

Root dependencies are managed with npm. The monitor is a Bun app, so install and run it from this directory with Bun:

```bash
cd apps/rlm-monitor
bun install
```

## Usage

### Start the Monitor

```bash
bun src/index.tsx
```

The monitor will start a WebSocket server on an available port (displayed in the UI).

### Connect from RLMAgent

In your RLMAgent code, use the instrumentation callbacks to send data to the monitor:

```typescript
import { RLMAgent } from "rlm/src/rlm.ts";

const agent = new RLMAgent({ model, subModel });

// WebSocket connection to monitor
const ws = new WebSocket("ws://localhost:PORT");

const result = await agent.generate({
  options: { 
    context: "Your context here...",
    onIterationStart: async (event) => {
      ws.send(JSON.stringify({ type: "iteration_start", ...event }));
    },
    onIterationComplete: async (event) => {
      ws.send(JSON.stringify({ type: "iteration_complete", ...event }));
    },
    onLLMCall: async (event) => {
      ws.send(JSON.stringify({ type: "llm_call", ...event }));
    },
    onError: async (event) => {
      ws.send(JSON.stringify({ type: "error", ...event }));
    },
  },
  prompt: "Your query here...",
});
```

## Keyboard Shortcuts

- `↑/↓` - Navigate between execution steps
- `q` or `Ctrl+C` - Quit the monitor
- `r` - Reset/clear all data

## Architecture

The monitor consists of:

1. **WebSocket Server**: Receives JSON events from RLMAgent
2. **React UI**: Real-time display using OpenTUI
3. **State Management**: Tracks metrics, steps, and events

## Event Types

The monitor accepts these event types:

- `iteration_start` - Begin of an iteration
- `iteration_complete` - End of an iteration with step data
- `llm_call` - LLM invocation
- `error` - Error during execution
- `connected/disconnected` - WebSocket connection status

## Development

```bash
# Run with hot reload
bun --hot src/index.tsx

# Type check
bun run typecheck
```

## Integration Example

See `../examples/instrumentation-example.ts` for a complete example of using the instrumentation callbacks.
