/**
 * RLM Monitor - Terminal UI for monitoring RLMAgent execution
 *
 * This app creates a WebSocket server that receives instrumentation
 * events from RLMAgent and displays them in real-time.
 */

import { createCliRenderer } from "@opentui/core";
import { createRoot, useKeyboard } from "@opentui/react";
import React, { useState, useEffect, useCallback } from "react";

// Event types matching RLM instrumentation
interface RLMEvent {
  type: "iteration_start" | "iteration_complete" | "llm_call" | "error" | "connected" | "disconnected";
  timestamp: number;
  data?: any;
  error?: string;
}

interface StepInfo {
  iteration: number;
  code: string;
  output: string;
  executionTimeMs: number;
  status: "running" | "complete" | "error";
}

// Create renderer first so it's available for cleanup
const renderer = await createCliRenderer();

function MonitorApp() {
  const [events, setEvents] = useState<RLMEvent[]>([]);
  const [steps, setSteps] = useState<StepInfo[]>([]);
  const [metrics, setMetrics] = useState({
    totalIterations: 0,
    totalLLMCalls: 0,
    totalExecutionTime: 0,
    errors: 0,
  });
  const [connected, setConnected] = useState(false);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [wsPort, setWsPort] = useState<number | null>(null);

  // WebSocket server setup
  useEffect(() => {
    const server = Bun.serve({
      port: 0, // Let OS assign a port
      fetch(req, server) {
        // Upgrade WebSocket connections
        if (server.upgrade(req)) {
          return; // Upgraded to WebSocket
        }
        // Simple HTTP endpoint for health check
        return new Response("RLM Monitor WebSocket Server", { status: 200 });
      },
      websocket: {
        open(ws) {
          setConnected(true);
          addEvent({
            type: "connected",
            timestamp: Date.now(),
          });
        },
        close(ws) {
          setConnected(false);
          addEvent({
            type: "disconnected",
            timestamp: Date.now(),
          });
        },
        message(ws, message) {
          try {
            const data = JSON.parse(message as string);
            handleRLMMessage(data);
          } catch (e) {
            console.error("Failed to parse message:", e);
          }
        },
      },
    });

    const port = server.port;
    setWsPort(port ?? null);
    // Write port to file for external detection
    Bun.write("/tmp/rlm-monitor-port.txt", String(port)).catch(() => {});
    console.log(`RLM Monitor WebSocket port: ${port}`);

    return () => {
      server.stop();
    };
  }, []);

  const addEvent = useCallback((event: RLMEvent) => {
    setEvents((prev) => [...prev.slice(-50), event]); // Keep last 50 events
  }, []);

  const handleRLMMessage = useCallback((data: any) => {
    const timestamp = Date.now();

    switch (data.type) {
      case "iteration_start":
        addEvent({
          type: "iteration_start",
          timestamp,
          data: { iteration: data.iteration },
        });
        setSteps((prev) => [
          ...prev,
          {
            iteration: data.iteration,
            code: "",
            output: "",
            executionTimeMs: 0,
            status: "running",
          },
        ]);
        setMetrics((m) => ({ ...m, totalIterations: data.iteration }));
        break;

      case "iteration_complete":
        addEvent({
          type: "iteration_complete",
          timestamp,
          data: {
            iteration: data.iteration,
            executionTimeMs: data.executionTimeMs,
          },
        });
        setSteps((prev) =>
          prev.map((step) =>
            step.iteration === data.iteration
              ? {
                  ...step,
                  code: data.step.code,
                  output: data.step.output,
                  executionTimeMs: data.executionTimeMs,
                  status: "complete",
                }
              : step
          )
        );
        setMetrics((m) => ({
          ...m,
          totalExecutionTime: m.totalExecutionTime + data.executionTimeMs,
        }));
        break;

      case "llm_call":
        addEvent({
          type: "llm_call",
          timestamp,
          data: { modelId: data.modelId, isSubCall: data.isSubCall },
        });
        setMetrics((m) => ({ ...m, totalLLMCalls: m.totalLLMCalls + 1 }));
        break;

      case "error":
        addEvent({
          type: "error",
          timestamp,
          error: data.error.message,
          data: { phase: data.phase, iteration: data.iteration },
        });
        setMetrics((m) => ({ ...m, errors: m.errors + 1 }));
        break;
    }
  }, []);

  // Keyboard shortcuts
  useKeyboard((key) => {
    if (key.name === "q" || (key.ctrl && key.name === "c")) {
      renderer.destroy();
    }
    if (key.name === "up" && selectedStep !== null && selectedStep > 0) {
      setSelectedStep(selectedStep - 1);
    }
    if (key.name === "down" && selectedStep !== null && selectedStep < steps.length - 1) {
      setSelectedStep(selectedStep + 1);
    }
    if (key.name === "r") {
      // Reset
      setEvents([]);
      setSteps([]);
      setMetrics({
        totalIterations: 0,
        totalLLMCalls: 0,
        totalExecutionTime: 0,
        errors: 0,
      });
      setSelectedStep(null);
    }
  });

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatTimestamp = (ts: number) => {
    const date = new Date(ts);
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <box flexDirection="column" padding={1}>
      {/* Header */}
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

      {/* Metrics Dashboard */}
      <box
        flexDirection="row"
        gap={2}
        marginBottom={1}
        padding={1}
        border
        backgroundColor="#1a1a2e"
      >
        <box flexDirection="column" paddingRight={2}>
          <text fg="cyan">
            <strong>Iterations</strong>
          </text>
          <text>{metrics.totalIterations}</text>
        </box>
        <box flexDirection="column" paddingRight={2}>
          <text fg="yellow">
            <strong>LLM Calls</strong>
          </text>
          <text>{metrics.totalLLMCalls}</text>
        </box>
        <box flexDirection="column" paddingRight={2}>
          <text fg="green">
            <strong>Total Time</strong>
          </text>
          <text>{formatTime(metrics.totalExecutionTime)}</text>
        </box>
        <box flexDirection="column">
          <text fg={metrics.errors > 0 ? "red" : "green"}>
            <strong>Errors</strong>
          </text>
          <text>{metrics.errors}</text>
        </box>
      </box>

      {/* Main Content */}
      <box flexDirection="row" flexGrow={1} gap={1}>
        {/* Steps List */}
        <box flexDirection="column" width="40%" border padding={1}>
          <text marginBottom={1}>
            <strong>Execution Steps</strong>
          </text>
          <scrollbox flexGrow={1}>
            {steps.length === 0 ? (
              <text fg="gray">Waiting for execution...</text>
            ) : (
              steps.map((step, index) => (
                <box
                  key={step.iteration}
                  flexDirection="column"
                  padding={1}
                  backgroundColor={
                    selectedStep === index ? "#2a2a4e" : undefined
                  }
                  onMouseDown={() => setSelectedStep(index)}
                >
                  <box flexDirection="row" gap={1}>
                    <text fg="cyan">#{step.iteration}</text>
                    <text
                      fg={
                        step.status === "complete"
                          ? "green"
                          : step.status === "error"
                          ? "red"
                          : "yellow"
                      }
                    >
                      {step.status === "complete"
                        ? "✓"
                        : step.status === "error"
                        ? "✗"
                        : "○"}
                    </text>
                    <text fg="gray">{formatTime(step.executionTimeMs)}</text>
                  </box>
                  <text fg="gray">
                    {step.code.substring(0, 40)}...
                  </text>
                </box>
              ))
            )}
          </scrollbox>
        </box>

        {/* Step Details */}
        <box flexDirection="column" flexGrow={1} border padding={1}>
          <text marginBottom={1}>
            <strong>Step Details</strong>
          </text>
          {selectedStep !== null && steps[selectedStep] ? (
            <scrollbox flexGrow={1}>
              <box flexDirection="column" gap={1}>
                <box>
                  <text fg="cyan">Iteration: </text>
                  <text>{steps[selectedStep].iteration}</text>
                </box>
                <box>
                  <text fg="cyan">Status: </text>
                  <text
                    fg={
                      steps[selectedStep].status === "complete"
                        ? "green"
                        : steps[selectedStep].status === "error"
                        ? "red"
                        : "yellow"
                    }
                  >
                    {steps[selectedStep].status}
                  </text>
                </box>
                <box>
                  <text fg="cyan">Execution Time: </text>
                  <text>{formatTime(steps[selectedStep].executionTimeMs)}</text>
                </box>
                <box marginTop={1}>
                  <text fg="yellow">
                    <strong>Code:</strong>
                  </text>
                </box>
                <box
                  padding={1}
                  backgroundColor="#0f0f1e"
                  border
                >
                  <scrollbox>
                    <text>{steps[selectedStep].code}</text>
                  </scrollbox>
                </box>
                <box marginTop={1}>
                  <text fg="yellow">
                    <strong>Output:</strong>
                  </text>
                </box>
                <box
                  padding={1}
                  backgroundColor="#0f0f1e"
                  border
                >
                  <text>
                    {steps[selectedStep].output || "No output"}
                  </text>
                </box>
              </box>
            </scrollbox>
          ) : (
            <text fg="gray">
              {steps.length > 0
                ? "Select a step to view details (↑/↓ keys)"
                : "No steps executed yet"}
            </text>
          )}
        </box>
      </box>

      {/* Event Log */}
      <box flexDirection="column" height="25%" border padding={1} marginTop={1}>
        <text marginBottom={1}>
          <strong>Event Log</strong>
        </text>
        <scrollbox flexGrow={1}>
          {events.length === 0 ? (
            <text fg="gray">No events yet...</text>
          ) : (
            events.slice(-20).map((event, i) => (
              <box key={i} flexDirection="row" gap={1}>
                <text fg="gray">{formatTimestamp(event.timestamp)}</text>
                <text
                  fg={
                    event.type === "iteration_complete"
                      ? "green"
                      : event.type === "error"
                      ? "red"
                      : event.type === "llm_call"
                      ? "yellow"
                      : "cyan"
                  }
                >
                  [{event.type}]
                </text>
                {event.data?.iteration && (
                  <text>iter={event.data.iteration}</text>
                )}
                {event.data?.executionTimeMs && (
                  <text>{formatTime(event.data.executionTimeMs)}</text>
                )}
                {event.error && (
                  <text fg="red">{event.error}</text>
                )}
              </box>
            ))
          )}
        </scrollbox>
      </box>
    </box>
  );
}

// Start the app
createRoot(renderer).render(<MonitorApp />);
