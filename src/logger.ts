export type RLMLogLevel =
  | "silent"
  | "error"
  | "warn"
  | "info"
  | "debug"
  | "trace";

export interface RLMLogger {
  error?(message: string, meta?: Record<string, unknown>): void;
  warn?(message: string, meta?: Record<string, unknown>): void;
  info?(message: string, meta?: Record<string, unknown>): void;
  debug?(message: string, meta?: Record<string, unknown>): void;
  trace?(message: string, meta?: Record<string, unknown>): void;
}

const LOG_LEVEL_PRIORITY: Record<RLMLogLevel, number> = {
  silent: 0,
  error: 1,
  warn: 2,
  info: 3,
  debug: 4,
  trace: 5,
};

const NOOP_LOGGER: Required<RLMLogger> = {
  error: () => {},
  warn: () => {},
  info: () => {},
  debug: () => {},
  trace: () => {},
};

type LogMethod = Exclude<RLMLogLevel, "silent">;

export interface LoggerOptions {
  logger?: RLMLogger;
  logLevel?: RLMLogLevel;
}

export function resolveLogLevel(options: LoggerOptions): RLMLogLevel {
  if (options.logLevel) {
    return options.logLevel;
  }

  return "silent";
}

export function createLogger(options: LoggerOptions = {}) {
  const logger = {
    ...NOOP_LOGGER,
    ...options.logger,
  };
  const logLevel = resolveLogLevel(options);

  return {
    level: logLevel,
    error(message: string, meta?: Record<string, unknown>) {
      if (shouldLog(logLevel, "error")) {
        logger.error(message, meta);
      }
    },
    warn(message: string, meta?: Record<string, unknown>) {
      if (shouldLog(logLevel, "warn")) {
        logger.warn(message, meta);
      }
    },
    info(message: string, meta?: Record<string, unknown>) {
      if (shouldLog(logLevel, "info")) {
        logger.info(message, meta);
      }
    },
    debug(message: string, meta?: Record<string, unknown>) {
      if (shouldLog(logLevel, "debug")) {
        logger.debug(message, meta);
      }
    },
    trace(message: string, meta?: Record<string, unknown>) {
      if (shouldLog(logLevel, "trace")) {
        logger.trace(message, meta);
      }
    },
  };
}

function shouldLog(currentLevel: RLMLogLevel, method: LogMethod): boolean {
  return LOG_LEVEL_PRIORITY[currentLevel] >= LOG_LEVEL_PRIORITY[method];
}
