export type RLMOutputSchema = unknown;

export interface RLMOutputValidationResult {
  success: boolean;
  value?: unknown;
  error?: string;
}

type JsonSchema = {
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
  additionalProperties?: boolean | JsonSchema;
};

export function describeOutputSchema(schema: RLMOutputSchema): string {
  if (!schema) {
    return "";
  }

  const zodJsonSchema = tryCall(schema, "toJSONSchema");
  if (zodJsonSchema !== undefined) {
    return JSON.stringify(zodJsonSchema);
  }

  return JSON.stringify(schema);
}

export function validateOutputSchema(
  value: unknown,
  schema: RLMOutputSchema | undefined
): RLMOutputValidationResult {
  if (!schema) {
    return { success: true, value };
  }

  const zodLike = schema as {
    safeParse?: (value: unknown) => { success: boolean; data?: unknown; error?: unknown };
    parse?: (value: unknown) => unknown;
  };

  if (typeof zodLike.safeParse === "function") {
    const result = zodLike.safeParse(value);
    return result.success
      ? { success: true, value: result.data }
      : { success: false, error: formatValidationError(result.error) };
  }

  if (typeof zodLike.parse === "function") {
    try {
      return { success: true, value: zodLike.parse(value) };
    } catch (error) {
      return { success: false, error: formatValidationError(error) };
    }
  }

  const errors: string[] = [];
  validateJsonSchema(value, schema as JsonSchema, "$", errors);
  return errors.length === 0
    ? { success: true, value }
    : { success: false, error: errors.join("; ") };
}

export function parseFinalTextForSchema(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function tryCall(target: unknown, method: string): unknown {
  const fn = (target as Record<string, unknown> | null)?.[method];
  if (typeof fn !== "function") {
    return undefined;
  }
  try {
    return fn.call(target);
  } catch {
    return undefined;
  }
}

function formatValidationError(error: unknown): string {
  if (error && typeof error === "object" && "issues" in error) {
    const issues = (error as { issues?: unknown }).issues;
    if (Array.isArray(issues)) {
      return issues
        .map((issue) => {
          const item = issue as { path?: unknown[]; message?: string };
          const path = item.path && item.path.length > 0 ? item.path.join(".") : "$";
          return `${path}: ${item.message ?? "invalid value"}`;
        })
        .join("; ");
    }
  }

  return error instanceof Error ? error.message : String(error);
}

function validateJsonSchema(
  value: unknown,
  schema: JsonSchema,
  path: string,
  errors: string[]
): void {
  if (schema.enum && !schema.enum.some((item) => Object.is(item, value))) {
    errors.push(`${path}: expected one of ${JSON.stringify(schema.enum)}`);
    return;
  }

  const type = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : [];
  if (type.length > 0 && !type.some((item) => matchesJsonType(value, item))) {
    errors.push(`${path}: expected ${type.join(" or ")}, got ${jsonTypeOf(value)}`);
    return;
  }

  if (schema.type === "object" || schema.properties || schema.required) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      errors.push(`${path}: expected object, got ${jsonTypeOf(value)}`);
      return;
    }

    const objectValue = value as Record<string, unknown>;
    for (const key of schema.required ?? []) {
      if (!(key in objectValue)) {
        errors.push(`${path}.${key}: required property missing`);
      }
    }

    for (const [key, childSchema] of Object.entries(schema.properties ?? {})) {
      if (key in objectValue) {
        validateJsonSchema(objectValue[key], childSchema, `${path}.${key}`, errors);
      }
    }
  }

  if (schema.type === "array" || schema.items) {
    if (!Array.isArray(value)) {
      errors.push(`${path}: expected array, got ${jsonTypeOf(value)}`);
      return;
    }

    if (schema.items) {
      value.forEach((item, index) =>
        validateJsonSchema(item, schema.items as JsonSchema, `${path}[${index}]`, errors)
      );
    }
  }
}

function matchesJsonType(value: unknown, type: string): boolean {
  switch (type) {
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "array":
      return Array.isArray(value);
    case "object":
      return !!value && typeof value === "object" && !Array.isArray(value);
    case "null":
      return value === null;
    default:
      return true;
  }
}

function jsonTypeOf(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}
