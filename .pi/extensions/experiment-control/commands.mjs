export const PI_EXTENSION_VERSION = "1";

export const EXPERIMENT_PATHS = Object.freeze({
  dscnet_standard: "configs/experiment/dscnet_standard.yaml",
  dscnet_optimized: "configs/experiment/dscnet_optimized.yaml",
});

const ACTIONS = new Set(["prepare", "train", "resume", "evaluate"]);
const RUN_ID = /^[0-9a-f]{16}(?:-a[1-9][0-9]*)?$/;

function experimentPath(name) {
  const path = EXPERIMENT_PATHS[name];
  if (!path) throw new Error(`Unsupported experiment: ${name}`);
  return path;
}

function runId(value) {
  if (!RUN_ID.test(value)) throw new Error(`Invalid run ID: ${value}`);
  return value;
}

export function buildControllerArgs(command, input) {
  if (command === "verify" || command === "submit") {
    const args = [command, experimentPath(input.experiment)];
    if (input.action !== undefined) {
      if (!ACTIONS.has(input.action)) throw new Error(`Unsupported action: ${input.action}`);
      args.push("--set", `action=${input.action}`);
    }
    if (input.formal !== undefined) {
      args.push("--set", `runtime.formal=${String(input.formal).toLowerCase()}`);
    }
    if (input.allow_dirty !== undefined) {
      args.push("--set", `runtime.allow_dirty=${String(input.allow_dirty).toLowerCase()}`);
    }
    if (command === "submit" && input.retry === true) args.push("--retry");
    return args;
  }

  if (["status", "cancel", "fetch"].includes(command)) {
    return [command, runId(input.run_id)];
  }

  if (command === "logs") {
    const lines = input.lines ?? 200;
    if (!Number.isInteger(lines) || lines < 1 || lines > 1000) {
      throw new Error("Log line count must be between 1 and 1000");
    }
    return [command, runId(input.run_id), "--lines", String(lines)];
  }

  throw new Error(`Unsupported controller command: ${command}`);
}

export function sanitizeTerminalControls(value) {
  return String(value ?? "")
    .replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))/g, "")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, "");
}

export function boundedError(value) {
  const plain = sanitizeTerminalControls(value).trim();
  if (!plain) return "no detail";
  return plain.length > 4096 ? `[truncated]\n${plain.slice(-4096)}` : plain;
}

export function parseControllerResult(command, result) {
  const stdout = result.stdout.trim();
  if (result.code !== 0) {
    const stderr = result.stderr.trim();
    let message = stderr || stdout || `failed with exit code ${result.code}`;
    try {
      const record = JSON.parse(stderr);
      if (typeof record.error === "string") message = record.error;
    } catch {
      // Keep the process error text when stderr is not JSON.
    }
    throw new Error(`expctl ${command}: ${boundedError(message)}`);
  }
  if (!stdout) throw new Error(`expctl ${command} returned no JSON record`);
  try {
    return JSON.parse(stdout);
  } catch {
    throw new Error(`expctl ${command} returned invalid JSON`);
  }
}

export async function executeControllerProcess({
  exec,
  repoRoot,
  python,
  controller,
  command,
  input,
  signal,
}) {
  const args = buildControllerArgs(command, input);
  const timeout = ["submit", "fetch"].includes(command)
    ? 90 * 60_000
    : 5 * 60_000;
  let result;
  try {
    result = await exec(
      "/usr/bin/env",
      ["MLFLOW_DISABLE_AGENT_HINT=1", python, controller, ...args],
      { cwd: repoRoot, signal, timeout },
    );
  } catch (error) {
    throw new Error(`expctl ${command}: ${boundedError(error?.message ?? error)}`);
  }
  return parseControllerResult(command, result);
}
