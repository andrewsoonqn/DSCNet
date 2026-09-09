import assert from "node:assert/strict";
import test from "node:test";

import {
  EXPERIMENT_PATHS,
  PI_EXTENSION_VERSION,
  boundedError,
  buildControllerArgs,
  executeControllerProcess,
  parseControllerResult,
  sanitizeTerminalControls,
} from "./commands.mjs";

test("maps bounded experiment input to expctl arguments", () => {
  assert.equal(PI_EXTENSION_VERSION, "1");
  assert.deepEqual(Object.keys(EXPERIMENT_PATHS), [
    "dscnet_standard",
    "dscnet_optimized",
  ]);
  assert.deepEqual(
    buildControllerArgs("submit", {
      experiment: "dscnet_standard",
      action: "resume",
      formal: false,
      allow_dirty: true,
      retry: true,
    }),
    [
      "submit",
      "configs/experiment/dscnet_standard.yaml",
      "--set",
      "action=resume",
      "--set",
      "runtime.formal=false",
      "--set",
      "runtime.allow_dirty=true",
      "--retry",
    ],
  );
});

test("maps every run operation without a host or command input", () => {
  const runId = "0123456789abcdef-a2";
  for (const command of ["status", "cancel", "fetch"]) {
    assert.deepEqual(buildControllerArgs(command, { run_id: runId }), [command, runId]);
  }
  assert.deepEqual(buildControllerArgs("logs", { run_id: runId, lines: 12 }), [
    "logs",
    runId,
    "--lines",
    "12",
  ]);
});

test("rejects inputs outside the controller contract", () => {
  assert.throws(
    () => buildControllerArgs("submit", { experiment: "../../arbitrary" }),
    /Unsupported experiment/,
  );
  assert.throws(
    () => buildControllerArgs("logs", { run_id: "not-a-run", lines: 10 }),
    /Invalid run ID/,
  );
  assert.throws(
    () => buildControllerArgs("logs", { run_id: "0123456789abcdef", lines: 1001 }),
    /between 1 and 1000/,
  );
  assert.throws(
    () => buildControllerArgs("shell", { command: "whoami" }),
    /Unsupported controller command/,
  );
});

test("returns successful controller JSON unchanged", () => {
  const record = { run_id: "0123456789abcdef", state: "RUNNING" };
  assert.deepEqual(
    parseControllerResult("status", {
      code: 0,
      stdout: JSON.stringify(record),
      stderr: "",
    }),
    record,
  );
});

test("invokes only the fixed controller and propagates limits", async () => {
  const calls = [];
  const signal = AbortSignal.abort();
  const record = await executeControllerProcess({
    exec: async (...args) => {
      calls.push(args);
      return { code: 0, stdout: '{"state":"verified"}', stderr: "" };
    },
    repoRoot: "/repo",
    python: "/repo/.venv/bin/python",
    controller: "/repo/tools/expctl.py",
    command: "verify",
    input: { experiment: "dscnet_optimized", action: "prepare" },
    signal,
  });
  assert.deepEqual(record, { state: "verified" });
  assert.deepEqual(calls, [[
    "/usr/bin/env",
    [
      "MLFLOW_DISABLE_AGENT_HINT=1",
      "/repo/.venv/bin/python",
      "/repo/tools/expctl.py",
      "verify",
      "configs/experiment/dscnet_optimized.yaml",
      "--set",
      "action=prepare",
    ],
    { cwd: "/repo", signal, timeout: 300000 },
  ]]);
});

test("strips terminal controls from successful display output", () => {
  const output = sanitizeTerminalControls(JSON.stringify({ log: "ok\u009b31munsafe" }));
  assert.ok(!output.includes("\u009b"));
  assert.match(output, /ok31munsafe/);
});

test("bounds process failures before returning tool errors", async () => {
  await assert.rejects(
    executeControllerProcess({
      exec: async () => { throw new Error("\x1b[31m" + "x".repeat(5000)); },
      repoRoot: "/repo",
      python: "/python",
      controller: "/controller",
      command: "status",
      input: { run_id: "0123456789abcdef" },
    }),
    (error) => !error.message.includes("\x1b") && error.message.length < 4200,
  );
  const bounded = boundedError("\u009dtitle\u009c" + "y".repeat(5000));
  assert.match(bounded, /^\[truncated\]/);
  assert.ok(bounded.length < 4200);
  assert.ok(!bounded.includes("\u009d"));
  assert.ok(!bounded.includes("\u009c"));
});

test("allows bounded controller transfer windows", async () => {
  const calls = [];
  await executeControllerProcess({
    exec: async (...args) => {
      calls.push(args);
      return { code: 0, stdout: "{}", stderr: "" };
    },
    repoRoot: "/repo",
    python: "/python",
    controller: "/controller",
    command: "submit",
    input: { experiment: "dscnet_standard" },
  });
  assert.equal(calls[0][2].timeout, 90 * 60_000);
});

test("turns controller failures into tool errors", () => {
  assert.throws(
    () =>
      parseControllerResult("submit", {
        code: 1,
        stdout: "",
        stderr: JSON.stringify({ command: "submit", error: "\x1b[31munsafe account" }),
      }),
    /expctl submit: unsafe account/,
  );
  assert.throws(
    () => parseControllerResult("status", { code: 0, stdout: "not json", stderr: "" }),
    /invalid JSON/,
  );
});
