import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateTail,
  withFileMutationQueue,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
  PI_EXTENSION_VERSION,
  executeControllerProcess,
  sanitizeTerminalControls,
} from "./commands.mjs";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(EXTENSION_DIR, "../../..");
const PYTHON = join(REPO_ROOT, ".venv", "bin", "python");
const CONTROLLER = join(REPO_ROOT, "tools", "expctl.py");
const RUN_ID_PATTERN = "^[0-9a-f]{16}(?:-a[1-9][0-9]*)?$";

const experimentFields = {
  experiment: StringEnum(["dscnet_standard", "dscnet_optimized"] as const, {
    description: "Checked-in experiment configuration",
  }),
  action: Type.Optional(
    StringEnum(["prepare", "train", "resume", "evaluate"] as const),
  ),
  formal: Type.Optional(Type.Boolean()),
  allow_dirty: Type.Optional(Type.Boolean()),
};

const experimentParameters = Type.Object(experimentFields, {
  additionalProperties: false,
});
const submitParameters = Type.Object(
  { ...experimentFields, retry: Type.Optional(Type.Boolean()) },
  { additionalProperties: false },
);
const runParameters = Type.Object(
  { run_id: Type.String({ pattern: RUN_ID_PATTERN }) },
  { additionalProperties: false },
);
const logsParameters = Type.Object(
  {
    run_id: Type.String({ pattern: RUN_ID_PATTERN }),
    lines: Type.Optional(Type.Integer({ minimum: 1, maximum: 1000, default: 200 })),
  },
  { additionalProperties: false },
);

interface ToolDetails {
  command: string;
  extensionVersion: string;
  record?: unknown;
  fullOutputPath?: string;
  truncated: boolean;
}

export default function (pi: ExtensionAPI) {
  async function executeController(
    command: string,
    params: Record<string, unknown>,
    signal?: AbortSignal,
    onUpdate?: (result: { content: Array<{ type: "text"; text: string }> }) => void,
  ) {
    onUpdate?.({
      content: [{ type: "text", text: `Running expctl ${command}...` }],
    });
    const record = await executeControllerProcess({
      exec: pi.exec.bind(pi),
      repoRoot: REPO_ROOT,
      python: PYTHON,
      controller: CONTROLLER,
      command,
      input: params,
      signal,
    });
    const output = sanitizeTerminalControls(JSON.stringify(record, null, 2)) + "\n";
    const truncation = truncateTail(output, {
      maxLines: DEFAULT_MAX_LINES,
      maxBytes: DEFAULT_MAX_BYTES,
    });
    const details: ToolDetails = {
      command,
      extensionVersion: PI_EXTENSION_VERSION,
      truncated: truncation.truncated,
    };
    let text = truncation.content;
    if (truncation.truncated) {
      const directory = await mkdtemp(join(tmpdir(), "pi-expctl-"));
      const path = join(directory, `${command}.json`);
      await withFileMutationQueue(path, () => writeFile(path, output, "utf8"));
      details.fullOutputPath = path;
      text += `\n[Output truncated to ${DEFAULT_MAX_LINES} lines or ${formatSize(DEFAULT_MAX_BYTES)}. Full JSON: ${path}]`;
    } else {
      details.record = record;
    }
    return { content: [{ type: "text" as const, text }], details };
  }

  pi.registerTool({
    name: "experiment_verify",
    label: "Verify Experiment",
    description: "Verify one checked-in DSCNet experiment through expctl without submitting it. Returns expctl JSON; output is limited to 2000 lines or 50KB.",
    parameters: experimentParameters,
    execute: (_id, params, signal, onUpdate) =>
      executeController("verify", params, signal, onUpdate),
  });

  pi.registerTool({
    name: "experiment_submit",
    label: "Submit Experiment",
    description: "Verify, snapshot, synchronize, and submit one checked-in DSCNet experiment through expctl. An identical submit is idempotent; retry explicitly creates another attempt. Returns expctl JSON; output is limited to 2000 lines or 50KB.",
    parameters: submitParameters,
    execute: (_id, params, signal, onUpdate) =>
      executeController("submit", params, signal, onUpdate),
  });

  pi.registerTool({
    name: "experiment_status",
    label: "Experiment Status",
    description: "Read status for a recorded DSCNet experiment run through expctl. Returns expctl JSON; output is limited to 2000 lines or 50KB.",
    parameters: runParameters,
    execute: (_id, params, signal, onUpdate) =>
      executeController("status", params, signal, onUpdate),
  });

  pi.registerTool({
    name: "experiment_logs",
    label: "Experiment Logs",
    description: "Read a bounded tail of logs for a recorded DSCNet experiment run through expctl. Returns expctl JSON; output is limited to 2000 lines or 50KB.",
    parameters: logsParameters,
    execute: (_id, params, signal, onUpdate) =>
      executeController("logs", params, signal, onUpdate),
  });

  pi.registerTool({
    name: "experiment_cancel",
    label: "Cancel Experiment",
    description: "Cancel only the Slurm job identifier already recorded for a DSCNet experiment run through expctl. Returns expctl JSON; output is limited to 2000 lines or 50KB.",
    parameters: runParameters,
    execute: (_id, params, signal, onUpdate) =>
      executeController("cancel", params, signal, onUpdate),
  });

  pi.registerTool({
    name: "experiment_fetch",
    label: "Fetch Experiment",
    description: "Fetch only declared artifacts for a recorded DSCNet experiment run and verify their checksums through expctl. Returns expctl JSON; output is limited to 2000 lines or 50KB.",
    parameters: runParameters,
    execute: (_id, params, signal, onUpdate) =>
      executeController("fetch", params, signal, onUpdate),
  });
}
