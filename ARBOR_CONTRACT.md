# DSCNet Arbor pilot contract

## Target

- Repository: `/Users/andrewsoon/dev/urop/dscnet`
- Base branch: `main`
- Research mode: keyless Pi skills plus restricted Arbor MCP tools
- Scope preference: mixed; prefer scientifically motivated DSConv changes, then judge them by reproducible validation effect

## Objective

Maximize the frozen best-checkpoint `validation.dice` returned through the trusted candidate broker.

The baseline anchor is `0.8569065042`, from the accepted standard DSCNet validation result. The pilot aims to beat this anchor within three candidate evaluations. Arbor may not substitute another metric or choose a checkpoint after seeing results.

## Evaluation boundary

The coordinator may call only:

```text
submit_candidate(run_name, node_id)
```

The broker resolves the registered candidate worktree and verifies its branch, clean commit, ancestry, allowed paths, safe Git modes, and three-candidate budget. It pins an immutable detached worktree and performs a non-executing Python syntax check before internally invoking `tools/arbor_eval.py` with the configured MiniVess mirror. The model cannot supply a command, path, metric, Hydra override, retry, or audit option.

`tools/arbor_eval.py` submits one clean formal training run through `expctl` with `runtime.mlflow.isolated_run_store=true`, waits, fetches verified evidence, and reads the validation result produced during that run. The run uses a run-owned SQLite database and artifact root. On Slurm, candidate tests and training execute from the immutable source snapshot with a sanitized environment, closed inherited file descriptors, and a seccomp-BPF filter that allows local Unix sockets but denies creation of every IP socket, including TCP and UDP. It does not rerun validation. The broker returns only candidate status and, after completion, the controller run ID and selected `validation.dice`.

MiniVess validation is Arbor's development evidence (`B_dev`). Arbor has no `B_test`. Training manifests include only train and validation data. The shared runtime configuration still derives canonical test paths, but `action=train` does not read the test manifests or volumes. Isolated research runs are not final candidates and `expctl audit` rejects them. The untouched MiniVess test split, central MLflow, MLflow registry operations, and `champion` are outside the approved workflow.

## Edit surface

Arbor may change only:

- `dscnet/models/**`
- `dscnet/training/losses.py`
- `configs/model/**`
- `configs/training/default.yaml`
- `tests/models/**`
- `tests/training/test_losses.py`

Everything else is read-only. In particular, Arbor may not change:

- `tools/arbor_broker.py`, `tools/arbor_eval.py`, `tools/arbor_preflight.py`, `tools/expctl.py`, or `tools/expctl_remote.py`
- `dscnet/data/**`, `dscnet/evaluation/**`, `dscnet/experiment/**`, or `dscnet/workflow.py`
- data, experiment, or runtime configuration
- dependency files or dataset manifests
- this contract, Arbor configuration, or Pi configuration

## Budget and gates

- Candidate evaluations: 3 maximum
- Concurrent candidate evaluations: 1
- Seed: fixed by the experiment configuration (`2026`)
- Evaluation fidelity: unchanged formal training protocol
- Failed or invalid candidates consume one cycle
- Interaction: autonomous for this explicitly authorized pilot
- Merge policy: no automatic merges; retain candidate branches for manual review
- Finalization: report hypotheses, diffs, run IDs, scores, failures, and estimated costs

This contract authorizes only the validation-only pilot after broker boundary tests pass. It does not authorize final-test audit, registration, or alias changes.

Candidate model and loss code executes in the same training process that sees validation labels and records validation evidence. The cluster does not provide a compatible unprivileged filesystem sandbox for CUDA, so candidate code is not treated as an adversarial binary: it can read same-user cluster files and can inspect test paths derived by the shared configuration. Pi reviews each candidate diff before calling the broker, but the broker does not cryptographically prove that review. The broker denies new IP socket creation, fixes the executable path, and prevents changes to protected Git paths. Every candidate is reviewed again before any manual merge. Candidate scores remain research evidence, not a security proof against evaluator manipulation.
