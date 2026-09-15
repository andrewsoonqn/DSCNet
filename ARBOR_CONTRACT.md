# DSCNet Arbor pilot contract

## Target

- Repository: `/Users/andrewsoon/dev/urop/dscnet`
- Base branch: `main`
- Research mode: keyless Pi skills plus restricted Arbor MCP tools
- Scope preference: mixed; prefer scientifically motivated DSConv changes, then judge them by reproducible validation effect

## Objective

Maximize the frozen best-checkpoint `validation.dice` returned by `tools/arbor_eval.py`.

The baseline anchor is `0.8569065042`, from the accepted standard DSCNet validation result. The pilot aims to beat this anchor within three candidate evaluations. Arbor may not substitute another metric or choose a checkpoint after seeing results.

## Evaluation boundary

Use this development evaluator from each candidate worktree:

```sh
DSCNET_ARBOR_DATA_DIR=/Users/andrewsoon/dev/urop/data/minivess-half \
  uv run --locked python tools/arbor_eval.py
```

`tools/arbor_eval.py` is the only experiment entrypoint available to the research loop. It submits one clean formal training run through `expctl`, waits, fetches verified evidence, and returns the selected validation Dice.

MiniVess validation is Arbor's development evidence (`B_dev`). Arbor has no `B_test`. The untouched MiniVess test split, `expctl audit`, MLflow registry operations, and `champion` are outside this contract.

## Edit surface

Arbor may change only:

- `dscnet/models/**`
- `dscnet/training/losses.py`
- `configs/model/**`
- `configs/training/default.yaml`
- `tests/models/**`
- `tests/training/test_losses.py`

Everything else is read-only. In particular, Arbor may not change:

- `tools/arbor_eval.py`, `tools/expctl.py`, or `tools/expctl_remote.py`
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

This contract authorizes only the validation-only pilot after evaluator isolation and preflight tests pass. It does not authorize final-test audit, registration, or alias changes.
