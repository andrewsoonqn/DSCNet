# DSCNet usage

Run commands from the repository root. Hydra YAML under `configs/` is the only experiment specification authority.

## Environment

Create or reconcile the local environment from the complete uv lock:

```sh
uv sync --locked
```

Check it without changing packages:

```sh
uv lock --check
uv pip check --python .venv/bin/python
```

`pyproject.toml` and the checked-in `uv.lock` are the only dependency authority. The local `.venv` is a developer environment: `uv sync` installs the checkout editably. Cluster jobs do not use or copy that environment.

DSCNet is repository-bound because experiment identity requires the checked-in Hydra configs, dependency lock, Git evidence, and source snapshot. Standalone wheel installation is not supported; run `python -m dscnet` or the checked-in launchers from a clone.

## Dataset

Local commands expect MiniVess at `Data/MiniVess_Half` unless `data.data_dir` overrides it. The directory must contain matching NIfTI image-label pairs:

```text
Data/MiniVess_Half/
├── train/{image,label}/
├── val/{image,label}/
└── test/{image,label}/
```

The dataset is deliberately excluded from Git.

## Local experiments

Prepare manifests and normalization:

```sh
bash scripts/run_local.sh prepare unet4 data.data_dir=/path/to/minivess-half
```

Train with validation checkpoint selection:

```sh
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_local.sh train unet4 data.data_dir=/path/to/minivess-half
```

Evaluate the frozen test split after fixing the protocol and checkpoint:

```sh
bash scripts/run_local.sh evaluate unet4 \
  data.data_dir=/path/to/minivess-half \
  data.Dir_Weights=/path/to/weights \
  data.model_name_max=<checkpoint-name>
```

Use `unet3`, `unet4`, or `unet5`. Additional arguments are strict Hydra overrides. Local generated model files default to `artifacts/runtime/`.

## Cluster experiments

Formal cluster runs require a clean committed execution tree. Each distinct `uv.lock` is assigned a shared cluster environment whose ID is the full SHA-256 of the lock bytes and whose fixed path is `/home/a/andrewsq/data/urop/environments/<sha256>`. This path is derived and cannot be overridden in experiment YAML.

`expctl submit` verifies source and data, then ensures this environment exists. A ready environment is used immediately. Otherwise the controller submits exactly one fixed CPU setup job and submits the experiment with an `afterok` dependency on it. The setup uses `uv sync --locked --no-install-project`, so DSCNet itself continues to execute from the immutable run source through `PYTHONPATH`; it is never installed into the shared environment. Setup failure is fail-closed and requires an explicit experiment `--retry`.

1. Validate without submitting:

   ```sh
   .venv/bin/python tools/expctl.py verify \
     configs/experiment/dscnet_standard.yaml --set action=prepare
   ```

2. Submit once and save the returned run ID:

   ```sh
   .venv/bin/python tools/expctl.py submit \
     configs/experiment/dscnet_standard.yaml --set action=prepare
   ```

3. Check scheduler state and bounded logs:

   ```sh
   .venv/bin/python tools/expctl.py status <run-id>
   .venv/bin/python tools/expctl.py logs <run-id>
   ```

4. After the state is `COMPLETED`, retrieve and checksum-verify the declared artifacts:

   ```sh
   .venv/bin/python tools/expctl.py fetch <run-id>
   ```

The fetched recovery bundle contains the resolved config, Git evidence, uv-lock/environment evidence, dataset hashes, exact source snapshot, submission receipt, stable logs, final metrics when produced, and latest and best checkpoints when produced.

After cluster cutover, administrators can invoke the same environment lifecycle directly from the allowlisted checkout:

```sh
ssh xlogin1 -- /usr/bin/python3 \
  /home/a/andrewsq/dev/urop/dscnet/tools/expctl_remote.py environment-ensure \
  --source-root /home/a/andrewsq/dev/urop/dscnet \
  --uv-lock-sha256 "$(sha256sum uv.lock | cut -d' ' -f1)" \
  --account allusers
```

Do not copy cluster environments between lock IDs or hosts. Old versioned environments are not deleted automatically; retirement is an explicit administrative operation after confirming that no retained run needs them.
### Resume interrupted training

Point `data.Dir_Weights` at the fetched weights directory, retain the recorded `data.model_name`, set `training.if_retrain=false`, set `training.start_train_epoch` to the checkpoint's next epoch, and submit with `--retry`. `expctl` creates a new immutable attempt and verifies the staged latest checkpoint.

A checkpoint's experiment digest includes its source snapshot. The package restructuring intentionally creates a new source identity. Resume older runs from their archived source snapshot; do not relabel an old checkpoint as a new-source run. Evaluation remains state-dictionary compatible while model attributes and checkpoint metadata remain unchanged.

## Temporary MLflow UI

Training does not require the UI. Starting the UI requires the cluster environment for the current checked-in `uv.lock` to have valid ready evidence. Submit an experiment (which ensures it automatically) or run `environment-ensure` first. `ui status` and `ui stop` continue to inspect or stop an existing reader even if the checkout lock later changes or its current environment is absent.

```sh
.venv/bin/python tools/expctl.py ui start
.venv/bin/python tools/expctl.py ui status
.venv/bin/python tools/expctl.py ui stop
```

Open the URL returned by `ui start` and use its generated credentials. Both the remote reader and local SSH tunnel bind to `127.0.0.1`; basic authentication also protects the reader because the login node is shared. The configured 60-minute timeout stops the reader and tunnel without interrupting training.

Cluster MLflow metadata lives at `/home/a/andrewsq/data/urop/experiments/mlflow.db`; artifacts live at `/home/a/andrewsq/data/urop/experiments/mlflow-artifacts`. Automatic retrieval rejects files above 8 GiB and runs above 12 GiB.
