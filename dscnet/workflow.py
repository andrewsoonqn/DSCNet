# -*- coding: utf-8 -*-
"""Dispatch a validated DSCNet runtime configuration to one pipeline action."""

from __future__ import annotations

import os
from pathlib import Path
import random


def Create_files(args):
    print("0 Start all process ...")
    for path in (args.Dir_Txt, args.Dir_Log, args.Dir_Save, args.Dir_Weights):
        Path(path).mkdir(parents=True, exist_ok=True)


def _require_files(paths, action):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{action} requires prepared files: {', '.join(missing)}"
        )


def _require_any_file(paths, action):
    if not any(Path(path).is_file() for path in paths):
        raise FileNotFoundError(
            f"{action} requires one checkpoint: {', '.join(map(str, paths))}"
        )


def _require_directories(paths, action):
    missing = [str(path) for path in paths if not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{action} requires dataset directories: {', '.join(missing)}"
        )


def validate_action_artifacts(args):
    """Fail before dispatch when an action's declared inputs do not exist."""
    if args.action == "prepare":
        _require_directories(
            [
                args.Tr_Image_dir,
                args.Tr_Label_dir,
                args.Va_Image_dir,
                args.Va_Label_dir,
                args.Te_Image_dir,
                args.Te_Label_dir,
            ],
            "preparation",
        )
    elif args.action == "train":
        _require_files(
            [
                args.Meanstd_path,
                args.Image_Tr_txt,
                args.Label_Tr_txt,
                args.Image_Va_txt,
                args.Label_Va_txt,
            ],
            "training",
        )
        if not args.if_retrain:
            _require_files(
                [Path(args.Dir_Weights) / args.model_name],
                "resumed training",
            )
    elif args.action == "evaluate":
        _require_files(
            [args.Meanstd_path, args.Image_Te_txt, args.Label_Te_txt],
            "evaluation",
        )
        _require_any_file(
            [
                Path(args.Dir_Weights) / args.model_name_max,
                Path(args.Dir_Weights) / args.model_name,
            ],
            "evaluation",
        )
    else:
        raise ValueError(f"unsupported action: {args.action}")


def apply_reproducibility(seed, deterministic, deterministic_warn_only):
    """Apply the resolved reproducibility settings before pipeline imports."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        deterministic, warn_only=deterministic_warn_only
    )
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def Process(args):
    if not getattr(args, "config_digest", None):
        raise RuntimeError("a resolved Hydra experiment digest is required")
    validate_action_artifacts(args)
    apply_reproducibility(
        args.seed, args.deterministic, args.deterministic_warn_only
    )
    Create_files(args)

    if args.action == "prepare":
        from dscnet.data.normalization import Getmeanstd
        from dscnet.data.manifests import Generate_Paired_Txt

        Getmeanstd(args.Tr_Image_dir, args.Meanstd_path)
        Generate_Paired_Txt(
            args.Tr_Image_dir,
            args.Tr_Label_dir,
            args.Image_Tr_txt,
            args.Label_Tr_txt,
        )
        Generate_Paired_Txt(
            args.Va_Image_dir,
            args.Va_Label_dir,
            args.Image_Va_txt,
            args.Label_Va_txt,
        )
        Generate_Paired_Txt(
            args.Te_Image_dir,
            args.Te_Label_dir,
            args.Image_Te_txt,
            args.Label_Te_txt,
        )
        return None

    from importlib import import_module

    operation = "Train" if args.action == "train" else "Evaluate"
    pipeline_name = (
        "dscnet.training.optimized"
        if args.training_pipeline == "optimized"
        else "dscnet.training.standard"
    )
    pipeline = import_module(pipeline_name)
    return getattr(pipeline, operation)(args)


def main(argv=None):
    """Keep the historical script path as a Hydra-owned entry point."""
    from dscnet.experiment.run import main as experiment_main

    return experiment_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
