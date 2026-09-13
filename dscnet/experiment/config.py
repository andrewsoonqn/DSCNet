"""Typed Hydra experiment configuration and semantic validation."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class ModelConfig:
    training_pipeline: str = "standard"
    n_channels: int = 1
    n_classes: int = 2
    kernel_size: int = 5
    extend_scope: float = 1.75
    if_offset: bool = True
    n_basic_layer: int = 16
    dim: int = 8
    unet_layers: int = 4


@dataclass
class DataConfig:
    root_dir: str = "artifacts/runtime"
    data_dir: str = "Data/MiniVess_Half"
    run_label: str = "DSCNet_3D"
    Tr_Image_dir: str | None = None
    Va_Image_dir: str | None = None
    Te_Image_dir: str | None = None
    Tr_Label_dir: str | None = None
    Va_Label_dir: str | None = None
    Te_Label_dir: str | None = None
    Meanstd_name: str | None = None
    Meanstd_path: str | None = None
    Dir_Txt: str | None = None
    Dir_Log: str | None = None
    Dir_Save: str | None = None
    Dir_Weights: str | None = None
    Image_Tr_txt: str | None = None
    Image_Va_txt: str | None = None
    Image_Te_txt: str | None = None
    Label_Tr_txt: str | None = None
    Label_Va_txt: str | None = None
    Label_Te_txt: str | None = None
    save_path: str | None = None
    save_path_max: str | None = None
    model_name: str | None = None
    model_name_max: str | None = None
    log_name: str | None = None


@dataclass
class TrainingConfig:
    ROI_shape: list[int] = field(default_factory=lambda: [64, 64, 64])
    batch_size: int = 1
    sample_count: int = 1
    predict_batch_size: int = 4
    lr: float = 1e-4
    min_lr: float = 5e-6
    poly_decay_power: float = 0.9
    beta: float = 1e-2
    min_beta: float = 1e-6
    beta_decay_power: float = 2.0
    use_rlrop: bool = False
    rlr_factor: float = 0.5
    rlr_threshold: float = 0.002
    rlr_patience: int = 10
    rlr_cooldown: int = 2
    start_train_epoch: int = 1
    start_verify_epoch: int = 51
    n_epochs: int = 100
    verify_gap: int = 1
    if_retrain: bool = True
    if_fullprecision: bool = True
    use_earlystop: bool = False
    earlystop_threshold: float = 0.002
    earlystop_patience: int = 30
    seed: int = 2026
    deterministic: bool = True


@dataclass
class MlflowUIConfig:
    enabled: bool = True
    location: str = "login"
    timeout_minutes: int = 60


@dataclass
class MlflowConfig:
    tracking_uri: str = "sqlite:///artifacts/experiments/mlflow.db"
    artifact_root: str = "artifacts/experiments/mlflow-artifacts"
    experiment_name: str = "dscnet"
    ui: MlflowUIConfig = field(default_factory=MlflowUIConfig)


@dataclass
class SlurmConfig:
    host: str = "xlogin1"
    max_concurrent_runs: int = 1
    partition: str = "gpu-long"
    gpu_type: str = "a100-40"
    gpus: int = 1
    memory_gb: int = 32
    cpus: int = 8
    time_hours: int = 48
    account: str | None = None


@dataclass
class RuntimeConfig:
    kind: str = "local"
    experiment_root: str = "artifacts/experiments"
    formal: bool = False
    allow_dirty: bool = True
    gpu_id: str = "0"
    remote_checkout: str = "/home/a/andrewsq/dev/urop/dscnet"
    remote_experiment_root: str = "/home/a/andrewsq/data/urop/experiments"
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


@dataclass
class ExperimentConfig:
    action: str = "train"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def register_configs() -> None:
    store = ConfigStore.instance()
    store.store(name="experiment_schema", node=ExperimentConfig)
    store.store(group="model", name="model_schema", node=ModelConfig)
    store.store(group="data", name="data_schema", node=DataConfig)
    store.store(group="training", name="training_schema", node=TrainingConfig)
    store.store(group="runtime", name="runtime_schema", node=RuntimeConfig)


register_configs()


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Reject unsupported or internally inconsistent experiment values."""
    if config.action not in {"prepare", "train", "evaluate"}:
        raise ValueError(f"unsupported action: {config.action}")
    if config.model.training_pipeline not in {"standard", "optimized"}:
        raise ValueError(
            f"unsupported training pipeline: {config.model.training_pipeline}"
        )
    if config.model.unet_layers not in {3, 4, 5}:
        raise ValueError("unet_layers must be one of 3, 4, or 5")
    if (
        config.model.training_pipeline == "optimized"
        and config.model.unet_layers != 4
    ):
        raise ValueError("optimized pipeline has a fixed unet_layers value of 4")
    if config.model.kernel_size != 5:
        raise ValueError("kernel_size is fixed at the supported value 5")
    if config.model.n_channels <= 0 or config.model.n_classes <= 1:
        raise ValueError("model channel counts must be positive and n_classes >= 2")
    if config.model.n_basic_layer <= 0 or config.model.dim <= 0:
        raise ValueError("model depth and base width must be positive")
    if config.model.extend_scope <= 0:
        raise ValueError("extend_scope must be positive")

    roi = config.training.ROI_shape
    if len(roi) != 3 or any(not isinstance(value, int) or value <= 0 for value in roi):
        raise ValueError("ROI_shape must contain three positive integers")
    divisor = 2 ** (config.model.unet_layers - 1)
    if any(value % divisor for value in roi):
        raise ValueError(
            f"every ROI_shape dimension must be divisible by {divisor} "
            f"for {config.model.unet_layers} U-Net levels"
        )

    positive_ints = {
        "batch_size": config.training.batch_size,
        "sample_count": config.training.sample_count,
        "predict_batch_size": config.training.predict_batch_size,
        "n_epochs": config.training.n_epochs,
        "verify_gap": config.training.verify_gap,
        "start_train_epoch": config.training.start_train_epoch,
        "start_verify_epoch": config.training.start_verify_epoch,
        "rlr_patience": config.training.rlr_patience,
        "earlystop_patience": config.training.earlystop_patience,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    positive_floats = {
        "lr": config.training.lr,
        "min_lr": config.training.min_lr,
        "poly_decay_power": config.training.poly_decay_power,
        "beta": config.training.beta,
        "min_beta": config.training.min_beta,
        "beta_decay_power": config.training.beta_decay_power,
        "rlr_factor": config.training.rlr_factor,
    }
    for name, value in positive_floats.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.training.start_train_epoch > config.training.n_epochs:
        raise ValueError("start_train_epoch must not exceed n_epochs")
    if not (
        config.training.start_train_epoch
        <= config.training.start_verify_epoch
        <= config.training.n_epochs
    ):
        raise ValueError(
            "start_verify_epoch must be between start_train_epoch and n_epochs"
        )
    if not 0 < config.training.rlr_factor < 1:
        raise ValueError("rlr_factor must be between 0 and 1")
    if config.training.rlr_cooldown < 0:
        raise ValueError("rlr_cooldown must be non-negative")
    for name, value in {
        "rlr_threshold": config.training.rlr_threshold,
        "earlystop_threshold": config.training.earlystop_threshold,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.training.min_lr > config.training.lr:
        raise ValueError("min_lr must not exceed lr")
    if config.training.min_beta > config.training.beta:
        raise ValueError("min_beta must not exceed beta")
    if config.training.seed < 0:
        raise ValueError("seed must be non-negative")

    if config.runtime.kind not in {"local", "slurm"}:
        raise ValueError("runtime.kind must be local or slurm")
    if config.runtime.slurm.max_concurrent_runs != 1:
        raise ValueError("the initial SQLite backend permits one concurrent run")
    if config.runtime.mlflow.ui.location not in {"login", "slurm"}:
        raise ValueError("mlflow.ui.location must be login or slurm")
    if not 1 <= config.runtime.mlflow.ui.timeout_minutes <= 480:
        raise ValueError("mlflow.ui.timeout_minutes must be between 1 and 480")
    if config.runtime.formal and config.runtime.allow_dirty:
        raise ValueError("formal runs cannot allow a dirty source tree")
    return config


def load_experiment_config(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    config_dir: str | Path = CONFIG_DIR,
) -> tuple[ExperimentConfig, DictConfig]:
    """Compose a strict Hydra config and return typed and resolved forms."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(config_dir).resolve()),
        job_name="dscnet_config",
    ):
        resolved = compose(config_name=config_name, overrides=list(overrides or ()))
    OmegaConf.resolve(resolved)
    typed = OmegaConf.to_object(resolved)
    if not isinstance(typed, ExperimentConfig):
        raise TypeError("Hydra composition did not produce ExperimentConfig")
    return validate_config(typed), resolved


def resolved_yaml(config: DictConfig) -> str:
    return OmegaConf.to_yaml(config, resolve=True, sort_keys=True)


def identity_config_yaml(config: DictConfig) -> str:
    """Normalize resume-only controls out of experiment identity."""
    identity = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    identity.training.if_retrain = True
    identity.training.start_train_epoch = 1
    return resolved_yaml(identity)


def resolve_runtime_paths(args: Namespace) -> Namespace:
    """Resolve every derived runtime path from the typed experiment config."""
    root = Path(args.root_dir)
    data = Path(args.data_dir)
    args.root_dir = str(root)
    args.data_dir = str(data)

    defaults = {
        "Tr_Image_dir": data / "train" / "image",
        "Va_Image_dir": data / "val" / "image",
        "Te_Image_dir": data / "test" / "image",
        "Tr_Label_dir": data / "train" / "label",
        "Va_Label_dir": data / "val" / "label",
        "Te_Label_dir": data / "test" / "label",
        "Dir_Txt": root / "Txt" / f"Txt_{args.run_label}",
        "Dir_Log": root / "Log" / args.run_label,
        "Dir_Save": root / "Results" / args.run_label,
        "Dir_Weights": root / "Weights" / args.run_label,
    }
    for attribute, default in defaults.items():
        value = getattr(args, attribute)
        setattr(args, attribute, str(Path(value) if value else default))
    args.Dir_Log = str(Path(args.Dir_Log)) + "/"

    txt_dir = Path(args.Dir_Txt)
    manifest_defaults = {
        "Image_Tr_txt": txt_dir / "Image_Tr.txt",
        "Image_Va_txt": txt_dir / "Image_Va.txt",
        "Image_Te_txt": txt_dir / "Image_Te.txt",
        "Label_Tr_txt": txt_dir / "Label_Tr.txt",
        "Label_Va_txt": txt_dir / "Label_Va.txt",
        "Label_Te_txt": txt_dir / "Label_Te.txt",
    }
    for attribute, default in manifest_defaults.items():
        value = getattr(args, attribute)
        setattr(args, attribute, str(Path(value) if value else default))

    if getattr(args, "Meanstd_name", None) is None:
        args.Meanstd_name = f"{args.run_label}_Meanstd.npy"
    meanstd_path = getattr(args, "Meanstd_path", None)
    args.Meanstd_path = str(
        Path(meanstd_path) if meanstd_path else root / args.Meanstd_name
    )
    if getattr(args, "save_path", None) is None:
        args.save_path = str(root / "Results" / args.run_label / "DSCNet")
    if getattr(args, "save_path_max", None) is None:
        args.save_path_max = str(root / "Results" / args.run_label / "DSCNet_max")
    if getattr(args, "model_name", None) is None:
        args.model_name = f"DSCNet_{args.run_label}"
    if getattr(args, "model_name_max", None) is None:
        args.model_name_max = f"DSCNet_{args.run_label}_max"
    if getattr(args, "log_name", None) is None:
        args.log_name = f"DSCNet_{args.run_label}.log"
    return args


def to_runtime_namespace(config: ExperimentConfig) -> Namespace:
    """Flatten the sole typed experiment config for the existing model code."""
    values: dict[str, Any] = {"action": config.action}
    values.update(vars(config.model))
    values.update(vars(config.training))
    values.update(vars(config.data))
    values["ROI_shape"] = tuple(config.training.ROI_shape)
    values["GPU_id"] = config.runtime.gpu_id
    return resolve_runtime_paths(Namespace(**values))
