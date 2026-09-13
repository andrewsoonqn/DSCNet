from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_OUTPUT_DIR = DEFAULT_LOGS_DIR / "parsed"

EPOCH_SUMMARY_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\["
    r"(?:S3_Train_Process|standard|optimized)\.py\] "
    r"Epoch:\[(?P<epoch>\d+)/(?P<total_epochs>\d+)\]\s+"
    r"lr=(?P<lr>[0-9.]+)\s+loss=(?P<loss>[0-9.]+)\s+"
    r"(?:counter=(?P<counter>\d+)\s+)?"
    r"dice_mean=(?P<dice_mean>[0-9.]+)\s+"
    r"max_dice=(?P<max_dice>[0-9.]+)\s+saved_dice=(?P<saved_dice>[0-9.]+)$"
)

ITERATION_PATTERN = re.compile(
    r"^Epoch:\s+\[(?P<epoch>\d+)/(?P<total_epochs>\d+)\]\s+"
    r"Iter:\s+\[(?P<iteration>\d+)/(?P<iterations_per_epoch>\d+)\]\s+"
    r"Lr:\s+\[(?P<lr>[0-9.]+)\]\s+Loss\s+(?P<loss>[0-9.]+)$"
)

DICE_ROW_PATTERN = re.compile(
    r"^(?P<case>[^\s]+\.nii\.gz)\s+(?P<primary>[-+0-9.eEnNaAfF]+)\s+(?P<secondary>[-+0-9.eEnNaAfF]+)$"
)
CLDICE_ROW_PATTERN = re.compile(r"^(?P<case>[^\s]+\.nii\.gz)\s+(?P<value>[-+0-9.eE]+)$")
PRA_ROW_PATTERN = re.compile(
    r"^(?P<case>[^\s]+\.nii\.gz)\s+"
    r"(?P<precision>[-+0-9.eE]+)\s+"
    r"(?P<recall>[-+0-9.eE]+)\s+"
    r"(?P<accuracy>[-+0-9.eE]+)$"
)
CASE_PATH_PATTERN = re.compile(
    r"^Data/.+/(?P<split>train|val|test)/image/(?P<case>[^/]+\.nii\.gz)$"
)
TIMESTAMP_PATTERN = re.compile(r"^\[(?P<timestamp>[^\]]+)\]")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"nan", "none", ""}:
        return None
    return float(value)


def parse_datetime(value: str | None, fmt: str) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value.strip(), fmt)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


@dataclass
class ParsedLog:
    summary: pd.DataFrame
    epochs: pd.DataFrame
    iterations: pd.DataFrame
    case_metrics: pd.DataFrame


def build_empty_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def summarize_split(
    case_metrics: pd.DataFrame, split: str, metric: str
) -> tuple[float | None, float | None, float | None, int]:
    if case_metrics.empty or metric not in case_metrics.columns:
        return None, None, None, 0

    subset = case_metrics[
        (case_metrics["split"] == split) & case_metrics[metric].notna()
    ]
    if subset.empty:
        return None, None, None, 0

    return (
        float(subset[metric].mean()),
        float(subset[metric].min()),
        float(subset[metric].std(ddof=0)),
        int(subset[metric].count()),
    )


def parse_log(log_path: Path) -> ParsedLog:
    lines = log_path.read_text(errors="replace").splitlines()
    run_id = log_path.stem

    metadata: dict[str, object] = {
        "run_id": run_id,
        "log_file": log_path.name,
        "job_id": None,
        "job_name": None,
        "partition": None,
        "node": None,
        "working_dir": None,
        "start_time": None,
        "end_time": None,
        "duration_human": None,
        "duration_seconds": None,
        "observed_duration_seconds": None,
        "run_label": "DSCNet_3D_default",
        "run_variant": "default",
        "unet_layers": None,
        "gpu": None,
        "gpu_mem_gb": None,
        "command": None,
        "exit_code": None,
        "pipeline_finished": False,
        "last_logged_at": None,
        "status": "unknown",
    }

    epoch_rows: list[dict[str, object]] = []
    iteration_rows: list[dict[str, object]] = []
    case_metrics: dict[tuple[str, str], dict[str, object]] = {}

    current_eval_split: str | None = None
    current_metric: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if current_metric == "dice":
            match = DICE_ROW_PATTERN.match(line)
            if match and current_eval_split:
                case_key = (current_eval_split, match.group("case"))
                row = case_metrics.setdefault(
                    case_key,
                    {
                        "run_id": run_id,
                        "split": current_eval_split,
                        "case_name": match.group("case"),
                        "dice": None,
                        "dice_secondary": None,
                        "cldice": None,
                        "precision": None,
                        "recall": None,
                        "accuracy": None,
                    },
                )
                row["dice"] = parse_float(match.group("primary"))
                row["dice_secondary"] = parse_float(match.group("secondary"))
                i += 1
                continue
            current_metric = None

        if current_metric == "cldice":
            match = CLDICE_ROW_PATTERN.match(line)
            if match and current_eval_split:
                case_key = (current_eval_split, match.group("case"))
                row = case_metrics.setdefault(
                    case_key,
                    {
                        "run_id": run_id,
                        "split": current_eval_split,
                        "case_name": match.group("case"),
                        "dice": None,
                        "dice_secondary": None,
                        "cldice": None,
                        "precision": None,
                        "recall": None,
                        "accuracy": None,
                    },
                )
                row["cldice"] = float(match.group("value"))
                i += 1
                continue
            current_metric = None

        if current_metric == "pra":
            match = PRA_ROW_PATTERN.match(line)
            if match and current_eval_split:
                case_key = (current_eval_split, match.group("case"))
                row = case_metrics.setdefault(
                    case_key,
                    {
                        "run_id": run_id,
                        "split": current_eval_split,
                        "case_name": match.group("case"),
                        "dice": None,
                        "dice_secondary": None,
                        "cldice": None,
                        "precision": None,
                        "recall": None,
                        "accuracy": None,
                    },
                )
                row["precision"] = float(match.group("precision"))
                row["recall"] = float(match.group("recall"))
                row["accuracy"] = float(match.group("accuracy"))
                i += 1
                continue
            current_metric = None

        timestamp_match = TIMESTAMP_PATTERN.match(line)
        if timestamp_match:
            metadata["last_logged_at"] = timestamp_match.group("timestamp")

        line_clean = line.strip()
        if line_clean.startswith("Job ID"):
            metadata["job_id"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Job Name"):
            metadata["job_name"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Partition"):
            metadata["partition"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Node"):
            metadata["node"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Working Dir"):
            metadata["working_dir"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Start Time"):
            metadata["start_time"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Ended"):
            metadata["end_time"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Duration"):
            duration_text = line_clean.split(":", 1)[1].strip()
            metadata["duration_human"] = duration_text
            seconds_match = re.search(r"\((\d+)s total\)", duration_text)
            metadata["duration_seconds"] = (
                int(seconds_match.group(1)) if seconds_match else None
            )
        elif line_clean.startswith("Run label"):
            metadata["run_label"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("U-Net layers"):
            metadata["unet_layers"] = int(line_clean.split(":", 1)[1].strip())
        elif line_clean.startswith("GPU") and not line_clean.startswith("GPU "):
            metadata["gpu"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("GPU Mem"):
            gpu_mem_text = line_clean.split(":", 1)[1].strip().split()[0]
            metadata["gpu_mem_gb"] = float(gpu_mem_text)
        elif line_clean.startswith("Command"):
            metadata["command"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("Exit code"):
            exit_code_text = line_clean.split(":", 1)[1].strip()
            metadata["exit_code"] = (
                int(exit_code_text) if exit_code_text.isdigit() else exit_code_text
            )
        elif line_clean == "Pipeline Finished":
            metadata["pipeline_finished"] = True
        elif line == "Predict test data":
            current_eval_split = None
        elif line.startswith("Data/") and line.endswith(".nii.gz"):
            case_match = CASE_PATH_PATTERN.match(line)
            if case_match:
                current_eval_split = case_match.group("split")
        elif line == "Dice:":
            current_metric = "dice"
        elif line == "clDice:":
            current_metric = "cldice"
        elif line == "Precision, Recall, Accuracy:":
            current_metric = "pra"

        epoch_match = EPOCH_SUMMARY_PATTERN.match(line)
        if epoch_match:
            epoch_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": epoch_match.group("timestamp"),
                    "epoch": int(epoch_match.group("epoch")),
                    "total_epochs": int(epoch_match.group("total_epochs")),
                    "lr": float(epoch_match.group("lr")),
                    "loss": float(epoch_match.group("loss")),
                    "counter": int(epoch_match.group("counter"))
                    if epoch_match.group("counter")
                    else None,
                    "dice_mean": float(epoch_match.group("dice_mean")),
                    "max_dice": float(epoch_match.group("max_dice")),
                    "saved_dice": float(epoch_match.group("saved_dice")),
                }
            )

        iteration_match = ITERATION_PATTERN.match(line)
        if iteration_match:
            epoch_value = int(iteration_match.group("epoch"))
            iteration_value = int(iteration_match.group("iteration"))
            total_iterations = int(iteration_match.group("iterations_per_epoch"))
            iteration_rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch_value,
                    "total_epochs": int(iteration_match.group("total_epochs")),
                    "iteration": iteration_value,
                    "iterations_per_epoch": total_iterations,
                    "global_step": (epoch_value - 1) * total_iterations
                    + iteration_value,
                    "lr": float(iteration_match.group("lr")),
                    "loss": float(iteration_match.group("loss")),
                }
            )

        i += 1

    if metadata["run_label"] in {"DSCNet_3D_default", "dscnet-default"}:
        metadata["run_label"] = "Test_Run_unet3"
    metadata["run_label_slug"] = slugify(metadata["run_label"])

    if metadata["run_label_slug"] == "test-run-unet3":
        metadata["run_variant"] = "unet3"
        if metadata["unet_layers"] is None:
            metadata["unet_layers"] = 3

    if metadata["unet_layers"] is not None:
        metadata["run_variant"] = f"unet{metadata['unet_layers']}"

    epochs_df = pd.DataFrame(epoch_rows)
    iterations_df = pd.DataFrame(iteration_rows)
    case_df = pd.DataFrame(case_metrics.values())

    if not epochs_df.empty:
        best_epoch_idx = epochs_df["dice_mean"].idxmax()
        metadata["best_train_dice_mean"] = float(
            epochs_df.loc[best_epoch_idx, "dice_mean"]
        )
        metadata["best_train_dice_epoch"] = int(epochs_df.loc[best_epoch_idx, "epoch"])
        metadata["best_saved_dice"] = float(epochs_df["saved_dice"].max())
        metadata["last_epoch"] = int(epochs_df["epoch"].max())
        metadata["final_epoch_loss"] = float(
            epochs_df.sort_values("epoch").iloc[-1]["loss"]
        )
    else:
        metadata["best_train_dice_mean"] = None
        metadata["best_train_dice_epoch"] = None
        metadata["best_saved_dice"] = None
        metadata["last_epoch"] = None
        metadata["final_epoch_loss"] = None

    for split in ("val", "test"):
        mean_value, min_value, std_value, count_value = summarize_split(
            case_df, split, "dice"
        )
        metadata[f"{split}_dice_mean"] = mean_value
        metadata[f"{split}_dice_min"] = min_value
        metadata[f"{split}_dice_std"] = std_value
        metadata[f"{split}_case_count"] = count_value

    test_cldice_mean, _, _, _ = summarize_split(case_df, "test", "cldice")
    metadata["test_cldice_mean"] = test_cldice_mean

    for metric in ("precision", "recall", "accuracy"):
        metric_mean, _, _, _ = summarize_split(case_df, "test", metric)
        metadata[f"test_{metric}_mean"] = metric_mean

    start_dt = parse_datetime(metadata["start_time"], "%Y-%m-%d %H:%M:%S")
    end_dt = parse_datetime(metadata["end_time"], "%Y-%m-%d %H:%M:%S")
    if start_dt and end_dt:
        metadata["observed_duration_seconds"] = int((end_dt - start_dt).total_seconds())
    elif start_dt and metadata["last_logged_at"]:
        last_dt = parse_datetime(
            str(metadata["last_logged_at"]), "%Y-%m-%d %H:%M:%S,%f"
        )
        metadata["observed_duration_seconds"] = (
            int((last_dt - start_dt).total_seconds()) if last_dt else None
        )

    if metadata["pipeline_finished"] and metadata["test_case_count"]:
        metadata["status"] = "full_eval"
    elif metadata["val_case_count"]:
        metadata["status"] = "partial_val_only"
    else:
        metadata["status"] = "training_only"

    if (
        metadata["duration_seconds"] is None
        and metadata["observed_duration_seconds"] is not None
    ):
        metadata["duration_seconds"] = metadata["observed_duration_seconds"]
    metadata["duration_hours"] = (
        round(float(metadata["duration_seconds"]) / 3600, 3)
        if metadata["duration_seconds"] is not None
        else None
    )

    summary_df = pd.DataFrame([metadata])
    return ParsedLog(
        summary=summary_df,
        epochs=epochs_df,
        iterations=iterations_df,
        case_metrics=case_df,
    )


def write_per_run_outputs(parsed: ParsedLog, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed.summary.to_csv(output_dir / "run_summary.csv", index=False)
    parsed.epochs.to_csv(output_dir / "epoch_metrics.csv", index=False)
    parsed.iterations.to_csv(output_dir / "iteration_metrics.csv", index=False)
    parsed.case_metrics.to_csv(output_dir / "case_metrics.csv", index=False)


def parse_logs(logs_dir: Path, output_dir: Path) -> None:
    log_paths = sorted(logs_dir.glob("*.out"))
    if not log_paths:
        raise FileNotFoundError(f"No .out files found in {logs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    epochs: list[pd.DataFrame] = []
    iterations: list[pd.DataFrame] = []
    case_metrics: list[pd.DataFrame] = []

    for log_path in log_paths:
        parsed = parse_log(log_path)
        summaries.append(parsed.summary)
        epochs.append(parsed.epochs)
        iterations.append(parsed.iterations)
        case_metrics.append(parsed.case_metrics)

        run_output_dir = output_dir / slugify(parsed.summary.iloc[0]["run_id"])
        write_per_run_outputs(parsed, run_output_dir)

    summary_df = pd.concat(summaries, ignore_index=True).sort_values(
        ["start_time", "run_id"], na_position="last"
    )
    epochs_df = (
        pd.concat(epochs, ignore_index=True)
        if any(not df.empty for df in epochs)
        else build_empty_dataframe(
            [
                "run_id",
                "timestamp",
                "epoch",
                "total_epochs",
                "lr",
                "loss",
                "counter",
                "dice_mean",
                "max_dice",
                "saved_dice",
            ]
        )
    )
    iterations_df = (
        pd.concat(iterations, ignore_index=True)
        if any(not df.empty for df in iterations)
        else build_empty_dataframe(
            [
                "run_id",
                "epoch",
                "total_epochs",
                "iteration",
                "iterations_per_epoch",
                "global_step",
                "lr",
                "loss",
            ]
        )
    )
    case_metrics_df = (
        pd.concat(case_metrics, ignore_index=True)
        if any(not df.empty for df in case_metrics)
        else build_empty_dataframe(
            [
                "run_id",
                "split",
                "case_name",
                "dice",
                "dice_secondary",
                "cldice",
                "precision",
                "recall",
                "accuracy",
            ]
        )
    )

    summary_df.to_csv(output_dir / "all_runs_summary.csv", index=False)
    epochs_df.to_csv(output_dir / "all_runs_epochs.csv", index=False)
    iterations_df.to_csv(output_dir / "all_runs_iterations.csv", index=False)
    case_metrics_df.to_csv(output_dir / "all_runs_case_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse DSCNet training logs into structured CSVs."
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    parse_logs(args.logs_dir, args.output_dir)
    print(f"Parsed logs from {args.logs_dir} into {args.output_dir}")


if __name__ == "__main__":
    main()
