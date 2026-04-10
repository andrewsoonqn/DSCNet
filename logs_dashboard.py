# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "pandas==3.0.2",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from pathlib import Path

    sns.set_theme(style="whitegrid")
    return Path, mo, pd, plt, sns


@app.cell
def _(Path):
    project_root = Path(__file__).resolve().parent
    parsed_dir = project_root / "logs" / "parsed"
    summary_path = parsed_dir / "all_runs_summary.csv"
    epochs_path = parsed_dir / "all_runs_epochs.csv"
    iterations_path = parsed_dir / "all_runs_iterations.csv"
    required_paths = [summary_path, epochs_path, iterations_path]
    return epochs_path, iterations_path, required_paths, summary_path


@app.cell
def _(mo, required_paths):
    missing_paths = [path.name for path in required_paths if not path.exists()]
    mo.stop(
        bool(missing_paths),
        mo.md(
            "Run `python utils/parse_training_logs.py` first. Missing files: "
            + ", ".join(missing_paths)
        ),
    )
    return


@app.cell
def _(epochs_path, iterations_path, pd, summary_path):
    summary_df = pd.read_csv(summary_path)
    epochs_df = pd.read_csv(epochs_path)
    iterations_df = pd.read_csv(iterations_path)

    for time_column in ["start_time", "end_time", "last_logged_at"]:
        if time_column in summary_df.columns:
            if time_column == "last_logged_at":
                summary_df[time_column] = pd.to_datetime(
                    summary_df[time_column],
                    format="%Y-%m-%d %H:%M:%S,%f",
                    errors="coerce",
                )
            else:
                summary_df[time_column] = pd.to_datetime(
                    summary_df[time_column],
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce",
                )

    if "timestamp" in epochs_df.columns:
        epochs_df["timestamp"] = pd.to_datetime(
            epochs_df["timestamp"],
            format="%Y-%m-%d %H:%M:%S,%f",
            errors="coerce",
        )

    summary_df = summary_df.sort_values("start_time", na_position="last").reset_index(
        drop=True
    )
    if "run_label_slug" in summary_df.columns:
        summary_df["run_name"] = summary_df["run_label_slug"]
    else:
        summary_df["run_name"] = summary_df["run_id"]
    epochs_df = epochs_df.merge(
        summary_df[["run_id", "run_name", "run_variant"]], on="run_id", how="left"
    )
    iterations_df = iterations_df.merge(
        summary_df[["run_id", "run_name", "run_variant"]], on="run_id", how="left"
    )
    return epochs_df, iterations_df, summary_df


@app.cell
def _(pd):
    metric_specs = pd.DataFrame(
        [
            {
                "metric": "val_dice_mean",
                "label": "Validation Dice Mean",
                "higher_is_better": True,
            },
            {
                "metric": "test_dice_mean",
                "label": "Test Dice Mean",
                "higher_is_better": True,
            },
            {
                "metric": "test_cldice_mean",
                "label": "Test clDice Mean",
                "higher_is_better": True,
            },
            {
                "metric": "test_precision_mean",
                "label": "Test Precision Mean",
                "higher_is_better": True,
            },
            {
                "metric": "test_recall_mean",
                "label": "Test Recall Mean",
                "higher_is_better": True,
            },
            {
                "metric": "test_accuracy_mean",
                "label": "Test Accuracy Mean",
                "higher_is_better": True,
            },
            {
                "metric": "duration_hours",
                "label": "Runtime Hours",
                "higher_is_better": False,
            },
        ]
    )
    return (metric_specs,)


@app.cell
def _(metric_specs, pd, summary_df):
    run_summary_columns = [
        "run_name",
        "run_variant",
        "status",
        "job_id",
        "node",
        "unet_layers",
        "best_train_dice_mean",
        "best_train_dice_epoch",
        "final_epoch_loss",
        "val_dice_mean",
        "val_dice_std",
        "test_dice_mean",
        "test_dice_std",
        "test_cldice_mean",
        "test_precision_mean",
        "test_recall_mean",
        "test_accuracy_mean",
        "duration_hours",
        "last_epoch",
    ]
    run_summary_df = summary_df[run_summary_columns].copy()
    run_numeric_columns = run_summary_df.select_dtypes(include="number").columns
    run_summary_df[run_numeric_columns] = run_summary_df[run_numeric_columns].round(4)

    winner_rows = []
    for _spec in metric_specs.itertuples(index=False):
        _metric_name = _spec.metric
        _metric_subset = summary_df[["run_name", _metric_name]].dropna()
        if _metric_subset.empty:
            continue
        _ascending = not _spec.higher_is_better
        _best_entry = _metric_subset.sort_values(
            _metric_name, ascending=_ascending
        ).iloc[0]
        winner_rows.append(
            {
                "metric": _spec.label,
                "best_run": _best_entry["run_name"],
                "best_value": round(float(_best_entry[_metric_name]), 4),
                "average_across_runs": round(
                    float(_metric_subset[_metric_name].mean()), 4
                ),
                "std_across_runs": round(
                    float(_metric_subset[_metric_name].std(ddof=0)), 4
                ),
            }
        )

    metric_winners_df = pd.DataFrame(winner_rows)
    return metric_winners_df, run_summary_df


@app.cell
def _(mo, summary_df):
    overview_lines = [
        "## Run Overview",
        "",
        f"- Runs loaded: `{len(summary_df)}`",
        f"- Completed full evaluations: `{int((summary_df['status'] == 'full_eval').sum())}`",
    ]
    if summary_df["test_dice_mean"].notna().any():
        best_test_row = summary_df.sort_values("test_dice_mean", ascending=False).iloc[
            0
        ]
        overview_lines.append(
            f"- Best test Dice mean: `{best_test_row['run_name']}` with `{best_test_row['test_dice_mean']:.4f}`"
        )
    if summary_df["duration_hours"].notna().any():
        fastest_row = summary_df.sort_values("duration_hours", ascending=True).iloc[0]
        overview_lines.append(
            f"- Fastest run: `{fastest_row['run_name']}` with `{fastest_row['duration_hours']:.3f}` hours"
        )
    mo.md("\n".join(overview_lines))
    return


@app.cell
def _(metric_winners_df, mo, run_summary_df):
    summary_tabs = mo.ui.tabs(
        {
            "Per Run Summary": mo.ui.table(run_summary_df),
            "Metric Winners": mo.ui.table(metric_winners_df),
        }
    )
    summary_tabs
    return


@app.cell
def _(metric_specs, mo, plt, sns, summary_df):
    metric_chart_tabs = {}
    for _spec in metric_specs.itertuples(index=False):
        _metric_name = _spec.metric
        _metric_label = _spec.label
        _metric_subset = (
            summary_df[["run_name", "run_variant", _metric_name]].dropna().copy()
        )
        if _metric_subset.empty:
            continue

        _ascending = not _spec.higher_is_better
        _metric_subset = _metric_subset.sort_values(_metric_name, ascending=_ascending)
        _best_metric_row = _metric_subset.iloc[0]

        fig_metric, ax_metric = plt.subplots(
            figsize=(11, max(4, 0.8 * len(_metric_subset)))
        )
        sns.barplot(
            data=_metric_subset,
            x=_metric_name,
            y="run_name",
            hue="run_variant",
            dodge=False,
            ax=ax_metric,
        )
        ax_metric.set_title(_metric_label)
        ax_metric.set_xlabel(_metric_label)
        ax_metric.set_ylabel("run")
        ax_metric.legend(title="variant", loc="best")
        fig_metric.tight_layout()

        metric_chart_tabs[_metric_label] = mo.vstack(
            [
                mo.md(
                    f"**Best:** `{_best_metric_row['run_name']}` with `{float(_best_metric_row[_metric_name]):.4f}`"
                ),
                fig_metric,
            ]
        )

    metric_tabs = mo.ui.tabs(metric_chart_tabs)
    metric_tabs
    return


@app.cell
def _(plt, sns, summary_df):
    time_accuracy_df = summary_df.dropna(
        subset=["duration_hours", "test_dice_mean"]
    ).copy()
    if time_accuracy_df.empty:
        time_accuracy_chart = None
    else:
        fig_time, ax_time = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=time_accuracy_df,
            x="duration_hours",
            y="test_dice_mean",
            hue="run_variant",
            style="run_variant",
            s=140,
            ax=ax_time,
        )
        for _, row in time_accuracy_df.iterrows():
            ax_time.text(
                row["duration_hours"],
                row["test_dice_mean"],
                f" {row['run_name']}",
                va="center",
            )
        ax_time.set_title("Runtime vs Test Dice Mean")
        ax_time.set_xlabel("Runtime (hours)")
        ax_time.set_ylabel("Test Dice Mean")
        fig_time.tight_layout()
        time_accuracy_chart = fig_time
    time_accuracy_chart
    return


@app.cell
def _(mo):
    training_header = mo.md("## Training Comparison")
    training_header
    return


@app.cell
def _(iterations_df, pd):
    iterations_overlay_df = iterations_df.sort_values(
        ["run_name", "global_step"]
    ).copy()
    if not iterations_overlay_df.empty:
        iterations_overlay_df["rolling_loss_25"] = iterations_overlay_df.groupby(
            "run_name"
        )["loss"].transform(lambda series: series.rolling(25, min_periods=1).mean())
    else:
        iterations_overlay_df = pd.DataFrame(
            columns=["run_name", "global_step", "rolling_loss_25"]
        )
    return (iterations_overlay_df,)


@app.cell
def _(epochs_df, plt, sns):
    fig_epoch_loss, ax_epoch_loss = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=epochs_df,
        x="epoch",
        y="loss",
        hue="run_name",
        marker="o",
        ax=ax_epoch_loss,
    )
    ax_epoch_loss.set_title("Epoch Loss Across All Runs")
    ax_epoch_loss.set_xlabel("Epoch")
    ax_epoch_loss.set_ylabel("Epoch Loss")
    ax_epoch_loss.legend(title="run", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig_epoch_loss.tight_layout()
    fig_epoch_loss
    return


@app.cell
def _(epochs_df, plt, sns):
    fig_epoch_dice, ax_epoch_dice = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=epochs_df,
        x="epoch",
        y="dice_mean",
        hue="run_name",
        marker="o",
        ax=ax_epoch_dice,
    )
    ax_epoch_dice.set_title("Validation Dice Mean Across All Runs")
    ax_epoch_dice.set_xlabel("Epoch")
    ax_epoch_dice.set_ylabel("Dice Mean")
    ax_epoch_dice.legend(title="run", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig_epoch_dice.tight_layout()
    fig_epoch_dice
    return


@app.cell
def _(iterations_overlay_df, pd, plt, sns):
    import math

    sampled_df = iterations_overlay_df
    if not iterations_overlay_df.empty:
        max_points_per_run = 1000
        sampled_parts = []
        for _, run_df in iterations_overlay_df.groupby("run_name", sort=False):
            step = max(1, math.ceil(len(run_df) / max_points_per_run))
            sampled_parts.append(run_df.iloc[::step])
        sampled_df = pd.concat(sampled_parts, ignore_index=True)

    fig_iter_loss, ax_iter_loss = plt.subplots(figsize=(11, 6))
    if sampled_df.empty:
        ax_iter_loss.set_title("Rolling Iteration Loss Across All Runs")
        ax_iter_loss.text(
            0.5,
            0.5,
            "No iteration data available",
            ha="center",
            va="center",
            transform=ax_iter_loss.transAxes,
        )
        ax_iter_loss.set_xticks([])
        ax_iter_loss.set_yticks([])
    else:
        sns.lineplot(
            data=sampled_df,
            x="global_step",
            y="rolling_loss_25",
            hue="run_name",
            ax=ax_iter_loss,
            estimator=None,
            lw=1,
        )
        ax_iter_loss.legend(title="run", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax_iter_loss.set_title("Rolling Iteration Loss Across All Runs")
    ax_iter_loss.set_xlabel("Global Step")
    ax_iter_loss.set_ylabel("Rolling Loss (window=25)")
    fig_iter_loss.tight_layout()
    fig_iter_loss
    return


if __name__ == "__main__":
    app.run()
