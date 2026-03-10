import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split paired raw/seg NIfTI files into train/val/test folders for DSCNet."
    )
    parser.add_argument(
        "--dataset-dir",
        default="Data/MiniVess_Half",
        help="Dataset directory containing raw/ and seg/.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of paired cases to place in train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of paired cases to place in val.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of paired cases to place in test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before splitting.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How files are materialized into train/val/test folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing train/ val/ test/ folders before writing the split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without creating files.",
    )
    return parser.parse_args()


def strip_extension(name):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Unsupported file extension: {name}")


def collect_cases(dataset_dir):
    raw_dir = dataset_dir / "raw"
    seg_dir = dataset_dir / "seg"

    if not raw_dir.is_dir() or not seg_dir.is_dir():
        raise FileNotFoundError(f"Expected both {raw_dir} and {seg_dir} to exist.")

    raw_files = {}
    for path in sorted(raw_dir.glob("*.nii*")):
        raw_files[strip_extension(path.name)] = path

    seg_files = {}
    for path in sorted(seg_dir.glob("*.nii*")):
        key = strip_extension(path.name)
        if not key.endswith("_y"):
            raise ValueError(
                f"Expected segmentation filename to end with '_y': {path.name}"
            )
        seg_files[key[:-2]] = path

    raw_ids = set(raw_files)
    seg_ids = set(seg_files)
    missing_seg = sorted(raw_ids - seg_ids)
    missing_raw = sorted(seg_ids - raw_ids)
    if missing_seg or missing_raw:
        lines = []
        if missing_seg:
            lines.append(f"Missing segmentations for: {', '.join(missing_seg)}")
        if missing_raw:
            lines.append(f"Missing raw files for: {', '.join(missing_raw)}")
        raise ValueError("\n".join(lines))

    case_ids = sorted(raw_ids)
    return [(case_id, raw_files[case_id], seg_files[case_id]) for case_id in case_ids]


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("train/val/test ratios must sum to 1.0")


def split_cases(cases, train_ratio, val_ratio, seed):
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_cases = shuffled[:n_train]
    val_cases = shuffled[n_train : n_train + n_val]
    test_cases = shuffled[n_train + n_val :]

    return {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
        "counts": {"train": len(train_cases), "val": len(val_cases), "test": n_test},
    }


def clear_split_dirs(dataset_dir):
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def ensure_split_dirs(dataset_dir):
    for split in ("train", "val", "test"):
        (dataset_dir / split / "image").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "label").mkdir(parents=True, exist_ok=True)


def materialize(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    os.symlink(src.resolve(), dst)


def write_split(dataset_dir, split_map, mode):
    manifest_rows = []

    for split in ("train", "val", "test"):
        image_dir = dataset_dir / split / "image"
        label_dir = dataset_dir / split / "label"
        for index, (case_id, raw_path, seg_path) in enumerate(
            split_map[split], start=1
        ):
            filename = f"{index}.nii.gz"
            image_dst = image_dir / filename
            label_dst = label_dir / filename
            materialize(raw_path, image_dst, mode)
            materialize(seg_path, label_dst, mode)
            manifest_rows.append(
                {
                    "split": split,
                    "index": index,
                    "case_id": case_id,
                    "image_source": str(raw_path),
                    "label_source": str(seg_path),
                    "image_output": str(image_dst),
                    "label_output": str(label_dst),
                }
            )

    manifest_path = dataset_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "index",
                "case_id",
                "image_source",
                "label_source",
                "image_output",
                "label_output",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    cases = collect_cases(dataset_dir)
    split_map = split_cases(cases, args.train_ratio, args.val_ratio, args.seed)

    print(f"Found {len(cases)} paired cases in {dataset_dir}")
    print(
        "Planned split:",
        ", ".join(
            f"{split}={split_map['counts'][split]}"
            for split in ("train", "val", "test")
        ),
    )

    if args.dry_run:
        for split in ("train", "val", "test"):
            case_ids = ", ".join(case_id for case_id, _, _ in split_map[split])
            print(f"{split}: {case_ids}")
        return

    if args.overwrite:
        clear_split_dirs(dataset_dir)

    ensure_split_dirs(dataset_dir)
    manifest_path = write_split(dataset_dir, split_map, args.mode)

    print(f"Wrote split folders under {dataset_dir}")
    print(f"Saved manifest to {manifest_path}")
    print(
        "Images and labels are renamed to 1.nii.gz, 2.nii.gz, ... per split for DSCNet compatibility."
    )


if __name__ == "__main__":
    main()
