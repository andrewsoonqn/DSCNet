"""Generate deterministic text manifests for NIfTI datasets."""

import re
from pathlib import Path


def _natural_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


def Get_file_list(file_dir):
    files = sorted(
        (
            path
            for path in Path(file_dir).iterdir()
            if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
        ),
        key=_natural_key,
    )
    return files, len(files)


def _write_manifest(files, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(path) for path in files))
    print("2 Finish Generate_Txt:", output_path)


def Generate_Txt(image_path, txt_name):
    files, _ = Get_file_list(image_path)
    _write_manifest(files, txt_name)


def Generate_Paired_Txt(image_path, label_path, image_txt, label_txt):
    """Write aligned image-label manifests or fail before training starts."""
    images, _ = Get_file_list(image_path)
    labels, _ = Get_file_list(label_path)
    if not images or not labels:
        raise ValueError(
            f"image-label split is empty: {image_path}, {label_path}"
        )

    image_names = {path.name for path in images}
    label_names = {path.name for path in labels}
    if image_names != label_names:
        missing_labels = sorted(image_names - label_names)
        missing_images = sorted(label_names - image_names)
        raise ValueError(
            "image-label filenames do not match; "
            f"missing labels={missing_labels}, missing images={missing_images}"
        )

    labels_by_name = {path.name: path for path in labels}
    paired_labels = [labels_by_name[path.name] for path in images]
    _write_manifest(images, image_txt)
    _write_manifest(paired_labels, label_txt)
