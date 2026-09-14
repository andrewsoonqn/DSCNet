"""Checkpoint-bound evidence from existing validation predictions; never inference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

from dscnet.evaluation.metrics import dice_score, cldice_score


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def root_for(args):
    return Path(args.save_path) / '_validation_comparisons'


def initialize(args):
    """New process never silently inherits unverified comparison state on resume."""
    root = root_for(args)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    atomic_json(root / 'status.json', {'state': 'no_validation',
        'reason': 'No checkpoint-bound validation retained in this execution; historical best unavailable on resume.'})


def manifest(path):
    paths = [Path(line) for line in Path(path).read_text().splitlines() if line.strip()]
    names = [p.name for p in paths]
    if not paths or len(names) != len(set(names)) or any(not n.endswith('.nii.gz') for n in names):
        raise ValueError('validation manifest must have unique NIfTI filenames')
    return dict(zip(names, paths))


def volume(path, reference=None, binary=False):
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError('validation volume must be three-dimensional')
    if reference is not None:
        if image.GetSize() != reference.GetSize() or any(
            not np.allclose(getattr(image, key)(), getattr(reference, key)(), rtol=0, atol=1e-6)
            for key in ('GetSpacing', 'GetOrigin', 'GetDirection')
        ):
            raise ValueError(f'validation geometry mismatch: {path}')
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all() or (binary and not np.isin(array, [0, 1]).all()):
        raise ValueError(f'nonfinite or nonbinary validation data: {path}')
    return image, array.astype(bool) if binary else array


def scores(pred, target):
    tp = int((pred & target).sum()); fp = int((pred & ~target).sum()); fn = int((~pred & target).sum())
    return {'dice': float(dice_score(pred, target)), 'cldice': float(cldice_score(pred, target)),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 1.0,
            'recall': tp / (tp + fn) if tp + fn else 1.0}


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def record_validation(args, epoch, is_best):
    """Called only AFTER authoritative checkpoint saves."""
    root = root_for(args)
    labels, images = manifest(args.Label_Va_txt), manifest(args.Image_Va_txt)
    if labels.keys() != images.keys():
        raise ValueError('validation image/label manifests differ')
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.stage-', dir=root))
    try:
        rows, files = [], {}
        (stage / 'masks').mkdir()
        for name, label in labels.items():
            ref, target = volume(label, binary=True)
            volume(images[name], ref)
            source = Path(args.save_path) / name
            _, prediction = volume(source, ref, binary=True)
            destination = stage / 'masks' / name
            shutil.copyfile(source, destination)
            rows.append({'volume': name, 'epoch': int(epoch), **scores(prediction, target)})
            files[name] = {'prediction_sha256': digest(destination), 'label_sha256': digest(label),
                'image_sha256': digest(images[name]), 'size': list(ref.GetSize()),
                'spacing': list(ref.GetSpacing()), 'origin': list(ref.GetOrigin()), 'direction': list(ref.GetDirection())}
        checkpoint = Path(args.Dir_Weights) / args.model_name
        evidence = {'epoch': int(epoch), 'checkpoint_sha256': digest(checkpoint),
            'experiment_digest': os.environ.get('DSCNET_EXPERIMENT_DIGEST'), 'files': files}
        if is_best and digest(Path(args.Dir_Weights) / args.model_name_max) != evidence['checkpoint_sha256']:
            # Torch serialization can differ between two saves of identical state.
            evidence['best_checkpoint_sha256'] = digest(Path(args.Dir_Weights) / args.model_name_max)
        csv_write(stage / 'per-volume.csv', rows)
        atomic_json(stage / 'evidence.json', evidence)
        snapshot = root / f'epoch-{epoch}'
        if snapshot.exists():
            raise ValueError('duplicate validation epoch in execution')
        os.replace(stage, snapshot)
        pointer = json.loads((root / 'retained.json').read_text()) if (root / 'retained.json').exists() else {}
        pointer['latest_validation'] = snapshot.name
        if is_best:
            pointer['best'] = snapshot.name
        atomic_json(root / 'retained.json', pointer)
        atomic_json(root / 'status.json', {'state': 'retained', 'epoch': int(epoch)})
        for previous in root.glob('epoch-*'):
            if previous.name not in pointer.values():
                shutil.rmtree(previous)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def error_rgb(prediction, target):
    prediction = np.asarray(prediction, dtype=bool); target = np.asarray(target, dtype=bool)
    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rgb[prediction & target] = (192, 192, 192)
    rgb[prediction & ~target] = (255, 0, 0)
    rgb[~prediction & target] = (0, 128, 255)
    return rgb


def finalize_validation_comparisons(args):
    root = root_for(args)
    pointer = root / 'retained.json'
    if not pointer.exists():
        return
    retained = json.loads(pointer.read_text())
    if 'best' not in retained:
        atomic_json(root / 'status.json', {'state': 'unavailable', 'reason': 'Historical best masks not available in this execution.'})
        return
    roles = {'best': retained['best'], 'final': retained['latest_validation']}
    metadata = {role: json.loads((root / folder / 'evidence.json').read_text()) for role, folder in roles.items()}
    for role, checkpoint_name in [('best', args.model_name_max), ('final', args.model_name)]:
        expected = metadata[role].get('best_checkpoint_sha256', metadata[role]['checkpoint_sha256']) if role == 'best' else metadata[role]['checkpoint_sha256']
        if digest(Path(args.Dir_Weights) / checkpoint_name) != expected:
            atomic_json(root / 'status.json', {'state': 'unavailable', 'reason': f'{role} checkpoint has no matching retained validation; no additional inference performed.'})
            return
    labels, images = manifest(args.Label_Va_txt), manifest(args.Image_Va_txt)
    if labels.keys() != images.keys() or any(set(m['files']) != set(labels) for m in metadata.values()):
        raise ValueError('retained validation manifest differs')
    publication = root / 'comparison'
    if publication.exists():
        raise ValueError('comparison already published; initialize a new execution before regenerating')
    # All rendered outputs, including their completion record, appear together.
    # Hidden staging trees are excluded from MLflow and recovery manifests.
    with tempfile.TemporaryDirectory(prefix='.render-', dir=root) as temporary:
        stage = Path(temporary)
        _render_comparisons(root, stage, roles, metadata, labels, images)
        os.replace(stage, publication)
    atomic_json(root / 'status.json', {'state': 'completed', 'roles': roles,
        'publication': 'comparison'})


def _render_comparisons(root, stage, roles, metadata, labels, images):
    import matplotlib
    import PIL
    from matplotlib.figure import Figure
    output = stage / 'figures'; output.mkdir()
    settings, slice_rows = [], []
    for name, label in labels.items():
        ref, target = volume(label, binary=True)
        _, image = volume(images[name], ref)
        masks = {}
        for role, folder in roles.items():
            entry = metadata[role]['files'][name]
            path = root / folder / 'masks' / name
            if digest(path) != entry['prediction_sha256'] or digest(label) != entry['label_sha256'] or digest(images[name]) != entry['image_sha256']:
                raise ValueError('retained validation input changed')
            _, masks[role] = volume(path, ref, binary=True)
        lo, hi = np.percentile(image, [1, 99]); hi = max(float(hi), float(lo) + 1e-6)
        indices = np.unique(np.linspace(0, len(image) - 1, 5, dtype=int))
        fig = Figure(figsize=(14, 3.6 * len(indices)))
        axes = fig.subplots(len(indices), 4, squeeze=False)
        for row, z in enumerate(indices):
            axes[row, 0].imshow(image[z], cmap='gray', vmin=lo, vmax=hi, origin='upper', interpolation='nearest')
            axes[row, 1].imshow(target[z], cmap='gray', vmin=0, vmax=1, origin='upper', interpolation='nearest')
            titles = [f'z={z} original', f'z={z} ground truth']
            for col, role in enumerate(('best', 'final'), 2):
                axes[row, col].imshow(error_rgb(masks[role][z], target[z]), origin='upper', interpolation='nearest')
                titles.append(f'z={z} {role}: epoch {metadata[role]["epoch"]}')
            for ax, title in zip(axes[row], titles):
                ax.set_title(title); ax.axis('off')
        for role in roles:
            for z in range(len(image)):
                slice_rows.append({'volume': name, 'role': role, 'epoch': metadata[role]['epoch'], 'z': z, **scores(masks[role][z], target[z])})
        fig.suptitle(f'{name}: grey correct vessel | red extra foreground | blue missed vessel')
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(output / f'{name[:-7]}-slices.png', dpi=130)
        settings.append({'volume': name, 'z_indices': indices.tolist(), 'contrast': [float(lo), hi],
            'figure_size_inches': [14, 3.6 * len(indices)]})
    csv_write(stage / 'slice-metrics.csv', slice_rows)
    atomic_json(stage / 'figure-settings.json', {'roles': roles, 'epochs': {r: m['epoch'] for r, m in metadata.items()},
        'volumes': settings, 'metric_definitions': 'dice_score/cldice_score from recorded run source; per-volume 3D, per-slice 2D; TP/FP/FN voxel counts',
        'legend': {'TP': 'grey', 'FP': 'red', 'FN': 'blue', 'TN': 'black'},
        'rendering': {'contrast_percentiles': [1, 99], 'minimum_contrast_span': 1e-6,
            'dpi': 130, 'image_colormap': 'gray', 'label_colormap': 'gray',
            'label_limits': [0, 1], 'overlay_rgb': {'TP': [192, 192, 192],
                'FP': [255, 0, 0], 'FN': [0, 128, 255], 'TN': [0, 0, 0]},
            'array_order': 'zyx', 'slice_axis': 0, 'display_origin': 'upper', 'interpolation': 'nearest',
            'orientation': 'native image index plane; no anatomical reorientation',
            'slice_selection': 'unique linspace(0, depth-1, 5, dtype=int)',
            'versions': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
                'pillow': PIL.__version__, 'SimpleITK': sitk.Version_VersionString()}},
        'source_sha256': digest(__file__)})
    atomic_json(stage / 'status.json', {'state': 'completed', 'roles': roles})
