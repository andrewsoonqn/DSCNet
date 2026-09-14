import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from dscnet.evaluation import validation_comparisons as vc


class ValidationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ['images', 'labels', 'predictions', 'weights']:
            (self.root / directory).mkdir()
        self.args = SimpleNamespace(save_path=str(self.root / 'predictions'),
            Dir_Weights=str(self.root / 'weights'), model_name='latest', model_name_max='best',
            Image_Va_txt=str(self.root / 'images.txt'), Label_Va_txt=str(self.root / 'labels.txt'))
        self.mask = np.zeros((3, 5, 5), dtype=np.uint16); self.mask[:, 2, 2] = 1
        for directory in ['images', 'labels', 'predictions']:
            self.save(directory, self.mask)
        for directory in ['images', 'labels']:
            (self.root / (directory + '.txt')).write_text(str(self.root / directory / '1.nii.gz') + '\n')
        self.checkpoint(1, best=True); vc.initialize(self.args)

    def save(self, directory, values, origin=None):
        image = sitk.GetImageFromArray(values)
        if origin is not None:
            image.SetOrigin(origin)
        sitk.WriteImage(image, str(self.root / directory / '1.nii.gz'))

    def checkpoint(self, epoch, best=False):
        (self.root / 'weights/latest').write_text(str(epoch))
        if best:
            (self.root / 'weights/best').write_text(str(epoch))

    def status(self):
        return json.loads((vc.root_for(self.args) / 'status.json').read_text())

    def test_integer_overlay_pixels(self):
        rgb = vc.error_rgb(np.array([[0, 1], [0, 1]], dtype='uint16'),
                           np.array([[0, 0], [1, 1]], dtype='uint16'))
        np.testing.assert_array_equal(rgb, [[[0, 0, 0], [255, 0, 0]], [[0, 128, 255], [192, 192, 192]]])

    def test_best_final_and_real_slice_metrics(self):
        vc.record_validation(self.args, 1, True)
        self.checkpoint(2)
        prediction = self.mask.copy(); prediction[0] = 0
        self.save('predictions', prediction)
        vc.record_validation(self.args, 2, False)
        vc.finalize_validation_comparisons(self.args)
        root = vc.root_for(self.args)
        self.assertEqual(self.status()['state'], 'completed')
        with (root / 'comparison/slice-metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        final = [r for r in rows if r['role'] == 'final']
        self.assertLess(float(final[0]['dice']), float(final[1]['dice']))
        self.assertTrue((root / 'comparison/figures/1-slices.png').is_file())
        self.assertEqual(json.loads((root / 'comparison/figure-settings.json').read_text())['epochs'], {'best': 1, 'final': 2})

    def test_geometry_rejected_even_when_shape_matches(self):
        self.save('predictions', self.mask, (1, 0, 0))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            vc.record_validation(self.args, 1, True)
        self.assertFalse((vc.root_for(self.args) / 'retained.json').exists())

    def test_binary_and_finite_rejected(self):
        values = self.mask.astype(float); values[0, 0, 0] = 2
        self.save('predictions', values)
        with self.assertRaisesRegex(ValueError, 'nonfinite or nonbinary'):
            vc.record_validation(self.args, 1, True)
        # NIfTI readers may sanitize NaN; exercise the returned-array boundary.
        values[0, 0, 0] = np.nan
        with patch.object(vc.sitk, 'GetArrayFromImage', return_value=values):
            with self.assertRaisesRegex(ValueError, 'nonfinite or nonbinary'):
                vc.volume(self.root / 'predictions/1.nii.gz', binary=True)

    def test_second_volume_missing_does_not_publish_partial_snapshot(self):
        import shutil
        for folder in ['images', 'labels']:
            shutil.copyfile(self.root / folder / '1.nii.gz', self.root / folder / '2.nii.gz')
            with (self.root / (folder + '.txt')).open('a') as stream:
                stream.write(str(self.root / folder / '2.nii.gz') + '\n')
        with self.assertRaises(RuntimeError):
            vc.record_validation(self.args, 1, True)
        root = vc.root_for(self.args)
        self.assertFalse((root / 'retained.json').exists())
        self.assertFalse((root / 'epoch-1').exists())
        self.assertFalse(list(root.glob('.stage-*')))

    def test_missing_prediction_rejected(self):
        (self.root / 'predictions/1.nii.gz').unlink()
        with self.assertRaises(RuntimeError):
            vc.record_validation(self.args, 1, True)

    def test_later_checkpoint_not_mislabelled_final(self):
        vc.record_validation(self.args, 1, True)
        self.checkpoint(2)
        vc.finalize_validation_comparisons(self.args)
        self.assertEqual(self.status()['state'], 'unavailable')

    def test_resume_does_not_reuse_stale_best(self):
        vc.record_validation(self.args, 1, True)
        vc.initialize(self.args)
        self.checkpoint(2)
        vc.record_validation(self.args, 2, False)
        vc.finalize_validation_comparisons(self.args)
        self.assertEqual(self.status()['state'], 'unavailable')

    def test_new_best_replaces_old_snapshot(self):
        vc.record_validation(self.args, 1, True)
        self.checkpoint(2, best=True)
        vc.record_validation(self.args, 2, True)
        self.assertFalse((vc.root_for(self.args) / 'epoch-1').exists())
        vc.finalize_validation_comparisons(self.args)
        self.assertEqual(self.status()['state'], 'completed')

    def test_tampered_retained_mask_rejected(self):
        vc.record_validation(self.args, 1, True)
        (vc.root_for(self.args) / 'epoch-1/masks/1.nii.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            vc.finalize_validation_comparisons(self.args)

    def test_render_failure_does_not_publish_partial_set(self):
        from matplotlib.figure import Figure
        vc.record_validation(self.args, 1, True)
        original = Figure.savefig
        def fail_after_write(figure, *args, **kwargs):
            original(figure, *args, **kwargs)
            raise RuntimeError('interrupted rendering')
        with patch.object(Figure, 'savefig', fail_after_write):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                vc.finalize_validation_comparisons(self.args)
        root = vc.root_for(self.args)
        self.assertFalse((root / 'comparison').exists())
        self.assertFalse(list(root.glob('.render-*')))
        self.assertTrue((root / 'epoch-1/masks/1.nii.gz').is_file())
        vc.finalize_validation_comparisons(self.args)
        self.assertEqual(self.status()['state'], 'completed')

    def test_publication_contains_completion_even_if_root_status_write_fails(self):
        vc.record_validation(self.args, 1, True)
        original = vc.atomic_json
        def fail_root_status(path, value):
            if Path(path) == vc.root_for(self.args) / 'status.json':
                raise RuntimeError('status interruption')
            return original(path, value)
        with patch.object(vc, 'atomic_json', fail_root_status):
            with self.assertRaisesRegex(RuntimeError, 'status interruption'):
                vc.finalize_validation_comparisons(self.args)
        publication = vc.root_for(self.args) / 'comparison'
        self.assertEqual(json.loads((publication / 'status.json').read_text())['state'], 'completed')
        self.assertTrue((publication / 'figures/1-slices.png').is_file())
        self.assertTrue((publication / 'slice-metrics.csv').is_file())

    def test_complete_rendering_settings(self):
        vc.record_validation(self.args, 1, True)
        vc.finalize_validation_comparisons(self.args)
        settings = json.loads((vc.root_for(self.args) / 'comparison/figure-settings.json').read_text())
        self.assertEqual(settings['rendering']['contrast_percentiles'], [1, 99])
        self.assertEqual(settings['rendering']['dpi'], 130)
        self.assertEqual(settings['rendering']['slice_axis'], 0)
        self.assertEqual(settings['volumes'][0]['figure_size_inches'], [14, 10.8])
        self.assertEqual(set(settings['rendering']['versions']), {'matplotlib', 'numpy', 'pillow', 'SimpleITK'})

    def test_spacing_direction_and_image_geometry_rejected(self):
        for directory, attribute, value in [('predictions', 'SetSpacing', (2, 1, 1)),
                ('predictions', 'SetDirection', (-1, 0, 0, 0, -1, 0, 0, 0, 1)),
                ('images', 'SetOrigin', (1, 0, 0))]:
            with self.subTest(directory=directory, attribute=attribute):
                for folder in ['predictions', 'images']:
                    self.save(folder, self.mask)
                path = self.root / directory / '1.nii.gz'
                image = sitk.ReadImage(str(path)); getattr(image, attribute)(value)
                sitk.WriteImage(image, str(path))
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    vc.record_validation(self.args, 1, True)

    def test_snapshot_pointer_failure_preserves_previous_best(self):
        vc.record_validation(self.args, 1, True)
        self.checkpoint(2, best=True)
        original = vc.atomic_json
        def fail_pointer(path, value):
            if Path(path).name == 'retained.json':
                raise RuntimeError('pointer interruption')
            return original(path, value)
        with patch.object(vc, 'atomic_json', fail_pointer):
            with self.assertRaisesRegex(RuntimeError, 'pointer interruption'):
                vc.record_validation(self.args, 2, True)
        root = vc.root_for(self.args)
        self.assertEqual(json.loads((root / 'retained.json').read_text())['best'], 'epoch-1')
        self.assertTrue((root / 'epoch-1/masks/1.nii.gz').is_file())
        vc.finalize_validation_comparisons(self.args)
        self.assertEqual(self.status()['state'], 'unavailable')

    def test_real_torch_checkpoint_serializations(self):
        import torch
        payload = {'epoch': 70, 'state_dict': {'weight': torch.ones(2)}}
        latest, best = self.root / 'weights/latest', self.root / 'weights/best'
        torch.save(payload, latest); torch.save(payload, best)
        self.assertNotEqual(vc.digest(latest), vc.digest(best))
        vc.record_validation(self.args, 70, True)
        vc.finalize_validation_comparisons(self.args)
        evidence = json.loads((vc.root_for(self.args) / 'epoch-70/evidence.json').read_text())
        self.assertEqual(evidence['epoch'], torch.load(best, weights_only=True)['epoch'])
        self.assertEqual(evidence['best_checkpoint_sha256'], vc.digest(best))
        self.assertEqual(self.status()['state'], 'completed')

    def test_real_mlflow_preserves_entire_tree_in_original_run(self):
        from mlflow import MlflowClient
        from dscnet.experiment.run import _log_declared_artifacts
        vc.record_validation(self.args, 1, True)
        self.checkpoint(2)
        self.save('predictions', np.zeros_like(self.mask))
        vc.record_validation(self.args, 2, False)
        vc.finalize_validation_comparisons(self.args)
        self.args.Dir_Log = str(self.root / 'logs')
        Path(self.args.Dir_Log).mkdir()
        client = MlflowClient(tracking_uri='sqlite:///' + str(self.root / 'mlflow.db'))
        experiment = client.create_experiment('validation', artifact_location=(self.root / 'mlartifacts').as_uri())
        run = client.create_run(experiment)
        run_id = run.info.run_id
        recorder = SimpleNamespace(log_artifact=lambda path, destination: client.log_artifact(run_id, str(path), destination))
        hidden = vc.root_for(self.args) / '.render-abandoned/partial.png'
        hidden.parent.mkdir(); hidden.write_bytes(b'partial')
        _log_declared_artifacts(recorder, self.args, 'train')
        download = self.root / 'download'; download.mkdir()
        location = Path(client.download_artifacts(run_id, 'validation-comparisons', str(download)))
        expected = {p.relative_to(vc.root_for(self.args)).as_posix(): p.read_bytes()
            for p in vc.root_for(self.args).rglob('*') if p.is_file()
            and not any(part.startswith('.') for part in p.relative_to(vc.root_for(self.args)).parts)}
        actual = {p.relative_to(location).as_posix(): p.read_bytes() for p in location.rglob('*') if p.is_file()}
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual['epoch-1/masks/1.nii.gz'], actual['epoch-2/masks/1.nii.gz'])
        self.assertEqual([r.info.run_id for r in client.search_runs([experiment])], [run_id])

    def test_no_validation_explicit(self):
        vc.finalize_validation_comparisons(self.args)
        self.assertEqual(self.status()['state'], 'no_validation')
