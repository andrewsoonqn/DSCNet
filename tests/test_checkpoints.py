import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

import S3_Train_Process
from S3_Checkpoint import (
    load_model_checkpoint,
    load_training_checkpoint,
    save_model_checkpoint,
    save_training_checkpoint,
)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_current_checkpoint_round_trip(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_checkpoint(source, path, pipeline="standard")
            load_model_checkpoint(target, path, pipeline="standard")

        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)

    def test_rejects_legacy_bare_state_dict(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
                load_model_checkpoint(model, path, pipeline="standard")

    def test_full_training_state_round_trip(self):
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
        source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1)
        target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=3)

        source_optimizer.zero_grad()
        source(torch.ones(1, 3)).sum().backward()
        source_optimizer.step()
        source_scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            save_training_checkpoint(
                source,
                path,
                "standard",
                optimizer=source_optimizer,
                scheduler=source_scheduler,
                scaler=None,
                epoch=7,
                best_score=0.81,
                config_digest="digest",
                loop_state={"early_stopping_counter": 3},
            )
            expected_random = (
                random.random(),
                np.random.random(),
                torch.rand(1).item(),
            )
            random.seed(99)
            np.random.seed(99)
            torch.manual_seed(99)
            state = load_training_checkpoint(
                target,
                path,
                "standard",
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                scaler=None,
                expected_config_digest="digest",
            )
            actual_random = (
                random.random(),
                np.random.random(),
                torch.rand(1).item(),
            )

        self.assertEqual(
            state,
            {
                "epoch": 7,
                "best_score": 0.81,
                "config_digest": "digest",
                "loop_state": {"early_stopping_counter": 3},
            },
        )
        self.assertEqual(actual_random, expected_random)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(target_scheduler.last_epoch, 1)
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)

    def test_standard_trainer_restores_loop_state_and_next_epoch(self):
        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
        source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, 1)
        target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_training_checkpoint(
                source,
                root / "latest.pt",
                "standard",
                optimizer=source_optimizer,
                scheduler=source_scheduler,
                scaler=None,
                epoch=4,
                best_score=0.7,
                config_digest="digest",
                loop_state={"dice_max": 0.72, "early_stopping_counter": 5},
            )
            state = S3_Train_Process._resume_training(
                target,
                SimpleNamespace(
                    Dir_Weights=str(root),
                    model_name="latest.pt",
                    config_digest="digest",
                ),
                target_optimizer,
                target_scheduler,
                None,
            )

        self.assertEqual(state, (5, 0.7, 0.72, 5))
        self.assertTrue(
            S3_Train_Process._early_stopping_reached(
                SimpleNamespace(use_earlystop=True, earlystop_patience=5), state[3]
            )
        )

    def test_resume_rejects_different_config(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            save_training_checkpoint(
                model,
                path,
                "standard",
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                epoch=1,
                best_score=0.0,
                config_digest="original",
            )
            with self.assertRaisesRegex(RuntimeError, "different experiment config"):
                load_training_checkpoint(
                    model,
                    path,
                    "standard",
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=None,
                    expected_config_digest="changed",
                )

    def test_rejects_checkpoint_from_other_pipeline(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimized.pt"
            save_model_checkpoint(model, path, pipeline="optimized")
            with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
                load_model_checkpoint(model, path, pipeline="standard")


if __name__ == "__main__":
    unittest.main()
