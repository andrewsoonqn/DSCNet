import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dscnet.experiment.config import PROJECT_ROOT, load_experiment_config


class PackageLayoutTests(unittest.TestCase):
    def test_project_root_and_hydra_config_do_not_depend_on_cwd(self):
        self.assertEqual(PROJECT_ROOT, REPO_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                config, _ = load_experiment_config(
                    "experiment/dscnet_standard",
                    ["action=prepare"],
                )
            finally:
                os.chdir(previous)
        self.assertEqual(config.action, "prepare")

    def test_module_entrypoint_runs_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            result = subprocess.run(
                [sys.executable, "-m", "dscnet", "--help"],
                cwd=directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resolved-config", result.stdout)

    def test_uv_is_the_only_dependency_authority(self):
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        self.assertNotIn("scripts", project["project"])
        pyproject_requirements = {
            Requirement(value).name.lower(): str(Requirement(value).specifier)
            for value in project["project"]["dependencies"]
        }
        self.assertTrue(pyproject_requirements)
        self.assertTrue((REPO_ROOT / "uv.lock").is_file())
        self.assertFalse((REPO_ROOT / "requirements.txt").exists())

    def test_flat_package_and_mirrored_tests_are_present(self):
        self.assertFalse((REPO_ROOT / "src").exists())
        self.assertTrue((REPO_ROOT / "dscnet" / "__init__.py").is_file())
        mirrored_modules = {
            "data/manifests.py": "data/test_manifests.py",
            "evaluation/metrics.py": "evaluation/test_metrics.py",
            "evaluation/sliding_window.py": "evaluation/test_sliding_window.py",
            "evaluation/summary.py": "evaluation/test_summary.py",
            "experiment/config.py": "experiment/test_config.py",
            "experiment/run.py": "experiment/test_run.py",
            "experiment/tracking.py": "experiment/test_tracking.py",
            "models/dsconv.py": "models/test_dsconv.py",
            "models/optimized_dsconv.py": "models/test_optimized_dsconv.py",
            "models/optimized.py": "models/test_optimized.py",
            "models/standard.py": "models/test_standard.py",
            "training/checkpoints.py": "training/test_checkpoints.py",
            "training/losses.py": "training/test_losses.py",
            "workflow.py": "test_workflow.py",
        }
        for source, test in mirrored_modules.items():
            with self.subTest(source=source):
                self.assertTrue((REPO_ROOT / "dscnet" / source).is_file())
                self.assertTrue((REPO_ROOT / "tests" / test).is_file())
        self.assertTrue((REPO_ROOT / "tests" / "tools" / "test_expctl.py").is_file())
        self.assertTrue(
            (REPO_ROOT / "tests" / "integration" / "test_package_layout.py").is_file()
        )
        self.assertFalse((REPO_ROOT / "DSCNet_3D_opensource").exists())
        self.assertTrue((REPO_ROOT / "scripts" / "run_local.sh").is_file())
        self.assertTrue((REPO_ROOT / "scripts" / "run_slurm.sh").is_file())


if __name__ == "__main__":
    unittest.main()
