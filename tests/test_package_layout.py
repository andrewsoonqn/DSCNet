import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

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
            environment["PYTHONPATH"] = str(SRC_DIR)
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

    def test_legacy_source_tree_is_gone(self):
        self.assertFalse((REPO_ROOT / "DSCNet_3D_opensource").exists())
        self.assertTrue((REPO_ROOT / "scripts" / "run_local.sh").is_file())
        self.assertTrue((REPO_ROOT / "scripts" / "run_slurm.sh").is_file())


if __name__ == "__main__":
    unittest.main()
