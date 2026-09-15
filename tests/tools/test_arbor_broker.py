import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import arbor_broker
import arbor_preflight


class CandidateBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git(self.repo, "init", "-b", "main")
        self._git(self.repo, "config", "user.email", "tests@example.com")
        self._git(self.repo, "config", "user.name", "Tests")
        files = {
            ".gitignore": "/.arbor/\n",
            "dscnet/models/model.py": "VALUE = 1\n",
            "dscnet/training/losses.py": "LOSS = 1\n",
            "tools/arbor_eval.py": "print('trusted')\n",
            "tests/models/test_model.py": "pass\n",
        }
        for relative, content in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self._git(self.repo, "add", *files)
        self._git(self.repo, "commit", "-m", "baseline")
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.config = arbor_broker.BrokerConfig.resolved(
            self.repo, self.data_dir, sys.executable
        )
        self.worktrees: list[Path] = []

    def tearDown(self):
        for worktree in self.worktrees:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=self.repo,
                check=False,
                capture_output=True,
            )
        self.temporary.cleanup()

    def _git(self, cwd, *arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _candidate(self, node_id="1", *, changed_path=None, dirty=False):
        branch = f"arbor/dscnet-pilot/n{node_id}"
        worktree = self.root / f"worktree-{node_id}"
        self._git(self.repo, "worktree", "add", "-b", branch, str(worktree))
        self.worktrees.append(worktree)
        if changed_path:
            path = worktree / changed_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.read_text() + "CHANGED = True\n" if path.exists() else "CHANGED = True\n")
            if not dirty:
                self._git(worktree, "add", changed_path)
                self._git(worktree, "commit", "-m", "candidate")
        tree_path = (
            self.repo
            / ".arbor"
            / "sessions"
            / "pilot"
            / ".coordinator"
            / "idea_tree.json"
        )
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree = {
            "version": 3,
            "nodes": {
                "ROOT": {"id": "ROOT", "status": "done"},
                node_id: {
                    "id": node_id,
                    "status": "running",
                    "code_ref": branch,
                },
            },
        }
        tree_path.write_text(json.dumps(tree))
        return worktree

    def test_submit_launches_fixed_worker_for_allowed_committed_candidate(self):
        self._candidate(changed_path="dscnet/models/model.py")
        process = SimpleNamespace(pid=os.getpid())
        launch = Mock(return_value=process)
        with patch("arbor_broker._process_identity", return_value="start"):
            result = arbor_broker.CandidateBroker(
                self.config, launcher=launch
            ).submit_candidate("pilot", "1")

        self.assertEqual(
            result,
            {"schema_version": 1, "node_id": "1", "status": "running"},
        )
        command = launch.call_args.args[0]
        self.assertEqual(command[2], "_worker")
        self.assertIn(str(self.data_dir.resolve()), command)
        self.assertNotIn("audit", command)
        self.assertNotIn("test", result)
        self.assertNotIn("branch", result)
        self.assertNotIn("worktree", result)

    def test_repeated_submission_returns_existing_state_without_relaunch(self):
        self._candidate(changed_path="dscnet/models/model.py")
        process = SimpleNamespace(pid=os.getpid())
        launch = Mock(return_value=process)
        broker = arbor_broker.CandidateBroker(self.config, launcher=launch)
        with patch("arbor_broker._process_identity", return_value="start"):
            first = broker.submit_candidate("pilot", "1")
            second = broker.submit_candidate("pilot", "1")
        self.assertEqual(first, second)
        self.assertEqual(launch.call_count, 1)

    def test_protected_path_change_is_rejected_before_launch(self):
        self._candidate(changed_path="tools/arbor_eval.py")
        launch = Mock()
        with self.assertRaisesRegex(
            arbor_broker.BrokerError, "outside the allowed edit surface"
        ):
            arbor_broker.CandidateBroker(
                self.config, launcher=launch
            ).submit_candidate("pilot", "1")
        launch.assert_not_called()
        saved = json.loads(
            (self.repo / ".arbor/sessions/pilot/.broker/submissions/1.json").read_text()
        )
        self.assertEqual(saved["status"], "rejected")

    def test_symlink_candidate_is_rejected_and_recorded(self):
        worktree = self._candidate()
        path = worktree / "dscnet/models/link.py"
        path.symlink_to("model.py")
        self._git(worktree, "add", "dscnet/models/link.py")
        self._git(worktree, "commit", "-m", "symlink")
        with self.assertRaisesRegex(arbor_broker.BrokerError, "unsafe Git mode"):
            arbor_broker.CandidateBroker(self.config).submit_candidate("pilot", "1")
        saved = json.loads(
            (self.repo / ".arbor/sessions/pilot/.broker/submissions/1.json").read_text()
        )
        self.assertEqual(saved["status"], "rejected")

    def test_trusted_checkout_must_be_main_at_main_head(self):
        self._git(self.repo, "checkout", "--detach")
        with self.assertRaisesRegex(arbor_broker.BrokerError, "on main"):
            arbor_broker.BrokerConfig.resolved(
                self.repo, self.data_dir, sys.executable
            )

    def test_second_eligible_submission_is_durably_rejected_while_active(self):
        first_worktree = self._candidate(changed_path="dscnet/models/model.py")
        tree_path = self.repo / ".arbor/sessions/pilot/.coordinator/idea_tree.json"
        tree = json.loads(tree_path.read_text())
        branch = "arbor/dscnet-pilot/n2"
        second = self.root / "worktree-2"
        self._git(self.repo, "worktree", "add", "-b", branch, str(second))
        self.worktrees.append(second)
        tree["nodes"]["2"] = {"id": "2", "status": "running", "code_ref": branch}
        tree_path.write_text(json.dumps(tree))
        process = SimpleNamespace(pid=os.getpid())
        with patch("arbor_broker._process_identity", return_value="start"):
            arbor_broker.CandidateBroker(self.config, launcher=Mock(return_value=process)).submit_candidate("pilot", "1")
            with self.assertRaisesRegex(arbor_broker.BrokerError, "active"):
                arbor_broker.CandidateBroker(self.config).submit_candidate("pilot", "2")
        saved = json.loads(
            (self.repo / ".arbor/sessions/pilot/.broker/submissions/2.json").read_text()
        )
        self.assertEqual(saved["status"], "rejected")

    def test_dirty_candidate_is_rejected_before_launch(self):
        self._candidate(changed_path="dscnet/models/model.py", dirty=True)
        launch = Mock()
        with self.assertRaisesRegex(arbor_broker.BrokerError, "clean and committed"):
            arbor_broker.CandidateBroker(
                self.config, launcher=launch
            ).submit_candidate("pilot", "1")
        launch.assert_not_called()

    def test_oversized_candidate_blob_is_rejected_before_launch(self):
        self._candidate(changed_path="dscnet/models/model.py")
        with patch("arbor_broker.MAX_CANDIDATE_BLOB_BYTES", 1):
            with self.assertRaisesRegex(arbor_broker.BrokerError, "size limit"):
                arbor_broker.CandidateBroker(self.config).submit_candidate("pilot", "1")

    def test_stale_reservation_is_reconciled_on_repeat(self):
        state = self.repo / ".arbor/sessions/pilot/.broker/submissions/1.json"
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"schema_version": 1, "node_id": "1", "status": "reserved"}))
        result = arbor_broker.CandidateBroker(self.config).submit_candidate("pilot", "1")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "worker")

    def test_model_config_surface_accepts_only_yaml(self):
        self.assertTrue(arbor_broker._is_allowed("configs/model/candidate.yaml"))
        self.assertFalse(arbor_broker._is_allowed("configs/model/candidate.py"))

    def test_invalid_identifiers_are_rejected(self):
        broker = arbor_broker.CandidateBroker(self.config)
        with self.assertRaisesRegex(arbor_broker.BrokerError, "invalid run name"):
            broker.submit_candidate("../pilot", "1")


class CandidateWorkerTests(unittest.TestCase):
    def test_evaluator_score_must_be_in_unit_interval(self):
        payload = json.dumps(
            {
                "schema_version": 1, "status": "completed",
                "controller_run_id": "a" * 16, "metric": "validation.dice",
                "score": 1.01,
            }
        )
        with self.assertRaisesRegex(arbor_broker.BrokerError, r"\[0, 1\]"):
            arbor_broker._parse_evaluation_result(payload)

    def test_bounded_run_keeps_only_bounded_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "output.log"
            result = arbor_broker._bounded_run(
                [sys.executable, "-c", "print('x' * 1100000)"],
                cwd=Path(directory), env={"PATH": os.environ.get("PATH", "")},
                log_path=log, timeout=10,
            )
        self.assertEqual(result.returncode, 0)
        self.assertLessEqual(len(result.stdout.encode()), arbor_broker.MAX_OUTPUT_BYTES)
    def test_worker_rejects_commit_mutation_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "1.json"
            for path in (root / "worktree", root / "data", root / "repo"):
                path.mkdir()
            state.write_text(json.dumps({"node_id": "1", "status": "running"}))
            options = SimpleNamespace(
                state=str(state), lock=str(root / ".lock"), worktree=str(root / "worktree"),
                data_dir=str(root / "data"), project_python=sys.executable,
                repo=str(root / "repo"), candidate_commit="expected",
            )
            with patch("arbor_broker._git", return_value="mutated"), patch(
                "arbor_broker._bounded_run"
            ) as run:
                self.assertEqual(arbor_broker._run_worker(options), 1)
            run.assert_not_called()
            self.assertIn("changed before evaluation", json.loads(state.read_text())["error"])

    def test_candidate_environment_excludes_credentials_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SSH_AUTH_SOCK": "secret", "AWS_SECRET_ACCESS_KEY": "secret"}
        ):
            environment = arbor_broker._candidate_env(Path(directory))
        self.assertNotIn("SSH_AUTH_SOCK", environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertFalse(any("data" in value.lower() for value in environment.values()))

    def test_bounded_run_kills_process_group_on_timeout(self):
        process = Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 1), None]
        with tempfile.TemporaryDirectory() as directory, patch(
            "arbor_broker.subprocess.Popen", return_value=process
        ), patch("arbor_broker.os.killpg") as kill:
            with self.assertRaisesRegex(arbor_broker.BrokerError, "timeout"):
                arbor_broker._bounded_run(
                    ["cmd"], cwd=Path(directory), env={},
                    log_path=Path(directory) / "log", timeout=1,
                )
        kill.assert_called_once_with(123, 15)
    def test_worker_returns_only_validated_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "1.json"
            lock = root / ".lock"
            worktree = root / "worktree"
            data = root / "data"
            repo = root / "repo"
            for path in (worktree / "tools", data, repo):
                path.mkdir(parents=True, exist_ok=True)
            (worktree / "tools" / "arbor_eval.py").write_text("pass\n")
            state.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_name": "pilot",
                        "node_id": "1",
                        "status": "running",
                    }
                )
            )
            responses = [
                subprocess.CompletedProcess([], 0, stdout="tests pass", stderr=""),
                subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=json.dumps(
                        {
                            "schema_version": 1,
                            "status": "completed",
                            "controller_run_id": "a" * 16,
                            "metric": "validation.dice",
                            "score": 0.86,
                        }
                    ),
                    stderr="",
                ),
            ]
            options = SimpleNamespace(
                state=str(state),
                lock=str(lock),
                worktree=str(worktree),
                data_dir=str(data),
                project_python=sys.executable,
                repo=str(repo),
                candidate_commit="candidate",
            )
            with patch("arbor_broker._git", side_effect=["candidate", ""]), patch(
                "arbor_broker._bounded_run", side_effect=responses
            ) as run:
                code = arbor_broker._run_worker(options)

            self.assertEqual(code, 0)
            saved = json.loads(state.read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["controller_run_id"], "a" * 16)
            self.assertEqual(saved["score"], 0.86)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0][0], str(Path(sys.executable).absolute()))
            self.assertNotIn("test", arbor_broker._public_state(saved))

    def test_worker_stops_before_evaluation_when_preflight_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "1.json"
            worktree = root / "worktree"
            data = root / "data"
            repo = root / "repo"
            for path in (worktree, data, repo):
                path.mkdir()
            state.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_name": "pilot",
                        "node_id": "1",
                        "status": "running",
                    }
                )
            )
            options = SimpleNamespace(
                state=str(state),
                lock=str(root / ".lock"),
                worktree=str(worktree),
                data_dir=str(data),
                project_python=sys.executable,
                repo=str(repo),
                candidate_commit="candidate",
            )
            failed = subprocess.CompletedProcess([], 1, stdout="", stderr="broken")
            with patch("arbor_broker._git", side_effect=["candidate", ""]), patch(
                "arbor_broker._bounded_run", return_value=failed
            ) as run:
                code = arbor_broker._run_worker(options)

            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 1)
            saved = json.loads(state.read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["failure_stage"], "preflight")
            self.assertEqual(saved["error"], "candidate syntax preflight failed")


class CandidatePreflightTests(unittest.TestCase):
    def test_syntax_check_is_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "broken.py").write_text("def broken(:\n")
            with patch.object(arbor_preflight, "PYTHON_ROOTS", (root,)):
                with self.assertRaises(SyntaxError):
                    arbor_preflight._compile_candidate_python()


if __name__ == "__main__":
    unittest.main()
