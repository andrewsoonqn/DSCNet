import json
from pathlib import Path
import shutil
import subprocess
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION = REPO_ROOT / ".pi" / "extensions" / "experiment-control" / "index.ts"
COMMAND_TESTS = EXTENSION.with_name("commands.test.mjs")

class PiExperimentExtensionTests(unittest.TestCase):
    def test_exposes_exactly_the_six_controller_tools(self):
        source = EXTENSION.read_text()
        names = {
            line.split('"')[1]
            for line in source.splitlines()
            if line.strip().startswith('name: "experiment_')
        }
        self.assertEqual(
            names,
            {
                "experiment_verify",
                "experiment_submit",
                "experiment_status",
                "experiment_logs",
                "experiment_cancel",
                "experiment_fetch",
            },
        )
        self.assertNotIn("host: Type.", source)
        self.assertNotIn("command: Type.", source)

    def test_bounded_command_mapping(self):
        if shutil.which("node") is None:
            self.skipTest("Node.js is not installed")
        result = subprocess.run(
            ["node", "--test", str(COMMAND_TESTS)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_extension_loads_in_pi_rpc_mode(self):
        if shutil.which("pi") is None:
            self.skipTest("Pi is not installed")
        process = subprocess.Popen(
            [
                "pi",
                "--mode",
                "rpc",
                "--no-session",
                "--no-extensions",
                "--no-tools",
                "--extension",
                str(EXTENSION),
            ],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(
                json.dumps({"id": "load", "type": "get_state"}) + "\n",
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail("Pi RPC extension load timed out\n" + stdout + stderr)
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        response = next(
            (
                event
                for event in events
                if event.get("type") == "response" and event.get("id") == "load"
            ),
            None,
        )
        self.assertIsNotNone(response, stdout + stderr)
        self.assertTrue(response["success"], response)

if __name__ == "__main__":
    unittest.main()
