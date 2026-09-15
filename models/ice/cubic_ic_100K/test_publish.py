"""Test isolated publication against a disposable local bare remote."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from models.ice.cubic_ic_100K import publish as publication


class PublicationTests(unittest.TestCase):
    def test_push_excludes_unrelated_work_and_rejects_remote_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, workspace = root / "remote.git", root / "workspace"
            def git(*args, cwd=workspace):
                return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
            git("init", "--bare", str(remote), cwd=root)
            git("init", "--initial-branch=ion_modular", str(workspace), cwd=root)
            git("config", "user.name", "Publication test")
            git("config", "user.email", "test@example.invalid")
            (workspace / "material").write_text("old")
            (workspace / "unrelated").write_text("old")
            git("add", "material", "unrelated")
            git("commit", "-m", "baseline")
            git("remote", "add", "origin", str(remote))
            git("push", "origin", "ion_modular")
            baseline = git("rev-parse", "HEAD")
            request = root / "request.json"
            request.write_text(json.dumps({"remote_url": str(remote), "base_entries": {
                "material": git("ls-tree", baseline, "--", "material")}}))
            (workspace / "unrelated").write_text("user's staged work")
            git("add", "unrelated")
            index_before = (workspace / ".git/index").read_bytes()
            with patch.object(publication, "ROOT", workspace), patch.object(publication, "REQUEST", request), \
                    patch.object(publication, "verify", return_value={"material": b"new"}):
                result = publication.publish()
                self.assertEqual(git("rev-parse", "HEAD"), baseline)
                self.assertEqual((workspace / ".git/index").read_bytes(), index_before)
                self.assertEqual(git("show", result["commit"] + ":material"), "new")
                self.assertEqual(git("show", result["commit"] + ":unrelated"), "old")
                self.assertEqual(git("ls-remote", "origin", "refs/heads/ion_modular").split()[0], result["commit"])
                with self.assertRaisesRegex(ValueError, "Remote task file changed"):
                    publication.publish()


if __name__ == "__main__":
    unittest.main()
