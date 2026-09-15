"""Exercise interruption, committed-block reuse, and corrupt-resume rejection."""
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from models.ice.cubic_ic_100K import run


class RestartTests(unittest.TestCase):
    def test_interruption_preserves_completed_block_and_rejects_corruption(self):
        real_sleep = time.sleep
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            length = run.cell_length()
            lattice = f"{length} 0 0 0 {length} 0 0 0 {length}"
            (root / "initial.xyz").write_text(
                f'3\nLattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
                'O 1 1 1\nH 2 1 1\nH 1 2 1\n')
            run.write_json(root / "preparation.json", {"signature": "test", "molecules": 1})
            executable = root / "fake_gpumd"
            executable.write_text(f'#!{sys.executable}\n' + '''
from pathlib import Path
import sys
root = Path.cwd().parent
with (root / "executions").open("a") as handle: handle.write(Path.cwd().name + "\\n")
if Path.cwd().name == "block001" and (root / "fail_once").exists():
    (root / "fail_once").unlink()
    Path("thermo.out").write_text("partial")
    sys.exit(1)
lines = Path("model.xyz").read_text().splitlines()
if "vel:R:3" not in lines[1]:
    lines[1] = lines[1].replace("pos:R:3", "pos:R:3:vel:R:3")
    lines[2:] = [line + " 0 0 0" for line in lines[2:]]
Path("dump.xyz").write_text("\\n".join(lines) + "\\n")
Path("thermo.out").write_text(("100 " + "0 " * 11 + "\\n") * 2)
''')
            executable.chmod(0o755)
            (root / "fail_once").touch()
            settings = {"block_steps": 2, "thermo_interval_steps": 1,
                        "equilibration_blocks": 1, "sampling_blocks": 1}
            with patch.object(run, "prepare", return_value=root), patch.dict(run.CONFIG, settings), \
                    patch.object(run.time, "sleep", side_effect=lambda _: real_sleep(0.01)):
                with self.assertRaisesRegex(RuntimeError, "GPUMD failed"):
                    run.simulate(1000, executable)
                first = (root / "block000/completed.json").read_bytes()
                self.assertFalse((root / "block001/completed.json").exists())
                run.simulate(1000, executable)
                self.assertEqual(first, (root / "block000/completed.json").read_bytes())
                run.simulate(1000, executable)
                self.assertEqual((root / "executions").read_text().splitlines(),
                                 ["block000", "block001", "block001"])
                source_sha = run.file_sha256(root / "initial.xyz")
                with self.assertRaisesRegex(ValueError, "Incompatible block lineage"):
                    run.completed_block(root / "block000", "different", source_sha)
                (root / "block000/thermo.out").write_text("corrupt")
                with self.assertRaisesRegex(ValueError, "Corrupted checkpoint"):
                    run.simulate(1000, executable)


if __name__ == "__main__":
    unittest.main()
