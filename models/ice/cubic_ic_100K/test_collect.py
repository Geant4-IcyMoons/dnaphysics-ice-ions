"""Check publication wiring in isolation using explicitly synthetic reports."""
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from models.ice.cubic_ic_100K import validate as audit


class CollectionTests(unittest.TestCase):
    def test_three_rows_registry_and_legacy_link(self):
        original = audit.HERE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models/ice/cubic_ic_100K"
            runs = model / "runs"
            (model / "validation").mkdir(parents=True)
            for phase, filename in [
                ("hexagonal_ih_100K_experimental", "seed1000_final.xyz.gz"),
                ("epsr_lda80k/artifacts", "lda80k_epsr.xyz.gz"),
            ]:
                destination = model.parent / phase / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original.parent / phase / filename, destination)
            shutil.copyfile(original.parent / "registry.json", model.parent / "registry.json")
            legacy = root / "old_comparison.png"
            legacy.write_bytes(b"synthetic previous plot")
            audit.write_json(runs / "campaign.json", {"previous_comparison": {
                "path": legacy.name, "sha256": audit.file_sha256(legacy)}})
            for seed in audit.CONFIG["seeds"]:
                report = model / "validation" / f"seed{seed}.json"
                audit.write_json(report, {"passed": True, "config": audit.CONFIG, "signature": "synthetic"})
                audit.write_json(runs / f"seed{seed}" / "preparation.json", {
                    "signature": "synthetic", "bernal_fowler": {"orientation_sha256": str(seed)}})
                snapshot = model / f"seed{seed}_final.xyz.gz"
                initial = original / "runs" / f"seed{seed}" / "initial.xyz"
                if not initial.exists():
                    self.skipTest("Local generation artifacts are required for the figure integration check")
                snapshot.write_bytes(gzip.compress(initial.read_bytes(), mtime=0))
                audit.write_json(snapshot.with_suffix(".json"), {
                    "sha256": audit.file_sha256(snapshot), "validation_sha256": audit.file_sha256(report)})
            with patch.object(audit, "HERE", model), patch.object(audit, "RUNS", runs):
                audit.collect()
                self.assertTrue(legacy.is_symlink())
                rows = json.loads((model.parent / "ice_structure_comparison.json").read_text())["rows"]
                self.assertEqual([row["label"] for row in rows], [
                    "Hexagonal ice Ih, 100 K", "Cubic ice Ic, 100 K", "Amorphous LDA, 80 K"])
                registry = json.loads((model.parent / "registry.json").read_text())
                self.assertFalse(registry["models"]["cubic_ic_100k"]["collision_ready"])
                self.assertEqual(registry["models"]["cubic_ic_100k"]["configuration_count"], 3)
                # Tampered accepted products must not be published on a rerun.
                (model / "seed1000_final.xyz.gz").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "snapshot or report changed"):
                    audit.collect()


if __name__ == "__main__":
    unittest.main()
