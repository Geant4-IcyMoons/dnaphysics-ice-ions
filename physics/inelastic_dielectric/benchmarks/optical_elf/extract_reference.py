"""Extract the ice optical curves from the vector paths of Matias+25 Fig. 1."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pdfplumber

SOURCE_URL = "https://research.science.eus/documentos/6900abd7ee6cd0098b614df1/f/6900abd7ee6cd0098b614df2.pdf"
HERE = Path(__file__).resolve().parent


def extract(pdf):
    with pdfplumber.open(pdf) as document:
        page = document.pages[2]
        output = {}
        for phase, color in (("amorphous", (0., 0., 1.)),
                             ("hexagonal", (1., .50196, 0.))):
            main, tail = [], []
            for line in page.lines:
                if line["stroking_color"] != color or line["height"] < .001:
                    continue  # Exclude horizontal arrows and legend strokes.
                # The PDF retains off-frame paths: linewidth identifies the
                # inset independently of location; then enforce its clip box.
                inset = abs(line["linewidth"] - .980) < .002
                main_curve = abs(line["linewidth"] - 1.336) < .002
                for x, y in line["pts"]:
                    if inset and 217.899 <= x <= 285.484 and 62.339 <= y <= 129.752:
                        tail.append((100 * 10**((x - 217.899) / (285.483 - 217.899)
                                               * np.log10(320)),
                                     10**(-2 - 8 * (y - 62.340) / (129.751 - 62.340))))
                    elif main_curve and 81.385 <= x <= 289.765 and 51.033 <= y <= 204.458:
                        main.append((10 * 10**((x - 119.231) / (289.764 - 119.231)),
                                     .8 * (204.458 - y) / (204.458 - 51.033)))
            for name, rows in (("optical", main), ("tail", tail)):
                values = np.array(sorted(set(rows)))
                if len(values) < 25 or np.any(values <= 0):
                    raise RuntimeError(f"Unexpected {phase} {name} paths")
                path = HERE / "references" / f"matias2025_{phase}_{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, values, delimiter=",", header="energy_eV,ELF", fmt="%.10g")
                output[f"{phase}_{name}"] = {"points": len(values),
                    "energy_range_eV": values[[0, -1], 0].tolist()}
    provenance = {"doi": "10.1103/ksdx-mnd7", "source_url": SOURCE_URL,
        "pdf_sha256": hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
        "figure": "1, PDF page 3", "reference_kind": "MELF-GOS fits, not raw measurements",
        "method": "Colored vector dash endpoints; main/inset identified by linewidth 1.336/0.980 PDF points, with each clip box enforced. Linear ELF/log energy in main panel, log/log in inset. Horizontal annotations excluded. No normalization imposed.",
        "main_calibration_pdf_points": {"10_eV_x": 119.231, "100_eV_x": 289.764,
            "ELF_0_y": 204.458, "ELF_0.8_y": 51.033},
        "inset_calibration_pdf_points": {"100_eV_x": 217.899, "32000_eV_x": 285.483,
            "ELF_0.01_y": 62.340, "ELF_1e-10_y": 129.751},
        "density_g_cm3": .94,
        "limits": "Finite-resolution figure extraction, not author-supplied numerical data. Gaps between dashes are interpolated. No extrapolation below/above the extracted domain; reference I is a truncated figure estimate only.",
        "curves": output}
    (HERE / "references/provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    extract(parser.parse_args().pdf)
