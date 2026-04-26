#!/usr/bin/env python3
"""Plot the W-value text output for example_4."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_txt = script_dir / "outputs" / "example_4_wvalue.txt"
    default_out = script_dir / "outputs" / "example_4_wvalue.png"

    parser = argparse.ArgumentParser(description="Plot W-value output for example_4")
    parser.add_argument("--txt", default=str(default_txt), help="Path to example_4 W-value text file")
    parser.add_argument("--out", default=str(default_out), help="Output PNG path")
    args = parser.parse_args()

    txt_path = Path(args.txt).expanduser()
    if not txt_path.exists():
        raise SystemExit(f"W-value text file not found: {txt_path}")

    data = np.loadtxt(txt_path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 5:
        raise SystemExit("Expected at least 5 columns: E0[eV], <Nion>, rms(Nion), W[eV], W_err[eV]")

    order = np.argsort(data[:, 0])
    energy_ev = data[order, 0]
    mean_nion = data[order, 1]
    rms_nion = data[order, 2]
    w_ev = data[order, 3]
    w_err_ev = data[order, 4]

    fig, (ax_w, ax_n) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_w.errorbar(energy_ev, w_ev, yerr=w_err_ev, fmt="o-", capsize=3, lw=1.5)
    ax_w.set_xscale("log")
    ax_w.set_ylabel("W value (eV)")
    ax_w.set_title("Example 4: Amorphous ice W-value scan")
    ax_w.grid(True, which="both", alpha=0.3)

    ax_n.errorbar(energy_ev, mean_nion, yerr=rms_nion, fmt="o-", capsize=3, lw=1.5)
    ax_n.set_xscale("log")
    ax_n.set_xlabel("Primary energy (eV)")
    ax_n.set_ylabel(r"$\langle N_{\mathrm{ion}} \rangle$")
    ax_n.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
