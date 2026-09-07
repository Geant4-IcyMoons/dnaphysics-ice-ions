"""Standalone imports and pre-move numerical equivalence, not physical validation."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "physics/inelastic_dielectric"


def test_reference_kernels_unchanged():
    from physics.inelastic_dielectric import generate_cross_sections as gen
    reference = json.loads((ROOT/"tests/data/modular_kernel_reference.json").read_text())
    values = []
    old_phase = gen.ICE_TYPE
    try:
        for phase in reference["phases"]:
            gen.ICE_TYPE = phase
            s = gen.model.epsilon_optical(phase)
            c = gen.model.default_dispersion_coefficients()
            for element, z in gen.projectile_ff.ELEMENTS.items():
                for charge in range(z+1):
                    gen.set_projectile(element)
                    gen._set_projectile_charge_state(charge)
                    for rel in (False, True):
                        gen._set_projectile_relativistic_dcs(rel)
                        t = gen.PROJECTILE_MASS_NUMBER*reference["T_eV_per_u"]
                        values.extend([
                            gen._selected_dsigma_excitation(15,t,0,s,c,reference["Nq"]),
                            gen._selected_dsigma_ionization(50,t,0,s,c,reference["Nq"]),
                            gen._selected_dsigma_kshell(1000,t,s,c,reference["Nq"]),
                        ])
        np.testing.assert_allclose(values, reference["values"], rtol=2e-13, atol=0)
    finally:
        gen.ICE_TYPE = old_phase
        gen.set_projectile("proton")
        gen._set_projectile_relativistic_dcs(False)


def test_no_legacy_imports():
    for path in (ROOT/"physics").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("python_scripts"), path
                assert node.module != "constants", path
            elif isinstance(node, ast.Import):
                assert all(not item.name.startswith("python_scripts") for item in node.names), path


def test_standalone_entrypoint_help(tmp_path):
    result = subprocess.run([sys.executable, str(PACKAGE/"generate_cross_sections.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    for flag in ("--include-barkas-dcs", "--relativistic-projectile-dcs", "--output-dir"):
        assert flag in result.stdout


def test_modules_import_without_old_tree():
    modules = ["pwba.kernels", "rpwba.kernels", "k_shell.hydrogenic",
               "projectile_potentials.generate_projectile_atomic_data",
               "polarization.benchmarking.benchmark_barkas_sbethe",
               "polarization.benchmarking.compare_screened_barkas_point_projectiles",
               "projectile_potentials.benchmarking.benchmark_projectile_form_factors"]
    for name in modules:
        importlib.import_module("physics.inelastic_dielectric."+name)


def test_component_outputs_are_ignored():
    paths = ["polarization/benchmarking/plots/nonlinear_oscillator/a.pdf",
             "projectile_potentials/benchmarking/runs/a.json", "output/tables/a.dat"]
    result = subprocess.run(["git", "check-ignore", *[str(PACKAGE/p) for p in paths]],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert len(result.stdout.splitlines()) == len(paths)
