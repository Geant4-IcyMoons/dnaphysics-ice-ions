#!/usr/bin/env python3
"""Compare unchanged screened forces with two nonrelativistic impact prescriptions.

Retains the existing phase-specific ice OOS. No production tables are written.
Uses total kinetic energy per ion; both curves use identical NR kinematics.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from physics import constants
from physics.inelastic_dielectric.k_shell import hydrogenic
from physics.inelastic_dielectric.projectile_potentials import projectile_form_factors
from physics.constants import (AASTEX_FULL_WIDTH_IN, PAPER_FONTSIZE, FONT_COURIER,
    RC_BASE_ELASTIC, rcparams_with_fontsize, AVOGADRO, H2O_MOLAR_MASS_G_MOL,
    PROJECTILE_LIBRARY, DEFAULT_WORKERS)
from physics.inelastic_dielectric.polarization import close_collisions as cc
from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import load_density, DATA_PATH

PHASES = ("amorphous", "hexagonal")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "plots/close_collisions"


def _node(task):
    state, energy, loss, velocity = task
    element, charge = state.split(":")
    try:
        result = cc.compare_kernels(loss, velocity, load_density(element, int(charge)))
        return {"state":state,"T_MeV":energy,"W_eV":loss,"velocity_au":velocity, **result}
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {"state":state,"T_MeV":energy,"W_eV":loss,"error":str(exc)}


def _spectrum(nodes, velocity, w, oos, key, stride=1):
    if key == "K_matched":
        return (_spectrum(nodes, velocity, w, oos, "K_close", stride)
                + _spectrum(nodes, velocity, w, oos, "K_distant", stride))
    source_w = np.array([r["W_eV"] for r in nodes])[::stride]
    k = np.array([r[key] for r in nodes])[::stride]
    if np.any(k <= 0):
        raise ValueError("Log interpolation requires strictly positive kernels.")
    interpolator = PchipInterpolator(np.log(source_w),np.log(k),extrapolate=False)
    evaluated = np.exp(interpolator(np.log(w)))
    if np.any(~np.isfinite(evaluated)):
        raise ValueError("Unresolved loss-grid domain.")
    return cc.equivalent_dcs(evaluated,velocity,oos)


def _moment(nodes, velocity, w, oos, key, stride=1):
    return float(np.trapezoid(w*_spectrum(nodes,velocity,w,oos,key,stride),w))


def _validity_fractions(nodes, velocity, w, oos):
    """Loss-weighted warning fractions, not estimates of physical accuracy.

    Count the entire matched moment at a flagged loss energy. This is not
    the fraction of individual collisions outside the small-impact domain.
    """
    source_w = np.array([r["W_eV"] for r in nodes])
    moment_density = w*_spectrum(nodes,velocity,w,oos,"K_matched")
    total = np.trapezoid(moment_density,w)
    result = {}
    for key, label in (
        ("shift_over_incident_electron_energy", "nonpositive_close_energy"),
        ("match_over_b90", "match_beyond_b90"),
    ):
        values = np.array([r[key] for r in nodes])
        interpolated = PchipInterpolator(np.log(source_w),values,extrapolate=False)(np.log(w))
        if not np.all(np.isfinite(interpolated)):
            raise ValueError("Unresolved validity-diagnostic loss grid.")
        result["matched_moment_fraction_with_"+label] = float(
            np.trapezoid(np.where(interpolated >= 1,moment_density,0),w)/total)
    return result


def _plot(report, output):
    font = None
    for candidate in (FONT_COURIER,"Courier New","Courier"):
        try:
            font_manager.findfont(candidate,fallback_to_default=False)
            font = candidate
            break
        except ValueError:
            continue
    if font is None:
        raise RuntimeError("Install a Courier-compatible font for this figure.")
    plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC,PAPER_FONTSIZE,{
        "font.family":font,"axes.grid":False,"pdf.fonttype":42,
        "mathtext.fontset":"custom","mathtext.rm":font,"mathtext.it":font,
        "mathtext.bf":font,"mathtext.fallback":None,
        "axes.labelpad":4,"axes.linewidth":.8,"axes.titlepad":5,
        "xtick.major.size":3,"ytick.major.size":3,
        "xtick.minor.size":1.7,"ytick.minor.size":1.7,
        "xtick.major.width":.7,"ytick.major.width":.7,
        "xtick.minor.width":.5,"ytick.minor.width":.5}))
    states = report["states"]
    colors = plt.get_cmap("plasma")(np.linspace(0,1,len(states)+1))[:-1]
    fig, axes = plt.subplots(2,2,figsize=(AASTEX_FULL_WIDTH_IN,4.8),sharex="col",sharey="row",
                            gridspec_kw={"height_ratios":[3,1.2],"hspace":.08})
    for col, phase in enumerate(PHASES):
        ax, residual = axes[:,col]
        for state,color in zip(states,colors):
            rows = [r for r in report["rows"] if r["phase"]==phase and r["state"]==state]
            lookup = {r["T_MeV"]: r for r in rows}
            t = report["energies_MeV"]
            values = lambda key: [lookup.get(e,{}).get(key,np.nan) for e in t]
            ax.loglog(t,values("S_cutoff_MeV_cm2_g"),ls="--",color=color,lw=1.2)
            ax.loglog(t,values("S_matched_MeV_cm2_g"),color=color,lw=1.2)
            residual.semilogx(t,100*np.array(values("relative_difference")),color=color,lw=1.2)
        ax.set_title(f"({'ab'[col]}) {phase.capitalize()} ice",loc="left")
        residual.set_title(f"({'cd'[col]})",loc="left",pad=2)
        residual.axhline(0,color=".5",lw=.7)
        residual.set_xlabel("Kinetic energy ($T$; MeV)")
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2,5),numticks=30))
        ax.yaxis.set_minor_formatter(NullFormatter())
        for panel in (ax,residual):
            panel.tick_params(which="both",direction="in",top=False,right=False)
            panel.grid(False,which="both")
    axes[0,0].set_ylabel("Polarization stopping\n(MeV cm$^2$ g$^{-1}$)")
    axes[1,0].set_ylabel("Change (%)")
    labels = []
    for state in states:
        element,charge = state.split(":")
        labels.append(element+"$^{"+(charge+"+" if int(charge)>0 else "0")+"}$")
    handles = [Line2D([],[],color=color,label=label,lw=1.2) for color,label in zip(colors,labels)]
    handles += [Line2D([],[],color=".2",ls="--",label="Salvat cutoff"),
                Line2D([],[],color=".2",ls="-",label="Close/distant matched")]
    fig.legend(handles=handles,ncol=4,loc="lower center",bbox_to_anchor=(.55,.025),
               frameon=False,borderpad=0,borderaxespad=0,labelspacing=.5)
    fig.subplots_adjust(left=.13,right=.985,bottom=.21,top=.92,wspace=.11)
    path=output/"close_collision_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states",nargs="+",default=["H:1","He:0","He:1","C:3","O:4","S:8"])
    parser.add_argument("--energies-MeV",nargs="+",type=float,default=[10,30,100])
    parser.add_argument("--loss-nodes",type=int,default=65)
    parser.add_argument("--workers",type=int,default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-pdf",type=Path)
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args()
    out=args.output_dir
    if args.plot_only:
        print(_plot(json.loads((out/"comparison.json").read_text()),out))
        return 0
    if args.loss_nodes<9 or args.loss_nodes%2!=1 or args.workers<1:
        parser.error("Require an odd --loss-nodes >= 9 and positive workers.")
    tasks=[]
    cases=[]
    for state in args.states:
        element,charge=state.split(":")
        density=load_density(element,int(charge))
        config=next(c for c in PROJECTILE_LIBRARY.values() if c["element"]==density.element)
        for energy in sorted(set(args.energies_MeV)):
            if not np.isfinite(energy) or energy<=0:
                parser.error("Energies must be finite and positive, in total MeV per ion.")
            velocity=float(np.sqrt(2*energy*1e6/(config["mass_au"]*cc.HARTREE_EV)))
            if velocity<=1:
                parser.error(f"{state} at {energy:g} MeV is not above the Bohr velocity.")
            wmax=2*cc.HARTREE_EV*velocity**2/(1+1/config["mass_au"])**2
            wmin=min(bd.model.epsilon_optical(p).Bmin for p in PHASES)
            losses=np.geomspace(wmin,wmax,args.loss_nodes)
            tasks.extend((state,energy,float(w),velocity) for w in losses)
            cases.append((state,energy,velocity,wmin,wmax))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
        nodes=list(tqdm(pool.map(_node,tasks),total=len(tasks),desc="Close-collision comparison"))
    failures=[r for r in nodes if "error" in r]
    rows=[]
    for state,energy,velocity,wmin,wmax in cases:
        group=[r for r in nodes if r["state"]==state and r["T_MeV"]==energy]
        if any("error" in r for r in group):
            continue
        for phase in PHASES:
            optical=bd.model.epsilon_optical(phase)
            w=np.geomspace(wmin,wmax,4097)
            edge=bd.model.OXYGEN_K_B_EV*np.array([1-1e-8,1,1+1e-8])
            w=np.unique(np.r_[w,edge[(edge>wmin)&(edge<wmax)]])
            oos=bd.oos_density(w,optical,phase)
            integrals={}
            coarse_integrals={}
            for key in ("K_cutoff","K_close","K_distant"):
                integrals[key]=_moment(group,velocity,w,oos.df_dW_total,key)
                coarse_integrals[key]=_moment(group,velocity,w,oos.df_dW_total,key,stride=2)
            for moments in (integrals,coarse_integrals):
                moments["K_matched"]=moments["K_close"]+moments["K_distant"]
            checks=[abs(coarse_integrals[key]/value-1) for key,value in integrals.items()]
            low_w=np.unique(np.r_[np.geomspace(wmin,wmax,2049),
                                  edge[(edge>wmin)&(edge<wmax)]])
            low_oos=bd.oos_density(low_w,optical,phase)
            grid_error=max(abs(_moment(group,velocity,low_w,low_oos.df_dW_total,key)/integrals[key]-1)
                           for key in ("K_cutoff","K_matched"))
            conversion=1e-2*AVOGADRO/H2O_MOLAR_MASS_G_MOL
            rows.append(dict(state=state,phase=phase,T_MeV=energy,velocity_au=velocity,
                Wmax_eV=wmax,S_cutoff_eV_m2=integrals["K_cutoff"],
                S_matched_eV_m2=integrals["K_matched"],
                S_cutoff_MeV_cm2_g=conversion*integrals["K_cutoff"],
                S_matched_MeV_cm2_g=conversion*integrals["K_matched"],
                relative_difference=integrals["K_matched"]/integrals["K_cutoff"]-1,
                close_fraction=integrals["K_close"]/integrals["K_matched"],
                max_kernel_interpolation_change=max(checks),W_grid_change=grid_error,
                max_quadrature_change=max(max(r["relative_refinement"].values()) for r in group),
                max_match_over_b90=max(r["match_over_b90"] for r in group),
                max_shift_over_incident_electron_energy=max(r["shift_over_incident_electron_energy"] for r in group),
                positive_effective_close_energy=all(r["positive_effective_close_energy"] for r in group),
                valence_oos_integral=oos.valence_integral_raw*oos.valence_norm,
                core_oos_integral=oos.ok_integral_raw*oos.ok_norm))
            rows[-1].update(_validity_fractions(group,velocity,w,oos.df_dW_total))
    converged=not failures and all(r["max_kernel_interpolation_change"]<.01 and r["W_grid_change"]<.005 for r in rows)
    sources=[Path(__file__),Path(cc.__file__),Path(cc.screened.__file__),Path(bd.__file__),
             Path(bd.model.__file__),Path(hydrogenic.__file__),Path(constants.__file__),
             Path(projectile_form_factors.__file__),DATA_PATH]
    if args.reference_pdf:
        sources.append(args.reference_pdf)
    report=dict(model=cc.MODEL,states=args.states,energies_MeV=sorted(set(args.energies_MeV)),
        energy_convention="total kinetic energy per ion",kinematics="common nonrelativistic limit; not the production relativistic prescription",
        comparison="Section-4-inspired induced-potential matching vs Salvat impact cutoff, same frozen force and ice OOS",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        workers=args.workers,loss_nodes=args.loss_nodes,numerical_status="PASS" if converged else "FAIL",
        physical_accuracy_validated=False,production_generator_changed=False,
        validity_warning="Small-impact asymptote may be matched beyond its domain; inspect b_match/b90 and shifted-energy diagnostics. No fitted data or physical accuracy claim.",
        validity_fractions="Fraction of the matched stopping moment at flagged loss energies, not a collision fraction or an error estimate.",
        units="eV m^2 per H2O; multiply by 1e-2*N_A/M_H2O for MeV cm^2/g",
        rows=rows,nodes=nodes,failures=failures)
    out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if rows:
        with (out/"comparison.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(_plot(report,out))
    print(report["numerical_status"],f"{len(rows)} rows; {len(failures)} unresolved loss nodes")
    return 0 if converged else 1


if __name__=="__main__":
    raise SystemExit(main())
