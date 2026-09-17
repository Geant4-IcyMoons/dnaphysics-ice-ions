"""Plot measured H+--H2O trajectories and projectile-region counting results."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.patches import Circle

from physics.constants import FONT_COURIER, RC_BASE_STANDARD


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, help='Export plot data from a completed case; otherwise use local CSVs')
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    if args.case:
        case = args.case.resolve()
        capture = case/'clearing_0024000/capture.json'
        coordinates = case/'collision/td.general/coordinates'
        result = case/'result.json'
        analysis = json.loads(capture.read_text())
        receipt = json.loads(result.read_text())
        frames = analysis['frames']
        data = np.array([[f['clearing_time_au'], *f['probabilities_by_electron_count']] for f in frames])
        np.savetxt(output/'capture.csv', data, delimiter=',', header='clearing_time_au,'+','.join(f'P{i}' for i in range(9)), comments='')
        trajectory = np.loadtxt(coordinates, usecols=range(14))
        # Retain the complete measured trajectory; this small table is sufficient to replot.
        np.savetxt(output/'trajectory.csv', trajectory, delimiter=',', header='step,time_au,'+','.join(f'{atom}_{axis}_bohr' for atom in ['O','H1','H2','projectile'] for axis in 'xyz'), comments='', fmt='%.10g')
        metadata = {'source_files': {str(p): digest(p) for p in [capture, coordinates, result]},
                    'energy_ev': receipt['energy_ev_total'], 'impact_bohr': receipt['settings']['impact'],
                    'handoff_time_au': receipt['handoff']['time_au'], 'orientation': receipt['orientation'],
                    'stationarity_tolerance': analysis['stationarity_tolerance'],
                    'stationarity_frames': analysis['stationarity_frames'],
                    'status': 'Stationarity passes; numerical convergence and literature comparison outstanding.'}
        (output/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    metadata = json.loads((output/'provenance.json').read_text())
    data = np.loadtxt(output/'capture.csv', delimiter=',', skiprows=1)
    trajectory = np.loadtxt(output/'trajectory.csv', delimiter=',', skiprows=1)
    times, probabilities = data[:,0], data[:,1:]
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    n = metadata['stationarity_frames']
    changes = np.array([np.ptp(probabilities[i-n+1:i+1], axis=0).max() for i in range(n-1,len(times))])
    positions = trajectory[:,2:14].reshape(-1,4,3)
    for font in Path('/usr/share/fonts/urw-base35').glob('NimbusMonoPS-*.otf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(RC_BASE_STANDARD)
    plt.rcParams.update({'font.family': FONT_COURIER, 'font.size': 10, 'axes.titlepad': 10})
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    colors = ['#C43C39','#167B9B','#669C38']
    ax=axes[0,0]
    for j, (label, color) in enumerate(zip(['O','H (water)','H (water)'],[colors[0],colors[1],colors[1]])):
        ax.plot(positions[:,j,0],positions[:,j,1],color=color,alpha=.8,label=label if j<2 else None)
        ax.scatter(*positions[0,j,:2],s=65 if j==0 else 28,color=color,zorder=4)
        ax.scatter(*positions[-1,j,:2],s=40,marker='x',color=color,zorder=4)
    ax.plot(positions[:,-1,0],positions[:,-1,1],color=colors[2],label='Projectile nucleus')
    ax.scatter(*positions[0,-1,:2],color=colors[2],s=28)
    ax.scatter(*positions[-1,-1,:2],color=colors[2],marker='x',s=40)
    ax.set(xlabel='Laboratory x (bohr)',ylabel='Laboratory y (bohr)',title='(a) Measured nuclear trajectories')
    ax.legend(fontsize=8,loc='lower right')
    ax.text(.02,.97,'Dots: start; crosses: handoff\nxy projection',transform=ax.transAxes,va='top',fontsize=8)
    ax.set_ylim(min(-4,positions[:,:,1].min()-1), max(7,positions[:,:,1].max()+1))
    ax=axes[0,1]
    for i, color in enumerate(colors):
        ax.plot(times, probabilities[:,i],'.-',color=color,label=f'P({i})')
    ax.plot(times,probabilities[:,3:].sum(axis=1),'.-',color='#825EA0',label='P(n > 2)')
    ax.set(xlabel='Clearing time (a.u.)',ylabel='Probability',title='(b) Electrons in projectile sphere',ylim=(0,.7))
    ax.legend(ncol=2,fontsize=9)
    ax=axes[1,0]
    ax.plot(times,probabilities[:,1],'o-',color=colors[1],markeredgecolor=colors[1],markersize=4)
    ax.set(xlabel='Clearing time (a.u.)',ylabel='P(1)',title='(c) Single-electron probability')
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.text(.97,.94,f'Final P(1) = {probabilities[-1,1]:.6f}',transform=ax.transAxes,ha='right',va='top')
    ax=axes[1,1]
    ax.semilogy(times[n-1:],changes,'o-',color='#825EA0',markeredgecolor='#825EA0',markersize=4)
    tol=metadata['stationarity_tolerance']
    ax.axhline(tol,color='0.3',ls='--',label=f'Tolerance = {tol:g}')
    ax.set(xlabel='Clearing time (a.u.)',ylabel='Maximum probability range',title=f'(d) Stationarity: last {n} samples')
    ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.grid(False)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'H+ + H2O | {metadata["energy_ev"]/1000:g} keV | b = {metadata["impact_bohr"]:g} bohr | Octopus 16.4',fontsize=15)
    fig.text(.5,.025,'Independent molecular calculation. Stationarity passes; numerical and literature validation remain outstanding.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95),h_pad=2,w_pad=2)
    fig.savefig(output/'benchmark.pdf')
    plt.close(fig)
    molecular_scene(output, trajectory, positions, metadata)
    print(f'Final stationarity range: {changes[-1]:.12g}; tolerance: {tol:g}')


def molecular_scene(output, trajectory, positions, metadata):
    """Equal-scale xy snapshots from the nuclear trajectory; no synthetic density."""
    relative = positions[:,3]-positions[:,0]
    closest = np.argmin(np.linalg.norm(relative, axis=1))
    indices = [np.argmin(abs(relative[:,0]+6)), closest, np.argmin(abs(relative[:,0]-6))]
    fig, axes = plt.subplots(1,3,figsize=(13,5.5))
    colors = ['#C43C39','#167B9B','#167B9B','#669C38']
    titles = ['Approach', 'Closest approach to oxygen', 'Departure']
    for ax, index, title in zip(axes, indices, titles):
        xyz = positions[index]
        # Lines connect the original water atom identities, not inferred bond orders.
        for j in [1,2]:
            ax.plot(xyz[[0,j],0],xyz[[0,j],1],color='#A9ADB2',lw=7,solid_capstyle='round',zorder=1)
        ax.plot(positions[:index+1,3,0],positions[:index+1,3,1],color=colors[3],lw=2,zorder=2)
        for j in range(4):
            radius = .52 if j==0 else .32
            ax.add_patch(Circle(xyz[j,:2],radius,facecolor=colors[j],edgecolor='white',lw=1.2,zorder=5))
            ax.text(*xyz[j,:2], 'O' if j==0 else ('p' if j==3 else 'H'),ha='center',va='center',color='white',fontsize=10,zorder=6)
        # Local measured motion determines the direction of the arrow.
        ahead = min(index+140,len(positions)-1)
        ax.annotate('',xy=positions[ahead,3,:2],xytext=xyz[3,:2],arrowprops=dict(arrowstyle='->',color=colors[3],lw=2),zorder=4)
        ax.set(xlim=(-8,8),ylim=(-4,5),aspect='equal')
        ax.axis('off')
        ax.set_title(title,fontsize=11,pad=24)
        ax.text(.5,1.02,f't = {trajectory[index,1]:.2f} a.u.',transform=ax.transAxes,ha='center',fontsize=9)
        ax.plot([-7,-5],[-3,-3],color='0.3',lw=2)
        ax.text(-6,-3.55,'2 bohr',ha='center',fontsize=9)
    fig.suptitle('H+–H2O collision: measured nuclear motion',fontsize=16,y=.94)
    fig.text(.5,.84,f'{metadata["energy_ev"]/1000:g} keV incoming proton | b = {metadata["impact_bohr"]:g} bohr | Octopus Ehrenfest dynamics',ha='center',fontsize=10)
    fig.text(.5,.18,'Red: oxygen   Blue: water hydrogens   Green: projectile nucleus and past trajectory',ha='center',fontsize=10)
    fig.text(.5,.10,'Equal-scale laboratory xy projection. Atom sizes and bond strokes are illustrative.\nNuclear positions come from the simulation; electron density and projectile charge state are not depicted.',ha='center',fontsize=9,linespacing=1.6)
    fig.text(.5,.035,'One collision trajectory, three time samples. Numerical uncertainty has not yet been quantified.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.025,right=.975,bottom=.24,top=.72,wspace=.1)
    fig.savefig(output/'collision.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
