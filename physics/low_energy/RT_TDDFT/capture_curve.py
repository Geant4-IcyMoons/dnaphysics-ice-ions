"""Collect stationary P(H0) estimates; compare Hong Figure 2's 2*pi*b*P1.

No orientation averaging, tail extrapolation, or reference digitization is
implicit. Every computed point retains its source analysis and checksum.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .capture import sha256


def collect(paths):
    rows=[]
    seen=set()
    for path in paths:
        path=Path(path).resolve()
        result=json.loads(path.read_text())
        if result['status']!='analyzed_not_literature_validated' or result['stationarity_passed'] is not True:
            raise ValueError(f'Analysis has not passed stationarity: {path}')
        collision=result['collision']
        orientation=collision['orientation']
        b=float(collision['impact_parameter_bohr'])
        p=float(result['frames'][-1]['p_one_electron'])
        if collision['energy_ev_total']!=1000 or not np.isfinite([b,p]).all() or b<0 or not 0<=p<=1:
            raise ValueError('Expected a finite 1 keV capture probability')
        if (orientation,b) in seen:
            raise ValueError('Duplicate orientation/impact parameter')
        seen.add((orientation,b))
        rows.append({'orientation':orientation,'impact_parameter_bohr':b,
                     'p_one_electron':p,'two_pi_b_p1_bohr':float(2*np.pi*b*p),
                     'maximum_probability_change':result['maximum_probability_change'],
                     'source_analysis':str(path),'source_sha256':sha256(path)})
    if not rows:
        raise ValueError('No analyses supplied')
    return sorted(rows,key=lambda row:(row['orientation'],row['impact_parameter_bohr']))


def compare(rows, reference_path):
    """Compare at supplied matching b values only; do not invent reference values."""
    with Path(reference_path).open() as stream:
        references=list(csv.DictReader(stream))
    if not references:
        raise ValueError('Empty reference table')
    keyed={}
    for row in references:
        key=(row['orientation'],float(row['impact_parameter_bohr']))
        value=float(row['two_pi_b_p1_bohr'])
        if key in keyed or not np.isfinite([key[1],value]).all() or key[1]<0 or value<0:
            raise ValueError('Invalid or duplicate reference point')
        if not row['provenance'].strip():
            raise ValueError('Reference rows require provenance')
        keyed[key]=value
    if set(keyed)!={(row['orientation'],row['impact_parameter_bohr']) for row in rows}:
        raise ValueError('Reference and calculation grids must match exactly; no implicit interpolation')
    return [dict(row,reference_two_pi_b_p1_bohr=keyed[(row['orientation'],row['impact_parameter_bohr'])],
                 residual_bohr=row['two_pi_b_p1_bohr']-keyed[(row['orientation'],row['impact_parameter_bohr'])],
                 reference_sha256=sha256(reference_path)) for row in rows]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analyses',type=Path,nargs='+')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists')
    rows=collect(args.analyses)
    if args.reference:rows=compare(rows,args.reference)
    with args.output.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(args.output)


if __name__=='__main__':main()
