#!/usr/bin/env python3
import uproot
import numpy as np

root_path = "build/dna_t0.root"
with uproot.open(root_path) as f:
    tree = f["step"]
    
    # Get process flags and names
    proc_flags = tree["flagProcess"].array(library="np")
    ke = tree["kineticEnergy"].array(library="np")
    
    print(f"Total events: {len(proc_flags)}")
    print(f"\nUnique process flags: {sorted(np.unique(proc_flags))}")
    
    # Count by process flag
    print(f"\nEvents by process flag:")
    for flag in sorted(np.unique(proc_flags)):
        mask = proc_flags == flag
        count = np.sum(mask)
        ke_range = f"{np.min(ke[mask]):.3f} - {np.max(ke[mask]):.3f} eV" if count > 0 else "N/A"
        
        # Map flag to name
        flag_map = {
            10: "Solvation", 11: "Elastic", 12: "Excitation", 
            13: "Ionisation", 14: "Attachment", 15: "VibExc",
            110: "mscEl"
        }
        name = flag_map.get(int(flag), f"Unknown({int(flag)})")
        print(f"  {int(flag):3d} ({name:12s}): {count:6d} events, KE: {ke_range}")
    
    # Check if processName exists
    if "processName" in tree.keys():
        print(f"\nProcessName column exists")
        proc_names = tree["processName"].array(library="np")
        print(f"ProcessName dtype: {proc_names.dtype}, shape: {proc_names.shape}")
        if len(proc_names) > 0:
            print(f"Sample processName values: {proc_names[:5]}")
    else:
        print(f"\nProcessName column NOT found")
        print(f"Available columns: {tree.keys()}")
