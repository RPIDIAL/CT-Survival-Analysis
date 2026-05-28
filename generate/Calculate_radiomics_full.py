
# Run this to generate features. A shell is also provided to automate SLAB_MM
# and BONE_ERODE_MM scanning

import os
from pathlib import Path
import importlib
from termcolor import colored
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np

from full_muscle_groups import *

print(colored(f"PYTHON NOTIF: Generating Statistics {SLAB_MM}mm Slab", 'green'))

SLAB_SLICE_THICKNESS_RELATIVE = 20 # how many slices +- you want instead of in mm
USE_RELATIVE_SLAB_THICKNESS = False

out_dir = Path("bodycomp_metrics/data_full")
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / f"bodycomp_metrics_slab{int(SLAB_MM)}mm_boneErode{int(BONE_ERODE_MM)}mm.csv"
print("Output CSV:", out_csv)

def _row_to_fixed_fields(row: dict, fieldnames):
    return {k: row.get(k, np.nan) for k in fieldnames}

def list_case_ids(root_ct: Path):
    case_ids = []
    for p in sorted(root_ct.glob("*.nii.gz")):
        case_ids.append(p.name[:-7])
    existing = set(case_ids)
    for p in sorted(root_ct.glob("*.nii")):
        cid = p.name[:-4]
        if cid not in existing:
            case_ids.append(cid)
    return case_ids

def _process_one_case(case_id, metadata_df):
    row = {
        "case_id": case_id,
        "slab_mm": float(SLAB_MM),
        "bone_erode_mm": float(BONE_ERODE_MM),
    }

    try:
        row.update(compute_stats(case_id, metadata_df))
        row["error"] = ""
    except Exception as e:
        row["error"] = str(e)

    return row

c_ids = list_case_ids(ROOT_CT)
print("Total CT cases:", len(c_ids))

df = get_patient_metadata()

rows = []
fieldnames = set()

max_workers = min(32, os.cpu_count() or 1)

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(_process_one_case, cid, df): cid for cid in c_ids}

    for i, fut in enumerate(as_completed(futures), 1):
        cid = futures[fut]

        try:
            row = fut.result()
        except Exception as e:
            row = {
                "case_id": cid,
                "slab_mm": float(SLAB_MM),
                "bone_erode_mm": float(BONE_ERODE_MM),
                "error": f"worker_crash: {e}",
            }

        rows.append(row)
        fieldnames |= set(row.keys())

        if (i % 10) == 0 or i == len(c_ids):
            print(f"[processed {i}/{len(c_ids)}] last={cid}", flush=True)

fieldnames = sorted(fieldnames)

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(_row_to_fixed_fields(row, fieldnames))

print("Done.")
print("CSV:", out_csv)
