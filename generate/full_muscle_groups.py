import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import SimpleITK as sitk
import os
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from icu_io import load_metadata_xlsx, trim_patient_rows
import pandas as pd
from cox_utils import resolve_first_match
from copy import deepcopy

ROOT_CT   = Path("/put/path/to/nifti/root/ct")
ROOT_SEG  = Path("/put/path/to/totalseg/root/here")
ROOT_METADATA = Path("/put/additional/metadata/features/here") # This contains the list if IBW per patient
MAP_NEW_NAMES = Path("/csv/for/case/number/inconsistency/correction") # This is because the provided metadata and ct files had inconsistent naming conventions, so this corrects mapping

SHEET_NAME = 0
HEADER_ROWS = 1
JOINER = " | "
COL_CASE = "Case #"

SLAB_MM = float(os.environ.get("SLAB_MM", 30.0))
BONE_ERODE_MM = float(os.environ.get("BONE_ERODE_MM", 0.0))

# Current Landmarks | you are welcome to add to the list
GENERIC_LANDMARK                = "max_area_slice"
T12_CENTER_LANDMARK             = "T12_center"
AORTA_SUPERIOR_PLUS_1_LANDMARK  = "aorta_superior_plus_1"
MID_T7_T8_LANDMARK              = "SAT_center"

def sitk_aff2axcodes(img: sitk.Image):
    """
    Return orientation codes like ('R','A','S') for a SimpleITK image.
    Order corresponds to array axes (z, y, x) -> returned as (Z,Y,X).
    """
    D = np.array(img.GetDirection()).reshape(3, 3)

    labels = {
        0: ('R', 'L'),  # +X, -X
        1: ('A', 'P'),  # +Y, -Y
        2: ('S', 'I'),  # +Z, -Z
    }

    axcodes_xyz = []
    for i in range(3):
        v = D[:, i]
        j = np.argmax(np.abs(v))
        sign = np.sign(v[j])
        axcodes_xyz.append(labels[j][0] if sign > 0 else labels[j][1])

    # Convert xyz → zyx (array order)
    axcodes_zyx = (axcodes_xyz[2], axcodes_xyz[1], axcodes_xyz[0])
    return axcodes_zyx

def find_si_axis(img: sitk.Image):
    """
    Return (si_axis, si_code) where:
      - si_axis is the array axis (0=z,1=y,2=x)
      - si_code is 'S' if increasing index goes Superior,
                 'I' if increasing index goes Inferior
    """
    codes = sitk_aff2axcodes(img)
    for i, c in enumerate(codes):
        if c in ("S", "I"):
            return i, c
    return 0, "S"

def _take_slice(mask3d: np.ndarray, axis: int, idx: int):
    return np.take(mask3d, idx, axis=axis)

def _mm2_to_cm2(x_mm2: float) -> float:
    return x_mm2 / 100.0

def _csa_cm2_on_axis(mask3d: np.ndarray, axis: int, idx: int, zooms_mm):
    """
    2D area on slice orthogonal to `axis`.
    zooms_mm: (sz,sy,sx)
    """
    in_axes = [0, 1, 2]
    in_axes.remove(axis)
    s0 = zooms_mm[in_axes[0]]
    s1 = zooms_mm[in_axes[1]]
    mask2d = _take_slice(mask3d, axis, idx)
    area_mm2 = float(mask2d.sum()) * (s0 * s1)
    return _mm2_to_cm2(area_mm2)

def center_index_on_axis(seg: np.ndarray, label_id: int, axis: int):
    idxs = np.where(seg == label_id)[axis]
    if idxs.size == 0:
        return None
    return int(np.round(float(idxs.mean())))

def safe_hu_stats(ct: np.ndarray, mask: np.ndarray, hu_min=-9999.0, hu_max=9999.0, min_voxels: int = 0):
    vals = ct[mask]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= hu_min) & (vals <= hu_max)]
    if vals.size < min_voxels:
        return dict(HU_mean=np.nan, HU_median=np.nan, HU_p10=np.nan, HU_p90=np.nan)
    return dict(
        HU_mean=float(np.mean(vals)),
        HU_median=float(np.median(vals)),
        HU_p10=float(np.percentile(vals, 10)),
        HU_p90=float(np.percentile(vals, 90)),
    )

def _pick_pma_slice_by_aorta(total_seg: np.ndarray, ct_img: sitk.Image, aorta_id: int):
    """
    Approximate paper landmark: "first axial slice above the superior aspect of the aortic arch".
    With only 'aorta' label, we do:
    1) find the most superior slice containing aorta (along SI axis)
    2) move 1 slice further superior

    Returns: (axis, idx, landmark_str)
    """
    aorta = (total_seg == aorta_id)
    if aorta.sum() == 0:
        return None, None, "aorta_not_found"

    axis, code = find_si_axis(ct_img)
    idxs = np.where(aorta)[axis]

    # superior-most index depends on whether axis direction is 'S' or 'I'
    sup_idx = int(idxs.max() if code == "S" else idxs.min())

    # move one slice further "superior"
    idx = sup_idx + 1 if code == "S" else sup_idx - 1
    idx = int(np.clip(idx, 0, aorta.shape[axis] - 1))
    return axis, idx, f"aorta_superior+1(axis={axis},code={code})"

def _pick_slice_mid_T7_T8(total_seg: np.ndarray, ct_img: sitk.Image, t7_id: int, t8_id: int):
    axis, code = find_si_axis(ct_img)
    z7 = center_index_on_axis(total_seg, t7_id, axis)
    z8 = center_index_on_axis(total_seg, t8_id, axis)
    if (z7 is not None) and (z8 is not None):
        return axis, int(np.round((z7 + z8) / 2.0)), "mid_T7_T8"
    if z7 is not None:
        return axis, z7, "T7_center"
    if z8 is not None:
        return axis, z8, "T8_center"
    return None, None, "T7T8_not_found"

def _pick_slice_T12_center(total_seg: np.ndarray, ct_img: sitk.Image, t12_id: int):
    axis, code = find_si_axis(ct_img)
    idx = center_index_on_axis(total_seg, t12_id, axis)
    if idx is None:
        return None, None, "T12_not_found"
    return axis, idx, "T12_center"

def _pick_slice_by_max_area(mask3d: np.ndarray, axis: int):

    if mask3d.sum() == 0:
        return None, "mask_empty"

    in_axes = [0, 1, 2]
    in_axes.remove(axis)
    per_idx = mask3d.sum(axis=tuple(in_axes))
    idx = int(np.argmax(per_idx))
    return idx, GENERIC_LANDMARK

def _mm3_to_cm3(x_mm3: float) -> float:
    return x_mm3 / 1000.0

def slab_thickness_cm(i1: int, i2: int, spacing_mm_along_axis: float):
    thickness_mm = (i2 - i1 + 1) * spacing_mm_along_axis
    return thickness_mm / 10.0  # mm -> cm

def total_mask_thickness_cm(mask3d, axis, spacing_zyx):

    spacing = spacing_zyx[axis]

    idxs = np.where(mask3d)[axis]

    if idxs.size == 0:
        return 0

    i1 = idxs.min()
    i2 = idxs.max()

    thickness_mm = (i2 - i1 + 1) * spacing

    return thickness_mm / 10.0

def _slab_bounds(center_idx: int, slab_mm: float, spacing_mm_along_axis: float, max_len: int):
    half_slices = int(np.round((slab_mm / 2.0) / spacing_mm_along_axis))
    i1 = max(0, center_idx - half_slices)
    i2 = min(max_len - 1, center_idx + half_slices)
    return i1, i2

def _take_slab(mask3d: np.ndarray, axis: int, i1: int, i2: int):
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = slice(i1, i2 + 1)
    return mask3d[tuple(sl)]

def _vol_cm3(mask3d: np.ndarray, zooms_mm):
    sz, sy, sx = zooms_mm
    vol_mm3 = float(mask3d.sum()) * (sz * sy * sx)
    return _mm3_to_cm3(vol_mm3)

####### Volume methods #########

def _slab_stats(mask3d: np.ndarray, ct_arr: np.ndarray, idx: int, axis: int, ct_sp_zyx):
    spacing_axis = ct_sp_zyx[axis]

    i1, i2 = _slab_bounds(idx, SLAB_MM, spacing_axis, mask3d.shape[axis])

    slab_mask = _take_slab(mask3d, axis, i1, i2)
    slab_ct = _take_slab(ct_arr, axis, i1, i2)

    slab_vol = _vol_cm3(slab_mask, ct_sp_zyx)

    thick_cm = slab_thickness_cm(i1, i2, spacing_axis)
    slab_mean_CSA_cm2 = slab_vol / thick_cm if thick_cm > 0 else np.nan

    hu = safe_hu_stats(slab_ct, slab_mask)

    return slab_vol, hu, slab_mean_CSA_cm2

def _generic_vol_stats(mask3d: np.ndarray, ct_arr: np.ndarray, idx: int, axis: int, ct_sp_zyx):

    vol_cm3 = _vol_cm3(mask3d, ct_sp_zyx)

    thick_cm = total_mask_thickness_cm(mask3d, axis, ct_sp_zyx)
    mean_CSA_cm2 = vol_cm3 / thick_cm if thick_cm > 0 else np.nan

    hu = safe_hu_stats(ct_arr, mask3d)

    return vol_cm3, hu, mean_CSA_cm2

####################################

def _empty_stats(prefix, landmark=np.nan):
    return {
        f"{prefix}_volume_cm3": np.nan,
        f"{prefix}_volume_over_IBW": np.nan,
        f"{prefix}_mean_CSA_cm2": np.nan,
        f"{prefix}_mean_CSA_cm2_over_IBW": np.nan,
        f"{prefix}_HU_mean": np.nan,
        f"{prefix}_HU_median": np.nan,
        f"{prefix}_HU_p10": np.nan,
        f"{prefix}_HU_p90": np.nan,
        f"{prefix}_slice_idx": np.nan,
        f"{prefix}_landmark": landmark,
        f"{prefix}_CSA_cm2": np.nan,
    }

def _full_vol_stats(mask3d: np.ndarray, ct_label, seg_label, slice_func, vol_func, ibw_kg=None, prefix="muscle"):
    out = {}

    ct_arr = ct_label["arr"]
    ct_img = ct_label["img"]
    ct_sp_zyx = ct_label["sp_zyx"]

    if mask3d.sum() == 0:
        return _empty_stats(prefix)

    axis_si, idx, rule = slice_func(mask3d, ct_img, seg_label)

    if axis_si is None or idx is None:
        return _empty_stats(prefix, landmark=rule)

    vol_cm3, hu, mean_CSA_cm2 = vol_func(mask3d, ct_arr, idx, axis_si, ct_sp_zyx)

    out[f"{prefix}_volume_cm3"] = vol_cm3
    out[f"{prefix}_volume_over_IBW"] = (
        vol_cm3 / float(ibw_kg)
        if ibw_kg is not None and np.isfinite(ibw_kg) and ibw_kg > 0
        else np.nan
    )
    out[f"{prefix}_mean_CSA_cm2"] = mean_CSA_cm2
    out[f"{prefix}_mean_CSA_cm2_over_IBW"] = (
        mean_CSA_cm2 / float(ibw_kg)
        if ibw_kg is not None and np.isfinite(ibw_kg) and ibw_kg > 0
        else np.nan
    )

    out[f"{prefix}_HU_mean"] = hu["HU_mean"]
    out[f"{prefix}_HU_median"] = hu["HU_median"]
    out[f"{prefix}_HU_p10"] = hu["HU_p10"]
    out[f"{prefix}_HU_p90"] = hu["HU_p90"]

    out[f"{prefix}_slice_idx"] = idx if idx is not None else np.nan
    out[f"{prefix}_landmark"] = rule

    if idx is None:
        out[f"{prefix}_CSA_cm2"] = np.nan
    else:
        out[f"{prefix}_CSA_cm2"] = _csa_cm2_on_axis(mask3d, axis_si, idx, ct_sp_zyx)

    return out

def resample_label_to_ref(seg_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    """ Resample using nearest neighbor """
    if (seg_img.GetSize() == ref_img.GetSize() and
        seg_img.GetSpacing() == ref_img.GetSpacing() and
        seg_img.GetOrigin() == ref_img.GetOrigin() and
        seg_img.GetDirection() == ref_img.GetDirection()):
        return seg_img
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    return r.Execute(seg_img)

def _read_label_on_ct(case_id, label, ct_img, ct_arr):
    label_path = ROOT_SEG / case_id / label

    img = sitk.ReadImage(str(label_path))
    img_rs = resample_label_to_ref(img, ct_img)   # NN

    out = sitk.GetArrayFromImage(img_rs).astype(np.int16)

    assert ct_arr.shape == out.shape

    return out

def read_sitk(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    sp = img.GetSpacing()              # (x,y,z)
    sp_zyx = (sp[2], sp[1], sp[0])
    return img, arr, sp_zyx

####### Create Function Wrappers ################################
def _pick_t12_center(mask3d, ct_img, seg_label):
    return _pick_slice_T12_center(
        seg_label["total"]["classes"],
        ct_img,
        seg_label["total"]["labels"]["vertebrae_T12"]
    )

def _pick_pma_aorta(mask3d, ct_img, seg_label):
    return _pick_pma_slice_by_aorta(
        seg_label["total"]["classes"],
        ct_img,
        seg_label["total"]["labels"]["aorta"]
    )

def _pick_sat_mid_t7_t8(mask3d, ct_img, seg_label):
    return _pick_slice_mid_T7_T8(
        seg_label["total"]["classes"],
        ct_img,
        seg_label["total"]["labels"]["vertebrae_T7"],
        seg_label["total"]["labels"]["vertebrae_T8"]
    )

def _generic_slice_pick_max_area(mask3d, ct_img, seg_label):
    axis, _ = find_si_axis(ct_img)
    idx, rule = _pick_slice_by_max_area(mask3d, axis)
    return axis, idx, rule

######################################################################################

# Little trick to append the classes into the dictionary

# WARNING!!!! Call this instead of using _PRIVATE_SEG_LABELS
def _runtime_construct_seg_label(case_id, seg_label):

    ct_path = ROOT_CT / f"{case_id}.nii.gz"

    ct_label = dict(zip(CT_LABEL, read_sitk(ct_path)))

    for parent_name, info in seg_label.items():
        info["classes"] = _read_label_on_ct(case_id, f"{parent_name}.nii", ct_label["img"], ct_label["arr"])

    return ct_label, seg_label

def _build_generic_mask(label_src:str, label, seg_label):
    arr = seg_label[label_src]["classes"]
    mask = (arr == label)

    return mask

def _build_special_mask(group, seg_label):

    try:
        ref_shape = next(iter(seg_label.values()))["classes"].shape
    except TypeError as e:
        raise TypeError("FAILED TO FIND CLASS | Did you call runtime_construct_seg_label before calling this function?") from e

    mask = np.zeros(ref_shape, dtype=bool)

    for src_name, label_names in group["sources"]:
        arr = seg_label[src_name]["classes"]
        label_map = seg_label[src_name]["labels"]

        for label_name in label_names:
            label_id = label_map[label_name]
            mask |= (arr == label_id)

    return mask

##### T12 BONE DENSITY SPECIAL PRUNING #######
def _prune_convex_pieces(mask2d: np.ndarray, spacing_zyx, start_diam_mm=7.0, allow_mm=7.0):

    m = mask2d.astype(bool)

    _, sy, sx = spacing_zyx
    dist = distance_transform_edt(mask2d, sampling=(sy, sx))

    seed = dist >= float(start_diam_mm)
    if seed.sum() == 0:
        # fallback: seed from top quantile of dist
        thr = np.quantile(dist[m], 0.95)
        seed = dist >= thr
        if seed.sum() == 0:
            return mask2d.astype(np.uint8)

    allowed = dist >= float(allow_mm)

    # Flood-fill (geodesic reconstruction): grow seed but only inside allowed
    kept = ndi.binary_propagation(seed, mask=allowed)

    lab, n = ndi.label(kept)
    if n > 1:
        counts = np.bincount(lab.ravel()); counts[0] = 0
        kept = (lab == counts.argmax())

    return kept.astype(np.uint8)

def _make_core_mask(**kwargs):
    """Core = inside voxels whose distance to boundary >= erode_mm."""

    ct_label = kwargs["ct_label"]
    spacing_zyx = ct_label["sp_zyx"]
    mask_zyx = kwargs["mask_zyx"]
    remove_tails = kwargs["remove_tails"]
    erode_mm = kwargs["erode_mm"]

    m = (mask_zyx > 0)
    if m.sum() == 0:
        return m

    if remove_tails:
        pruned = np.zeros_like(m, dtype=np.uint8)
        for z in range(m.shape[0]):
            if m[z].sum() == 0:
                continue
            pruned[z] = _prune_convex_pieces(m[z], spacing_zyx).astype(np.uint8)
        m2 = pruned.astype(bool)
        if m2.sum() == 0:
            m2 = m
    else:
        m2 = m

    dist = distance_transform_edt(m2, sampling=spacing_zyx)  # spacing in (z,y,x)

    core = m2 & (dist >= float(erode_mm))
    if core.sum() == 0:
        core = m
    return core

##################################################################

def _norm_id(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().replace("\u00a0", "").replace(" ", "")

    # If it looks like a float ending in .0, collapse only that case
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            s = head

    return s

def _resolve_kwargs(spec_kwargs, context):
    resolved = {}
    for param_name, source in spec_kwargs.items():
        if isinstance(source, str) and source in context:
            resolved[param_name] = context[source]
        else:
            resolved[param_name] = source
    return resolved

def _metadata_id_from_case_id(case_id):
    case_id = _norm_id(case_id)
    return case_id[:-8] if case_id.endswith("_AMCAnon") else case_id

def _get_ibw_of_subject(case_id, df):
    meta_id = _metadata_id_from_case_id(case_id)

    if meta_id not in df.index:
        print(f"[WARN] Metadata not found for case_id={case_id} -> meta_id={meta_id}", flush=True)
        return np.nan

    return df.at[meta_id, "IBW in Kg"]

################## Forward facing functions #########################

def get_patient_metadata():
    df = load_metadata_xlsx(
        ROOT_METADATA,
        sheet_name=SHEET_NAME,
        header_rows=HEADER_ROWS,
        joiner=JOINER,
    )

    print("MAP_NEW_NAMES =", MAP_NEW_NAMES)

    mp = pd.read_csv(MAP_NEW_NAMES)

    lookup = pd.Series(
        mp["missing metadata"].map(_norm_id).values,
        index=mp["Case #"].map(_norm_id)
    )

    subj_col_raw = resolve_first_match(df.columns, "Subject ID")
    case_col_raw = resolve_first_match(df.columns, "Case #")

    mask = df[case_col_raw].map(_norm_id).isin(lookup.index)
    df = df.copy()
    df.loc[mask, subj_col_raw] = df[case_col_raw].map(_norm_id).loc[mask].map(lookup)

    df, case_col = trim_patient_rows(df, case_col_name=COL_CASE)
    print("Patient metadata rows after trimming:", len(df))
    print("Case col ->", case_col)

    subj_col_raw = resolve_first_match(df.columns, "Subject ID")

    df[subj_col_raw] = df[subj_col_raw].map(_norm_id)
    df = df.set_index(subj_col_raw, drop=False)

    return df

def compute_stats(case_id, metadata_df):
    out = {}

    ibw_kg = _get_ibw_of_subject(case_id, metadata_df)

    seg_label = deepcopy(_PRIVATE_SEG_LABELS)
    ct_label, seg_label = _runtime_construct_seg_label(case_id, seg_label)

    for classes_cat, classes in seg_label.items():
        for label_name, labels in classes["labels"].items():

            mask3d = _build_generic_mask(classes_cat, labels, seg_label=seg_label)

            gen_stats = _full_vol_stats(
                mask3d,
                ct_label=ct_label,
                seg_label=seg_label,
                slice_func=_generic_slice_pick_max_area,
                vol_func=_generic_vol_stats,
                ibw_kg=ibw_kg,
                prefix=label_name)

            out.update(gen_stats)

    for special_label, special in SPECIAL_GROUPS.items():
        mask3d = _build_special_mask(special, seg_label=seg_label)

        special_stats = _full_vol_stats(
            mask3d,
            ct_label=ct_label,
            seg_label=seg_label,
            slice_func=special["slice_func"],
            vol_func=special["vol_func"],
            ibw_kg=ibw_kg,
            prefix=special_label)

        out.update(special_stats)

        if "mask_modify" in special:

            # These are the known args you can choose from in SPECIAL_GROUPS
            # Any additional necessary args, you will have to compute them yourself
            AVAILABLE_RUNTIME_ARGS = {
                "mask3d": mask3d,
                "ct_label": ct_label,
                "ibw_kg": ibw_kg,
                "case_id": case_id,
            }

            modify_spec = special["mask_modify"]

            runtime_kwargs = _resolve_kwargs(
                modify_spec["kwargs"],
                AVAILABLE_RUNTIME_ARGS
            )

            new_mask = modify_spec["func"](**runtime_kwargs)
            modified_prefix = f"{special_label}{modify_spec['title_suffix']}"

            special_stats_modify = _full_vol_stats(
                new_mask,
                ct_label=ct_label,
                seg_label=seg_label,
                slice_func=special["slice_func"],
                vol_func=special["vol_func"],
                ibw_kg=ibw_kg,
                prefix=modified_prefix,
            )

            out.update(special_stats_modify)

    return out

#########################################################################################

CT_LABEL = {
        "img": None,
        "arr": None,
        "sp_zyx": None,
}

# Parent members should be named after files storing classes
_PRIVATE_SEG_LABELS = {
    "abdominal_muscles": {
        "classes": None,                  # populate this with runtime_seg_label_construct above
        "labels": {
            "pectoralis_major_right": 1,
            "pectoralis_major_left": 2,
            "rectus_abdominis_right": 3,
            "rectus_abdominis_left": 4,
            "serratus_anterior_right": 5,
            "serratus_anterior_left": 6,
            "latissimus_dorsi_right": 7,
            "latissimus_dorsi_left": 8,
            "trapezius_right": 9,
            "trapezius_left": 10,
            "external_oblique_right": 11,
            "external_oblique_left": 12,
            "internal_oblique_right": 13,
            "internal_oblique_left": 14,
            "erector_spinae_right": 15,
            "erector_spinae_left": 16,
            "transversospinalis_right": 17,
            "transversospinalis_left": 18,
            "psoas_major_right": 19,
            "psoas_major_left": 20,
            "quadratus_lumborum_right": 21,
            "quadratus_lumborum_left": 22,
        },
    },
    "thigh_shoulder_muscles": {
        "classes": None,
        "labels": {
            "quadriceps_femoris_left": 1,
            "quadriceps_femoris_right": 2,
            "thigh_medial_compartment_left": 3,
            "thigh_medial_compartment_right": 4,
            "thigh_posterior_compartment_left": 5,
            "thigh_posterior_compartment_right": 6,
            "sartorius_left": 7,
            "sartorius_right": 8,
            "deltoid": 9,
            "supraspinatus": 10,
            "infraspinatus": 11,
            "subscapularis": 12,
            "coracobrachial": 13,
            "trapezius": 14,
            "pectoralis_minor": 15,
            "serratus_anterior": 16,
            "teres_major": 17,
            "triceps_brachii": 18,
        },

    },
    "tissue_types": {
        "classes": None,
        "labels": {
            "subcutaneous_fat": 1,
            "torso_fat": 2,
            "skeletal_muscle": 3,
        },

    },
    "total": {
        "classes": None,
        "labels": {
            "spleen": 1,
            "kidney_right": 2,
            "kidney_left": 3,
            "gallbladder": 4,
            "liver": 5,
            "stomach": 6,
            "pancreas": 7,
            "adrenal_gland_right": 8,
            "adrenal_gland_left": 9,
            "lung_upper_lobe_left": 10,
            "lung_lower_lobe_left": 11,
            "lung_upper_lobe_right": 12,
            "lung_middle_lobe_right": 13,
            "lung_lower_lobe_right": 14,
            "esophagus": 15,
            "trachea": 16,
            "thyroid_gland": 17,
            "small_bowel": 18,
            "duodenum": 19,
            "colon": 20,
            "urinary_bladder": 21,
            "prostate": 22,
            "kidney_cyst_left": 23,
            "kidney_cyst_right": 24,
            "sacrum": 25,
            "vertebrae_S1": 26,
            "vertebrae_L5": 27,
            "vertebrae_L4": 28,
            "vertebrae_L3": 29,
            "vertebrae_L2": 30,
            "vertebrae_L1": 31,
            "vertebrae_T12": 32,
            "vertebrae_T11": 33,
            "vertebrae_T10": 34,
            "vertebrae_T9": 35,
            "vertebrae_T8": 36,
            "vertebrae_T7": 37,
            "vertebrae_T6": 38,
            "vertebrae_T5": 39,
            "vertebrae_T4": 40,
            "vertebrae_T3": 41,
            "vertebrae_T2": 42,
            "vertebrae_T1": 43,
            "vertebrae_C7": 44,
            "vertebrae_C6": 45,
            "vertebrae_C5": 46,
            "vertebrae_C4": 47,
            "vertebrae_C3": 48,
            "vertebrae_C2": 49,
            "vertebrae_C1": 50,
            "heart": 51,
            "aorta": 52,
            "pulmonary_vein": 53,
            "brachiocephalic_trunk": 54,
            "subclavian_artery_right": 55,
            "subclavian_artery_left": 56,
            "common_carotid_artery_right":  57,
            "common_carotid_artery_left": 58,
            "brachiocephalic_vein_left": 59,
            "brachiocephalic_vein_right": 60,
            "atrial_appendage_left": 61,
            "superior_vena_cava": 62,
            "inferior_vena_cava": 63,
            "portal_vein_and_splenic_vein": 64,
            "iliac_artery_left": 65,
            "iliac_artery_right": 66,
            "iliac_vena_left": 67,
            "iliac_vena_right": 68,
            "humerus_left": 69,
            "humerus_right": 70,
            "scapula_left": 71,
            "scapula_right": 72,
            "clavicula_left": 73,
            "clavicula_right": 74,
            "femur_left": 75,
            "femur_right": 76,
            "hip_left": 77,
            "hip_right": 78,
            "spinal_cord": 79,
            "gluteus_maximus_left": 80,
            "gluteus_maximus_right": 81,
            "gluteus_medius_left": 82,
            "gluteus_medius_right": 83,
            "gluteus_minimus_left": 84,
            "gluteus_minimus_right": 85,
            "autochthon_left": 86,
            "autochthon_right": 87,
            "iliopsoas_left": 88,
            "iliopsoas_right": 89,
            "brain": 90,
            "skull": 91,
            "rib_left_1": 92,
            "rib_left_2": 93,
            "rib_left_3": 94,
            "rib_left_4": 95,
            "rib_left_5": 96,
            "rib_left_6": 97,
            "rib_left_7": 98,
            "rib_left_8": 99,
            "rib_left_9": 100,
            "rib_left_10": 101,
            "rib_left_11": 102,
            "rib_left_12": 103,
            "rib_right_1": 104,
            "rib_right_2": 105,
            "rib_right_3": 106,
            "rib_right_4": 107,
            "rib_right_5": 108,
            "rib_right_6": 109,
            "rib_right_7": 110,
            "rib_right_8": 111,
            "rib_right_9": 112,
            "rib_right_10": 113,
            "rib_right_11": 114,
            "rib_right_12": 115,
            "sternum": 116,
            "costal_cartilages": 117,
        }

    }
}

SPECIAL_GROUPS = {
    "ESM": {
        "sources": [
            ("abdominal_muscles", ["erector_spinae_left", "erector_spinae_right"]),
        ],
        "landmark": T12_CENTER_LANDMARK,
        "slice_func": _pick_t12_center,
        "vol_func": _slab_stats,       # must return vol_cm3, hu, mean_CSA_cm2
    },
    "PMA": {
        "sources": [
            ("abdominal_muscles", ["pectoralis_major_left", "pectoralis_major_right"]),
            ("thigh_shoulder_muscles", ["pectoralis_minor"]),
        ],
        "landmark": AORTA_SUPERIOR_PLUS_1_LANDMARK,
        "slice_func": _pick_pma_aorta,
        "vol_func": _slab_stats,
    },
    "AMC_method_T12": {
        "sources": [
            ("total", ["vertebrae_T12"])
        ],
        "landmark": T12_CENTER_LANDMARK,
        "slice_func": _pick_t12_center,
        "vol_func": _slab_stats,

# Any modification from _build_special_mask should be formatted like this.
# Look and see if the params are already available in compute_stats -> AVAILABLE_RUNTIME_ARGS and map them in kwargs
        "mask_modify": {
            "func": _make_core_mask,
            "kwargs": {
                "mask_zyx": "mask3d",
                "ct_label": "ct_label",     # dict of original ct params
                "erode_mm": BONE_ERODE_MM,
                "remove_tails": True,
            },
            "title_suffix": "_core"
        }
    },
    "SAT": {
        "sources": [
            ("tissue_types", ["subcutaneous_fat"])
        ],
        "landmark": MID_T7_T8_LANDMARK,
        "slice_func": _pick_sat_mid_t7_t8,
        "vol_func": _slab_stats,
    },
    "Aggregated_rib": {
        "sources": [
            ("total", [
            "rib_left_1",
            "rib_left_2",
            "rib_left_3",
            "rib_left_4",
            "rib_left_5",
            "rib_left_6",
            "rib_left_7",
            "rib_left_8",
            "rib_left_9",
            "rib_left_10",
            "rib_left_11",
            "rib_left_12",
            "rib_right_1",
            "rib_right_2",
            "rib_right_3",
            "rib_right_4",
            "rib_right_5",
            "rib_right_6",
            "rib_right_7",
            "rib_right_8",
            "rib_right_9",
            "rib_right_10",
            "rib_right_11",
            "rib_right_12"])
        ],
        "landmark": GENERIC_LANDMARK,
        "slice_func": _generic_slice_pick_max_area,
        "vol_func": _generic_vol_stats,
    },
    "Aggregated_spine": {
        "sources": [
            ("total", [
            "vertebrae_S1",
            "vertebrae_L5",
            "vertebrae_L4",
            "vertebrae_L3",
            "vertebrae_L2",
            "vertebrae_L1",
            "vertebrae_T12",
            "vertebrae_T11",
            "vertebrae_T10",
            "vertebrae_T9",
            "vertebrae_T8",
            "vertebrae_T7",
            "vertebrae_T6",
            "vertebrae_T5",
            "vertebrae_T4",
            "vertebrae_T3",
            "vertebrae_T2",
            "vertebrae_T1",
            "vertebrae_C7",
            "vertebrae_C6",
            "vertebrae_C5",
            "vertebrae_C4",
            "vertebrae_C3",
            "vertebrae_C2",
            "vertebrae_C1"])
        ],
        "landmark": GENERIC_LANDMARK,
        "slice_func": _generic_slice_pick_max_area,
        "vol_func": _generic_vol_stats,
    }
}