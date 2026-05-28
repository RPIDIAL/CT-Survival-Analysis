import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

sys.path.append('../')
from icu_io import load_metadata_xlsx, trim_patient_rows, build_6m_survival_cohort
from cox_utils import (
    resolve_first_match,
    drop_forbidden,
    CoxPreprocessor,
    fit_cox,
    save_cox_results,
    plot_cv_roc
)
from xgb_utils import (
    fit_xgb_survival_cox_full_data,
    compute_xgb_feature_contributions,
)

XLSX_PATH = "/ICU/metadata"
BODYCOMP_CSV = "/generated/bodycomp/csv"
MAP_NEW_NAMES = "/csv/for/case/number/inconsistency/correction"
# OUTER_SPLIT_CSV = "five/fold/split/csv"

SHEET_NAME = 0
HEADER_ROWS = 1
JOINER = " | "
DATE_FORMAT = "%m/%d/%Y"
CENSOR_DAYS = 180

COL_CASE = "Case #"
COL_SUBJ = "Subject ID"
COL_ADMITTED = "Admitted"
COL_DEATH_DATE = "Date of passing"
COL_SURV6M = "6-month survival: 1=alive; 0= dead; empty=unknlwn"

USE_SEVERITY_SCORE = False
SEVERITY_SCORE = "SAPSII"

# fixed value kept only as fallback / reference
TOP_K_FOR_COX = 10

PENALIZER = 0.1
L1_RATIO = 0.0
N_SPLITS = 5
SEED = 42
RISK_GROUPS = 4

PREPROCESS_KWARGS = dict(
    topk_cat=20,
    max_feat_missing=0.50,
    numeric_coerce_ratio=0.60,
    near_const_thresh=0.995,
    drop_first=True,
    standardize=True,
)

XGB_PARAMS = {
    "eta": 0.01,
    "max_depth": 2,
    "subsample": 0.6,
    "colsample_bytree": 0.2,
    "colsample_bynode": 0.2,
    "min_child_weight": 10.0,
    "reg_lambda": 10.0,
    "reg_alpha": 2.0,
}

XGB_FEATURE_FILTER_KWARGS = dict(
    enabled=False,
    top_k=50,
    corr_threshold=0.95,
    min_non_missing=0.50,
    min_unique=5,
    always_keep=[],
)

OUT_DIR = Path("xgb_topk_to_cox_outputs")

suffixes = (
    "volume_cm3",
    "volume_over_IBW",
    "mean_CSA_cm2",
    "mean_CSA_cm2_over_IBW",
    "HU_mean",
    "HU_median",
    "HU_p10",
    "HU_p90",
    "CSA_cm2",
)

_body_df = pd.read_csv(BODYCOMP_CSV)
body_comp_cols = [c for c in _body_df.columns if c.endswith(suffixes)]

FEATURES_BASE = [
    *body_comp_cols,
]

BASELINE = []
BLACKLIST = ["Case #"]

def _norm_id(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().replace("\u00a0", "").replace(" ", "")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def _unique_keep_order(items):
    return list(dict.fromkeys([x for x in items if x is not None and str(x).strip() != ""]))


def load_bodycomp_all(path: str) -> pd.DataFrame:
    bc = pd.read_csv(path)

    if "case_id" in bc.columns:
        bc["Subject ID"] = bc["case_id"].astype(str).str.replace(r"_AMCAnon$", "", regex=True)
    elif "Subject ID" not in bc.columns:
        raise RuntimeError("Bodycomp CSV must have 'case_id' or 'Subject ID'")

    bc["Subject ID"] = bc["Subject ID"].map(_norm_id)

    num_cols = bc.select_dtypes(include=[np.number]).columns.tolist()
    for c in bc.columns:
        if c in ["Subject ID"] or c in num_cols:
            continue
        s = pd.to_numeric(bc[c], errors="coerce")
        if s.notna().mean() >= 0.6:
            bc[c] = s
            num_cols.append(c)

    keep = ["Subject ID"] + sorted(set(num_cols))
    return bc[keep].copy()

def resolve_feature_lists(df2: pd.DataFrame):
    baseline = []
    for name in BASELINE:
        c = resolve_first_match(df2.columns, name)
        if c is not None:
            baseline.append(c)

    candidates = []
    for name in FEATURES_BASE:
        c = resolve_first_match(df2.columns, name)
        if c is not None:
            candidates.append(c)

    if USE_SEVERITY_SCORE:
        score_col = resolve_first_match(df2.columns, SEVERITY_SCORE)
        if score_col is None:
            raise RuntimeError(f"Cannot find severity score column: {SEVERITY_SCORE}")
        baseline.append(score_col)

    baseline = _unique_keep_order(drop_forbidden(baseline))
    candidates = _unique_keep_order(drop_forbidden(candidates))
    xgb_feature_cols = _unique_keep_order(baseline + candidates)
    return baseline, candidates, xgb_feature_cols

def build_merged_dataset():
    df = load_metadata_xlsx(
        XLSX_PATH,
        sheet_name=SHEET_NAME,
        header_rows=HEADER_ROWS,
        joiner=JOINER,
    )

    mp = pd.read_csv(MAP_NEW_NAMES)
    lookup = pd.Series(
        mp["missing metadata"].map(_norm_id).values,
        index=mp["Case #"].map(_norm_id),
    )

    subj_col_raw = resolve_first_match(df.columns, COL_SUBJ)
    case_col_raw = resolve_first_match(df.columns, COL_CASE)

    mask = df[case_col_raw].map(_norm_id).isin(lookup.index)
    df = df.copy()
    df.loc[mask, subj_col_raw] = df[case_col_raw].map(_norm_id).loc[mask].map(lookup)

    df, _ = trim_patient_rows(df, case_col_name=COL_CASE)
    df2, duration, event, info, _ = build_6m_survival_cohort(
        df,
        admitted_col_name=COL_ADMITTED,
        death_col_name=COL_DEATH_DATE,
        surv6m_col_name=COL_SURV6M,
        date_format=DATE_FORMAT,
        censor_days=CENSOR_DAYS,
        save_filter_info=False,
        subject_id_label=COL_SUBJ,
    )

    bc = load_bodycomp_all(BODYCOMP_CSV)

    subj_col = resolve_first_match(df2.columns, COL_SUBJ)
    if subj_col is None:
        raise RuntimeError("Cannot find 'Subject ID' column after XLSX processing.")

    df2 = df2.copy()
    df2["Subject ID"] = df2[subj_col].map(_norm_id)
    df2["duration"] = duration.loc[df2.index].values
    df2["event"] = event.loc[df2.index].values
    df2 = df2.merge(bc, on="Subject ID", how="inner")

    duration = df2["duration"].astype(float)
    event = df2["event"].astype(int)
    return df2, duration, event, info

def main():
    df2, duration, event, info = build_merged_dataset()
    baseline, candidates, xgb_feature_cols = resolve_feature_lists(df2)

    print("Final cohort:", info["n"])
    print("Merged N:", len(df2), "| events:", int(event.sum()))
    print("Resolved baseline features:", len(baseline))
    print("Resolved candidate features:", len(candidates))
    print("Resolved XGBoost ranking features:", len(xgb_feature_cols))
    print("Selection note: XGBoost feature ranking is performed inside each CV training fold.")

    condition = _safe(SEVERITY_SCORE) if USE_SEVERITY_SCORE else "no_severity_score"
    out_dir = OUT_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_dir / "nested_cv_k_sweep_fold_selected_features.csv")

    final_k = [5] # Selected k

    for k in final_k:

        out_dir = out_dir / f"top_{k}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_k = df[df["k"] == k]

        summary = (
            df_k.groupby("feature")
            .agg(
                n_folds=("fold", "nunique"),
                mean_rank=("rank_within_fold_topk", "mean"),
            )
            .sort_values(["n_folds", "mean_rank"], ascending=[False, True])
        )

        summary["rank_norm"] = summary["mean_rank"] / k

        summary["score"] = summary["n_folds"] - summary["rank_norm"]

        summary.to_csv(out_dir / "summary.csv", index=False)

        selected = (
            summary.sort_values("score", ascending=False)
            .head(k)
            .index
            .tolist()
        )

        final_features = selected

        from cox_utils import (
            CoxPreprocessor,
            fit_cox,
            save_cox_results,
            cox_kfold_cindex,
            cox_kfold_auc,
        )

        cv_cidxs = cox_kfold_cindex(
            df=df2,
            duration=duration,
            event=event,
            feature_cols=final_features,
            n_splits=N_SPLITS,
            seed=SEED,
            penalizer=PENALIZER,
            l1_ratio=L1_RATIO,
            preprocess_kwargs=PREPROCESS_KWARGS,
        )

        cv_aucs = cox_kfold_auc(
            df=df2,
            duration=duration,
            event=event,
            feature_cols=final_features,
            n_splits=N_SPLITS,
            seed=SEED,
            penalizer=PENALIZER,
            l1_ratio=L1_RATIO,
            preprocess_kwargs=PREPROCESS_KWARGS,
        )

        Xraw = df2[final_features].copy()
        pre = CoxPreprocessor(**PREPROCESS_KWARGS).fit(Xraw)
        X = pre.transform(Xraw)

        model_df = X.copy()
        model_df["duration"] = duration.values
        model_df["event"] = event.values

        cph = fit_cox(model_df, penalizer=PENALIZER, l1_ratio=L1_RATIO)

        save_cox_results(
            out_dir=str(out_dir / "cox_results_stable_features"),
            cph=cph,
            model_df=model_df,
            feature_cols_final=final_features,
            config={
                "feature_selection_method": "stability_aggregation",
                "selected_features": final_features,
                "k_used": final_k,
            },
            cv_cidxs=cv_cidxs,
            cv_aucs=cv_aucs,
        )

        model_defs = {
                "Albumin": ["Albumin"],
                "SAPSII": ["SAPSII"],
                "Body composition": [
                    *final_features,
                ],
                "Combined": [
                    "Albumin",
                    "SAPSII",
                    *final_features,
                ],
            }

        AUC_curve_results = plot_cv_roc(
            df2=df2,
            duration=duration,
            event=event,
            penalizer=PENALIZER,
            l1_ratio=L1_RATIO,
            preprocess_kwargs=PREPROCESS_KWARGS,
            model_defs=model_defs,
            n_splits=N_SPLITS,
            seed=SEED,
            out_png="compare_top_features.pdf"
        )

        print(AUC_curve_results)

        model_defs = {
            "SAPSII": ["SAPSII"],
            "APACHE II": ["APACHE II"],
            "SOFA": ["SOFA"],
            "Body composition": [
                *final_features,
            ],
        }

        AUC_curve_results = plot_cv_roc(
            df2=df2,
            duration=duration,
            event=event,
            penalizer=PENALIZER,
            l1_ratio=L1_RATIO,
            preprocess_kwargs=PREPROCESS_KWARGS,
            model_defs=model_defs,
            n_splits=N_SPLITS,
            seed=SEED,
            out_png="compare_validated_scores.pdf",
        )

        print(AUC_curve_results["delong"])

if __name__ == "__main__":
    main()