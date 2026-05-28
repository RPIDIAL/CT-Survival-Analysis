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
)
from xgb_utils import (
    fit_xgb_survival_cox_full_data,
    compute_xgb_feature_contributions,
)

XLSX_PATH = "/ICU/metadata"
BODYCOMP_CSV = "/generated/bodycomp/csv"
MAP_NEW_NAMES = "/csv/for/case/number/inconsistency/correction"

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

# ixed value kept only as fallback / reference
TOP_K_FOR_COX = 10

# candidate K values to evaluate
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

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

OUT_DIR = Path("xgb_topk_to_cox_outputs_nested_k_sweep")

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


def safe_event_auc(event_test: pd.Series, risk_test: np.ndarray):
    y = np.asarray(event_test).astype(int)
    if np.unique(y).size < 2:
        return np.nan
    return roc_auc_score(y, risk_test)


def fit_rank_xgb_on_train_only(
    df_train: pd.DataFrame,
    duration_train: pd.Series,
    event_train: pd.Series,
    xgb_feature_cols: list[str],
):
    fit_res = fit_xgb_survival_cox_full_data(
        df_train,
        duration_train,
        event_train,
        xgb_feature_cols,
        preprocess_kwargs=PREPROCESS_KWARGS,
        xgb_params=XGB_PARAMS,
        seed=SEED,
        risk_groups=RISK_GROUPS,
        feature_filter_kwargs=XGB_FEATURE_FILTER_KWARGS,
    )

    _, _, xgb_summary = compute_xgb_feature_contributions(
        fit_res["booster"],
        fit_res["X_full"],
    )

    ranked_features = xgb_summary["feature"].tolist()
    ranked_features = [f for f in ranked_features if f in df_train.columns]
    return fit_res, ranked_features


def fit_eval_single_k(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_train: pd.Series,
    duration_test: pd.Series,
    event_train: pd.Series,
    event_test: pd.Series,
    baseline: list[str],
    ranked_features: list[str],
    k: int,
):
    top_k_features = _unique_keep_order(baseline + ranked_features[:k])

    if len(top_k_features) == 0:
        raise RuntimeError("No features selected.")

    X_train_raw = df_train[top_k_features].copy()
    X_test_raw = df_test[top_k_features].copy()

    pre = CoxPreprocessor(**PREPROCESS_KWARGS).fit(X_train_raw)
    X_train = pre.transform(X_train_raw)
    X_test = pre.transform(X_test_raw)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    model_df_train = X_train.copy()
    model_df_train["duration"] = duration_train.values
    model_df_train["event"] = event_train.values

    cph = fit_cox(model_df_train, penalizer=PENALIZER, l1_ratio=L1_RATIO)

    risk_test = cph.predict_partial_hazard(X_test).to_numpy().reshape(-1)

    cidx = concordance_index(
        duration_test.values,
        -risk_test,
        event_test.values,
    )
    auc = safe_event_auc(event_test, risk_test)

    return {
        "top_k_features": top_k_features,
        "risk_test": risk_test,
        "cindex": float(cidx),
        "event_auc": float(auc) if pd.notna(auc) else np.nan,
    }


def run_nested_xgb_topk_cox_cv_k_sweep(
    df2: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    baseline: list[str],
    xgb_feature_cols: list[str],
    k_values: list[int],
    n_splits: int,
    seed: int,
):
    k_values = sorted(set(int(k) for k in k_values if int(k) > 0))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_k_rows = []
    heldout_rows = []
    fold_feature_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df2), start=1):
        df_train = df2.iloc[train_idx].copy()
        df_test = df2.iloc[test_idx].copy()

        duration_train = duration.iloc[train_idx].astype(float)
        duration_test = duration.iloc[test_idx].astype(float)

        event_train = event.iloc[train_idx].astype(int)
        event_test = event.iloc[test_idx].astype(int)

        _, ranked_features = fit_rank_xgb_on_train_only(
            df_train=df_train,
            duration_train=duration_train,
            event_train=event_train,
            xgb_feature_cols=xgb_feature_cols,
        )

        max_available = len(ranked_features)
        valid_k_values = [k for k in k_values if k <= max_available]

        best_fold_result = None

        for k in valid_k_values:
            res = fit_eval_single_k(
                df_train=df_train,
                df_test=df_test,
                duration_train=duration_train,
                duration_test=duration_test,
                event_train=event_train,
                event_test=event_test,
                baseline=baseline,
                ranked_features=ranked_features,
                k=k,
            )

            fold_k_rows.append({
                "fold": fold_idx,
                "k": int(k),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "events_train": int(event_train.sum()),
                "events_test": int(event_test.sum()),
                "n_features_selected": int(len(res["top_k_features"])),
                "cindex": float(res["cindex"]),
                "event_auc": float(res["event_auc"]) if pd.notna(res["event_auc"]) else np.nan,
            })

            for rank_i, feat in enumerate(res["top_k_features"], start=1):
                fold_feature_rows.append({
                    "fold": fold_idx,
                    "k": int(k),
                    "rank_within_fold_topk": rank_i,
                    "feature": feat,
                })

            if (best_fold_result is None) or (res["cindex"] > best_fold_result["cindex"]):
                best_fold_result = {
                    "k": int(k),
                    "top_k_features": res["top_k_features"],
                    "risk_test": res["risk_test"],
                    "cindex": float(res["cindex"]),
                    "event_auc": float(res["event_auc"]) if pd.notna(res["event_auc"]) else np.nan,
                }

            print(
                f"Fold {fold_idx} | k={k}: "
                f"train={len(train_idx)} test={len(test_idx)} "
                f"events_train={int(event_train.sum())} events_test={int(event_test.sum())} "
                f"cindex={res['cindex']:.4f} "
                f"auc={res['event_auc'] if pd.notna(res['event_auc']) else np.nan}"
            )

        heldout_rows.append(pd.DataFrame({
            "fold": fold_idx,
            "selected_k_for_fold": int(best_fold_result["k"]),
            "Subject ID": df_test["Subject ID"].values,
            "duration": duration_test.values,
            "event": event_test.values,
            "risk_score": best_fold_result["risk_test"],
        }))

        print(
            f"Fold {fold_idx} best k = {best_fold_result['k']} "
            f"(cindex={best_fold_result['cindex']:.4f}, "
            f"auc={best_fold_result['event_auc'] if pd.notna(best_fold_result['event_auc']) else np.nan})"
        )

    fold_k_df = pd.DataFrame(fold_k_rows)
    heldout_df = pd.concat(heldout_rows, ignore_index=True)
    fold_features_df = pd.DataFrame(fold_feature_rows)

    k_summary_df = (
        fold_k_df.groupby("k", as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_cindex=("cindex", "mean"),
            std_cindex=("cindex", "std"),
            mean_event_auc=("event_auc", "mean"),
            std_event_auc=("event_auc", "std"),
        )
        .sort_values(["mean_cindex", "mean_event_auc", "k"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    return fold_k_df, heldout_df, fold_features_df, k_summary_df

def choose_final_k(k_summary_df: pd.DataFrame) -> int:
    if len(k_summary_df) == 0:
        raise RuntimeError("k_summary_df is empty.")

    k_summary_df = k_summary_df.sort_values(
        ["mean_cindex", "mean_event_auc", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return int(k_summary_df.loc[0, "k"])


def fit_final_models_on_full_data(
    df2: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    baseline: list[str],
    xgb_feature_cols: list[str],
    top_k: int,
):
    fit_res, ranked_features = fit_rank_xgb_on_train_only(
        df_train=df2,
        duration_train=duration,
        event_train=event,
        xgb_feature_cols=xgb_feature_cols,
    )
    top_k_features = _unique_keep_order(baseline + ranked_features[:top_k])

    Xraw = df2[top_k_features].copy()
    pre = CoxPreprocessor(**PREPROCESS_KWARGS).fit(Xraw)
    X = pre.transform(Xraw)

    model_df = X.copy()
    model_df["duration"] = duration.values
    model_df["event"] = event.values

    cph = fit_cox(model_df, penalizer=PENALIZER, l1_ratio=L1_RATIO)
    return fit_res, ranked_features, top_k_features, pre, cph, model_df


def main():
    df2, duration, event, info = build_merged_dataset()
    baseline, candidates, xgb_feature_cols = resolve_feature_lists(df2)

    print("Final cohort:", info["n"])
    print("Merged N:", len(df2), "| events:", int(event.sum()))
    print("Resolved baseline features:", len(baseline))
    print("Resolved candidate features:", len(candidates))
    print("Resolved XGBoost ranking features:", len(xgb_feature_cols))
    print("K values to test:", K_VALUES)
    print("Selection note: XGBoost feature ranking is performed inside each CV training fold.")

    condition = _safe(SEVERITY_SCORE) if USE_SEVERITY_SCORE else "no_severity_score"
    out_dir = OUT_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_k_df, heldout_df, fold_features_df, k_summary_df = run_nested_xgb_topk_cox_cv_k_sweep(
        df2=df2,
        duration=duration,
        event=event,
        baseline=baseline,
        xgb_feature_cols=xgb_feature_cols,
        k_values=K_VALUES,
        n_splits=N_SPLITS,
        seed=SEED,
    )

    fold_k_df.to_csv(out_dir / "nested_cv_k_sweep_fold_metrics.csv", index=False)
    heldout_df.to_csv(out_dir / "nested_cv_bestk_heldout_predictions.csv", index=False)
    fold_features_df.to_csv(out_dir / "nested_cv_k_sweep_fold_selected_features.csv", index=False)
    k_summary_df.to_csv(out_dir / "nested_cv_k_summary.csv", index=False)

    final_k = choose_final_k(k_summary_df)
    print(f"Chosen final k from outer-fold performance: {final_k}")

    fit_res_full, ranked_features_full, top_k_features_full, pre_full, cph_full, model_df_full = (
        fit_final_models_on_full_data(
            df2=df2,
            duration=duration,
            event=event,
            baseline=baseline,
            xgb_feature_cols=xgb_feature_cols,
            top_k=final_k,
        )
    )

    pd.DataFrame({
        "rank": np.arange(1, len(ranked_features_full) + 1),
        "feature": ranked_features_full,
    }).to_csv(out_dir / "full_data_xgb_ranked_features_after_k_selection.csv", index=False)

    pd.DataFrame({
        "feature": top_k_features_full,
    }).to_csv(out_dir / "full_data_topk_features_after_k_selection.csv", index=False)

    cfg = {
        "selection_method": "xgboost_train_fold_only_mean_abs_contribution",
        "selection_is_nested_within_cox_cv": True,
        "k_values_tested": K_VALUES,
        "chosen_final_k": int(final_k),
        "baseline_features": baseline,
        "candidate_feature_count": len(candidates),
        "xgb_feature_count": len(xgb_feature_cols),
        "cox_feature_cols_final_full_data_refit": top_k_features_full,
        "xgb_feature_filter_kwargs": XGB_FEATURE_FILTER_KWARGS,
        "xgb_params": XGB_PARAMS,
        "cox_penalizer": PENALIZER,
        "cox_l1_ratio": L1_RATIO,
        "preprocess_kwargs": PREPROCESS_KWARGS,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "risk_groups": RISK_GROUPS,
        "k_summary": k_summary_df.to_dict(orient="records"),
    }

    fixedk_df = (
        fold_k_df[fold_k_df["k"] == final_k]
        .sort_values("fold")
        .reset_index(drop=True)
    )

    if fixedk_df["fold"].nunique() != N_SPLITS:
        raise RuntimeError(
            f"Expected {N_SPLITS} folds for final_k={final_k}, got {fixedk_df['fold'].nunique()}"
        )

    save_cox_results(
        out_dir=str(out_dir / "cox_results_full_data_refit"),
        cph=cph_full,
        model_df=model_df_full,
        feature_cols_final=top_k_features_full,
        config=cfg,
        cv_cidxs=fixedk_df["cindex"].to_numpy(dtype=float),
        cv_aucs=fixedk_df["event_auc"].to_numpy(dtype=float),
    )

    manifest = {
        "cohort_n": int(len(df2)),
        "events": int(event.sum()),
        "k_values_tested": K_VALUES,
        "chosen_final_k": int(final_k),
        "cox_feature_count_final_full_refit": len(top_k_features_full),
        "nested_cv_finalk_cindex_mean": float(fixedk_df["cindex"].mean()),
        "nested_cv_finalk_cindex_std": float(fixedk_df["cindex"].std(ddof=1)) if len(fixedk_df) > 1 else 0.0,
        "nested_cv_finalk_event_auc_mean": float(fixedk_df["event_auc"].mean(skipna=True)),
        "nested_cv_finalk_event_auc_std": (
            float(fixedk_df["event_auc"].std(ddof=1, skipna=True))
            if fixedk_df["event_auc"].notna().sum() > 1 else 0.0
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved outputs to:", out_dir.resolve())
    print(" - nested_cv_k_sweep_fold_metrics.csv")
    print(" - nested_cv_bestk_heldout_predictions.csv")
    print(" - nested_cv_k_sweep_fold_selected_features.csv")
    print(" - nested_cv_k_summary.csv")
    print(" - full_data_xgb_ranked_features_after_k_selection.csv")
    print(" - full_data_topk_features_after_k_selection.csv")
    print(" - cox_results_full_data_refit/")
    print(" - manifest.json")

    # Trying to figure out ways to select a final feature set instead of varying by CV

if __name__ == "__main__":
    main()