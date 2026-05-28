
import numpy as np
import pandas as pd
import re

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve

FORBIDDEN_KEYWORDS_DEFAULT = [
    "off vent", "day out of icu", "days vented", "days of icu",
    "icu free", "time to event", "date dead", "# days", "day out",
    "dead in hospital", "place of discharge", "discharge location", "case #"
]


def drop_forbidden(features, forbidden_keywords=FORBIDDEN_KEYWORDS_DEFAULT):
    out = []
    for f in features:
        fl = str(f).lower()
        if any(k in fl for k in forbidden_keywords):
            continue
        out.append(f)
    return out


def resolve_first_match(columns, name: str):
    cols = list(columns)
    if name in cols:
        return name
    key = str(name).lower()
    for c in cols:
        if key in str(c).lower():
            return c
    return None


class CoxPreprocessor:
    """
    Fit/transform preprocessor for Cox to avoid convergence issues:
      - numeric coercion for numeric-like cols
      - topK for categoricals (others -> "Other")
      - get_dummies with dummy_na
      - drop feature cols with missing rate > max_feat_missing
      - add missing indicators for numeric cols with missing in TRAIN
      - impute median (TRAIN medians)
      - drop constant / near-constant cols (based on TRAIN)
      - standardize using TRAIN mean/std
      - align TEST columns to TRAIN columns
    """
    def __init__(
        self,
        topk_cat: int = 20,
        max_feat_missing: float = 0.50,
        numeric_coerce_ratio: float = 0.60,
        near_const_thresh: float = 0.995,
        drop_first: bool = True,
        standardize: bool = True,
    ):
        self.topk_cat = topk_cat
        self.max_feat_missing = max_feat_missing
        self.numeric_coerce_ratio = numeric_coerce_ratio
        self.near_const_thresh = near_const_thresh
        self.drop_first = drop_first
        self.standardize = standardize

        # learned
        self.cat_keep_levels_ = {}
        self.train_columns_ = None
        self.medians_ = None
        self.means_ = None
        self.stds_ = None
        self.missing_indicator_cols_ = None

    def _coerce_numeric_like(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in X.columns:
            num = pd.to_numeric(X[c], errors="coerce")
            non_missing = X[c].notna().sum()
            ratio = (num.notna().sum() / non_missing) if non_missing > 0 else 0.0
            if ratio >= self.numeric_coerce_ratio:
                X[c] = num
        return X

    def _reduce_categoricals(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()
        cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("string"))]
        for c in cat_cols:
            s = X[c].astype("string")
            if fit:
                vc = s.value_counts(dropna=True)
                keep = set(vc.head(self.topk_cat).index.astype(str).tolist())
                self.cat_keep_levels_[c] = keep
            keep = self.cat_keep_levels_.get(c, set())
            X[c] = s.apply(lambda v: v if (pd.notna(v) and str(v) in keep) else ("Other" if pd.notna(v) else np.nan))
        return X

    def _to_dummies(self, X: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("string"))]
        if not cat_cols:
            return X
        return pd.get_dummies(X, columns=cat_cols, dummy_na=True, drop_first=self.drop_first)

    @staticmethod
    def _drop_constant_and_near_constant(X: pd.DataFrame, near_const_thresh: float) -> pd.DataFrame:
        # constant
        const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        X = X.drop(columns=const_cols, errors="ignore")

        # near-constant
        near_const_cols = []
        n = len(X)
        if n > 0:
            for c in X.columns:
                vc = X[c].value_counts(dropna=False)
                if len(vc) > 0 and (vc.iloc[0] / n) > near_const_thresh:
                    near_const_cols.append(c)
        X = X.drop(columns=near_const_cols, errors="ignore")
        return X

    def fit(self, X_raw: pd.DataFrame):
        X = self._coerce_numeric_like(X_raw)
        X = self._reduce_categoricals(X, fit=True)
        X = self._to_dummies(X)

        # drop feature columns with too much missingness (on TRAIN only)
        miss = X.isna().mean()
        keep_cols = miss[miss <= self.max_feat_missing].index
        X = X.loc[:, keep_cols].copy()

        # missing indicators (only for columns that have missing in TRAIN)
        miss_cols = X.columns[X.isna().any()].tolist()
        self.missing_indicator_cols_ = miss_cols

        if miss_cols:
            miss_df = X[miss_cols].isna().astype(int)
            miss_df = miss_df.add_suffix("_missing")
            X = pd.concat([X, miss_df], axis=1)

        # impute with TRAIN median
        X = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = X.median(numeric_only=True)
        X = X.fillna(self.medians_).fillna(0)

        # drop constant / near-constant on TRAIN
        X = self._drop_constant_and_near_constant(X, self.near_const_thresh)

        # standardize using TRAIN mean/std
        if self.standardize:
            self.means_ = X.mean()
            self.stds_ = X.std(ddof=0).replace(0, 1)
            X = (X - self.means_) / self.stds_

        self.train_columns_ = X.columns.tolist()
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.train_columns_ is None:
            raise RuntimeError("CoxPreprocessor must be fit() before transform().")

        X = self._coerce_numeric_like(X_raw)
        X = self._reduce_categoricals(X, fit=False)
        X = self._to_dummies(X)

        # keep same columns prior to missing-indicators stage is not necessary;
        # we align after building missing indicators
        # add missing indicators for TRAIN missing cols
        X = X.replace([np.inf, -np.inf], np.nan)
        miss_cols = self.missing_indicator_cols_ or []
        if miss_cols:
            miss_parts = {}
            for c in miss_cols:
                if c in X.columns:
                    miss_parts[c + "_missing"] = X[c].isna().astype(int)
                else:
                    miss_parts[c + "_missing"] = pd.Series(0, index=X.index, dtype=int)

            miss_df = pd.DataFrame(miss_parts, index=X.index)
            X = pd.concat([X, miss_df], axis=1)

        # impute using TRAIN medians
        if self.medians_ is not None:
            for c, med in self.medians_.items():
                if c in X.columns:
                    X[c] = X[c].fillna(med)
        X = X.fillna(0)

        # align to TRAIN columns
        X = X.reindex(columns=self.train_columns_, fill_value=0).copy()

        # standardize with TRAIN stats
        if self.standardize and (self.means_ is not None) and (self.stds_ is not None):
            X = (X - self.means_) / self.stds_

        # ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        return X

def fit_cox(model_df: pd.DataFrame, penalizer: float = 1.0, l1_ratio: float = 0.0) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(model_df, duration_col="duration", event_col="event")
    return cph

def make_strata(duration, event, risk_groups=3):
    duration = pd.Series(duration).copy()
    event = pd.Series(event).copy()

    # start with a numeric array
    strata = pd.Series(np.zeros(len(event), dtype=np.int64), index=event.index)

    died = (event == 1)
    if died.sum() > 0:
        death_bins = pd.qcut(
            duration[died],
            q=risk_groups,
            labels=False
        )

        strata.loc[died] = death_bins.to_numpy(dtype=np.int64) + 1

    return strata

def _norm_id(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().replace("\u00a0", "").replace(" ", "")

def cox_kfold_roc_from_split_csv_for_roc_curve(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    split_csv: str,
    subject_id_col: str = "Subject ID",
    split_id_col: str = "index",
    fold_col: str = "fold",
    split_col: str = "split",
    penalizer: float = 1.0,
    l1_ratio: float = 0.0,
    preprocess_kwargs=None,
):
    """
    Returns fold-wise ROC information using the exact same split/eval logic as
    cox_kfold_auc_from_split_csv.

    Output dict:
      {
        "folds": [
          {"fold": 1, "fpr": ..., "tpr": ..., "auc": ..., "n_test": ...},
          ...
        ],
        "mean_auc": ...,
        "std_auc": ...,
      }
    """
    preprocess_kwargs = preprocess_kwargs or {}
    split_csv = Path(split_csv)

    df = df.copy()
    duration = duration.loc[df.index].copy()
    event = event.loc[df.index].astype(int).copy()

    if subject_id_col not in df.columns:
        raise RuntimeError(f"'{subject_id_col}' not found in df columns.")

    df[subject_id_col] = df[subject_id_col].map(_norm_id)

    split_df = pd.read_csv(split_csv)
    needed = {fold_col, split_col, split_id_col}
    missing_cols = needed - set(split_df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Split CSV is missing columns: {sorted(missing_cols)}. "
            f"Found: {list(split_df.columns)}"
        )

    split_df = split_df.copy()
    split_df[split_id_col] = split_df[split_id_col].map(_norm_id)
    split_df[split_col] = split_df[split_col].astype(str).str.strip().str.lower()

    dup_mask = df[subject_id_col].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = sorted(df.loc[dup_mask, subject_id_col].dropna().unique().tolist())
        raise RuntimeError(
            "Duplicate Subject IDs found in df, so split IDs cannot be mapped uniquely. "
            f"Examples: {dup_ids[:10]}"
        )

    id_to_idx = pd.Series(df.index.values, index=df[subject_id_col].values)

    fold_results = []
    aucs = []

    for fold in sorted(split_df[fold_col].dropna().unique()):
        fold_rows = split_df.loc[split_df[fold_col] == fold]

        train_ids = fold_rows.loc[fold_rows[split_col] == "train", split_id_col].dropna().tolist()
        test_ids  = fold_rows.loc[fold_rows[split_col] == "test",  split_id_col].dropna().tolist()

        if len(train_ids) == 0 or len(test_ids) == 0:
            raise RuntimeError(f"Fold {fold} has empty train or test IDs.")

        missing_train = sorted(set(train_ids) - set(id_to_idx.index))
        missing_test  = sorted(set(test_ids) - set(id_to_idx.index))
        if missing_train or missing_test:
            raise RuntimeError(
                f"Fold {fold} contains IDs not present in df.\n"
                f"Missing train IDs (first 10): {missing_train[:10]}\n"
                f"Missing test IDs (first 10): {missing_test[:10]}"
            )

        tr_idx = id_to_idx.loc[train_ids].to_numpy()
        te_idx = id_to_idx.loc[test_ids].to_numpy()

        Xtr_raw = df.loc[tr_idx, feature_cols]
        Xte_raw = df.loc[te_idx, feature_cols]

        pre = CoxPreprocessor(**preprocess_kwargs).fit(Xtr_raw)
        Xtr = pre.transform(Xtr_raw)
        Xte = pre.transform(Xte_raw)

        train_df = Xtr.copy()
        train_df["duration"] = duration.loc[tr_idx].values
        train_df["event"] = event.loc[tr_idx].values

        test_df = Xte.copy()
        test_df["duration"] = duration.loc[te_idx].values
        test_df["event"] = event.loc[te_idx].values

        cph = fit_cox(train_df, penalizer=penalizer, l1_ratio=l1_ratio)

        risk = cph.predict_partial_hazard(
            test_df.drop(columns=["duration", "event"])
        ).values.reshape(-1)

        y = test_df["event"].values.astype(int)

        if np.unique(y).size < 2:
            print(f"Fold {fold}: ROC/AUC skipped (only one class in test fold)")
            fold_results.append({
                "fold": int(fold),
                "fpr": np.array([0.0, 1.0]),
                "tpr": np.array([0.0, 1.0]),
                "auc": np.nan,
                "n_test": len(y),
            })
            aucs.append(np.nan)
            continue

        fpr, tpr, _ = roc_curve(y, risk)
        auc = roc_auc_score(y, risk)

        fold_results.append({
            "fold": int(fold),
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(auc),
            "n_test": len(y),
        })
        aucs.append(auc)

        print(f"Fold {fold}: AUC = {auc:.4f}")

    aucs = np.asarray(aucs, dtype=float)
    valid = np.isfinite(aucs)

    out = {
        "folds": fold_results,
        "aucs": aucs,
        "mean_auc": float(np.nanmean(aucs)) if valid.any() else np.nan,
        "std_auc": float(np.nanstd(aucs, ddof=1)) if valid.sum() > 1 else 0.0,
    }

    if valid.any():
        print(f"\nMean fold AUC = {out['mean_auc']:.4f} ± {out['std_auc']:.4f}")
    else:
        print("\nAll fold AUC are NaN.")

    return out

def cox_kfold_cindex(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    n_splits: int = 5,
    seed: int = 42,
    penalizer: float = 1.0,
    l1_ratio: float = 0.0,
    preprocess_kwargs = None,
):
    """
    KFold CV with preprocessing FIT on train only (no leakage).
    """
    preprocess_kwargs = preprocess_kwargs or {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cidxs = []
    split_rows = []

    df = df.copy()
    duration = duration.loc[df.index]
    event = event.loc[df.index]

    for fold, (tr, te) in enumerate(kf.split(df), 1):
        tr_idx = df.index[tr]
        te_idx = df.index[te]

        for idx in tr_idx:
            split_rows.append({
                "fold": fold,
                "split": "train",
                "index": df.loc[idx, "Subject ID"],
            })

        for idx in te_idx:
            split_rows.append({
                "fold": fold,
                "split": "test",
                "index": df.loc[idx, "Subject ID"],
            })

        Xtr_raw = df.loc[tr_idx, feature_cols]
        Xte_raw = df.loc[te_idx, feature_cols]

        pre = CoxPreprocessor(**preprocess_kwargs).fit(Xtr_raw)
        Xtr = pre.transform(Xtr_raw)
        Xte = pre.transform(Xte_raw)

        train_df = Xtr.copy()
        train_df["duration"] = duration.loc[tr_idx].values
        train_df["event"] = event.loc[tr_idx].values

        test_df = Xte.copy()
        test_df["duration"] = duration.loc[te_idx].values
        test_df["event"] = event.loc[te_idx].values

        cph = fit_cox(train_df, penalizer=penalizer, l1_ratio=l1_ratio)

        # partial hazard: higher -> higher risk -> shorter survival, so use -risk for c-index
        risk = cph.predict_partial_hazard(test_df.drop(columns=["duration", "event"])).values.reshape(-1)
        ci = concordance_index(test_df["duration"].values, -risk, test_df["event"].values)
        cidxs.append(ci)

        # print(f"Fold {fold}: C-index = {ci:.4f}")

    # split_df = pd.DataFrame(split_rows)
    # split_df.to_csv("data/train_test_split.csv", index=False)

    cidxs = np.array(cidxs)
    # print(f"\n{n_splits}-fold mean C-index = {cidxs.mean():.4f} ± {cidxs.std(ddof=1):.4f}")
    return cidxs

def cox_kfold_auc(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    n_splits: int = 5,
    seed: int = 42,
    penalizer: float = 1.0,
    l1_ratio: float = 0.0,
    preprocess_kwargs = None,
):
    """
    KFold CV ROC-AUC for 6-month mortality classification:
      y = event (1=dead within 180d, 0=alive at 180d)
      score = Cox partial hazard (higher => higher death risk)
    """
    preprocess_kwargs = preprocess_kwargs or {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []

    df = df.copy()
    duration = duration.loc[df.index]
    event = event.loc[df.index].astype(int)

    for fold, (tr, te) in enumerate(kf.split(df), 1):
        tr_idx = df.index[tr]
        te_idx = df.index[te]

        Xtr_raw = df.loc[tr_idx, feature_cols]
        Xte_raw = df.loc[te_idx, feature_cols]

        pre = CoxPreprocessor(**preprocess_kwargs).fit(Xtr_raw)
        Xtr = pre.transform(Xtr_raw)
        Xte = pre.transform(Xte_raw)

        train_df = Xtr.copy()
        train_df["duration"] = duration.loc[tr_idx].values
        train_df["event"] = event.loc[tr_idx].values

        test_df = Xte.copy()
        test_df["duration"] = duration.loc[te_idx].values
        test_df["event"] = event.loc[te_idx].values

        cph = fit_cox(train_df, penalizer=penalizer, l1_ratio=l1_ratio)

        risk = cph.predict_partial_hazard(test_df.drop(columns=["duration", "event"])).values.reshape(-1)
        y = test_df["event"].values.astype(int)

        if np.unique(y).size < 2:
            # print(f"Fold {fold}: AUC = NaN (only one class in test fold)")
            aucs.append(np.nan)
            continue

        auc = roc_auc_score(y, risk)
        aucs.append(auc)
        # print(f"Fold {fold}: AUC = {auc:.4f}")

    aucs = np.array(aucs, dtype=float)
    valid = np.isfinite(aucs)
    # if valid.any():
    #     print(f"\n{n_splits}-fold mean AUC = {np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}")
    # else:
    #     print("\nAll folds AUC are NaN (check class balance / folds).")
    return aucs

def auc_from_oof(oof_risk: pd.Series, event: pd.Series) -> float:
    y = event.loc[oof_risk.index].astype(int).values
    s = oof_risk.values.astype(float)
    ok = np.isfinite(s)
    y = y[ok]; s = s[ok]
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, s))

def cox_kfold_roc_for_roc_curve(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    n_splits: int = 5,
    seed: int = 42,
    penalizer: float = 1.0,
    l1_ratio: float = 0.0,
    preprocess_kwargs=None,
):
    """
    Fold-wise ROC information using the same KFold split logic as
    cox_kfold_auc / cox_kfold_cindex.
    """
    preprocess_kwargs = preprocess_kwargs or {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    df = df.copy()
    duration = duration.loc[df.index]
    event = event.loc[df.index].astype(int)

    fold_results = []
    aucs = []

    for fold, (tr, te) in enumerate(kf.split(df), 1):
        tr_idx = df.index[tr]
        te_idx = df.index[te]

        Xtr_raw = df.loc[tr_idx, feature_cols]
        Xte_raw = df.loc[te_idx, feature_cols]

        pre = CoxPreprocessor(**preprocess_kwargs).fit(Xtr_raw)
        Xtr = pre.transform(Xtr_raw)
        Xte = pre.transform(Xte_raw)

        train_df = Xtr.copy()
        train_df["duration"] = duration.loc[tr_idx].values
        train_df["event"] = event.loc[tr_idx].values

        test_df = Xte.copy()
        test_df["duration"] = duration.loc[te_idx].values
        test_df["event"] = event.loc[te_idx].values

        cph = fit_cox(train_df, penalizer=penalizer, l1_ratio=l1_ratio)

        risk = cph.predict_partial_hazard(
            test_df.drop(columns=["duration", "event"])
        ).values.reshape(-1)

        y = test_df["event"].values.astype(int)

        if np.unique(y).size < 2:
            fold_results.append({
                "fold": int(fold),
                "fpr": np.array([0.0, 1.0]),
                "tpr": np.array([0.0, 1.0]),
                "auc": np.nan,
                "n_test": len(y),
            })
            aucs.append(np.nan)
            continue

        fpr, tpr, _ = roc_curve(y, risk)
        auc = roc_auc_score(y, risk)

        fold_results.append({
            "fold": int(fold),
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(auc),
            "n_test": len(y),
        })
        aucs.append(auc)

    aucs = np.asarray(aucs, dtype=float)
    valid = np.isfinite(aucs)

    return {
        "folds": fold_results,
        "aucs": aucs,
        "mean_auc": float(np.nanmean(aucs)) if valid.any() else np.nan,
        "std_auc": float(np.nanstd(aucs, ddof=1)) if valid.sum() > 1 else 0.0,
    }

def plot_cv_mean_roc(
    roc_info,
    label,
    ax=None,
    show_fold_curves=False,
    shade=True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for fr in roc_info["folds"]:
        fpr = fr["fpr"]
        tpr = fr["tpr"]
        auc = fr["auc"]

        if np.isfinite(auc):
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            if show_fold_curves:
                ax.plot(fpr, tpr, alpha=0.2, linewidth=1)

    if len(tprs) == 0:
        raise RuntimeError("No valid fold ROC curves to plot.")

    tprs = np.asarray(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0, ddof=1) if len(tprs) > 1 else np.zeros_like(mean_tpr)
    mean_tpr[-1] = 1.0

    ax.plot(
        mean_fpr,
        mean_tpr,
        linewidth=2.5,
        label=f"{label} (AUC = {roc_info['mean_auc']:.3f} ± {roc_info['std_auc']:.3f})",
    )

    if shade:
        lower = np.maximum(mean_tpr - std_tpr, 0)
        upper = np.minimum(mean_tpr + std_tpr, 1)
        ax.fill_between(mean_fpr, lower, upper, alpha=0.15)

    return ax

def plot_cv_roc(
    df2,
    duration,
    event,
    penalizer,
    l1_ratio,
    preprocess_kwargs,
    model_defs,
    out_png="cv_roc.png",
    split_csv=None,
    n_splits=5,
    seed=42,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    results = {}
    oof_risks = {}

    for label, cols in model_defs.items():
        if split_csv is not None:
            roc_info = cox_kfold_roc_from_split_csv_for_roc_curve(
                df=df2,
                duration=duration,
                event=event,
                feature_cols=cols,
                split_csv=split_csv,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                preprocess_kwargs=preprocess_kwargs,
            )

            oof_risks[label] = cox_kfold_oof_risk_from_split_csv(
                df=df2,
                duration=duration,
                event=event,
                feature_cols=cols,
                split_csv=split_csv,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                preprocess_kwargs=preprocess_kwargs,
            )

        else:
            roc_info = cox_kfold_roc_for_roc_curve(
                df=df2,
                duration=duration,
                event=event,
                feature_cols=cols,
                n_splits=n_splits,
                seed=seed,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                preprocess_kwargs=preprocess_kwargs,
            )

        results[label] = roc_info
        plot_cv_mean_roc(
            roc_info,
            label=label,
            ax=ax,
            show_fold_curves=False,
            shade=False,
        )

    delong_df = None

    if split_csv is not None:
        y = event.loc[df2.index].astype(int).to_numpy()
        delong_rows = []

        # Compare every model against Body composition
        ref_label = "Body composition"

        if ref_label in oof_risks:
            for label, risk in oof_risks.items():
                if label == ref_label:
                    continue

                p, z, auc_ref, auc_model, se_diff = delong_roc_test(
                    y,
                    oof_risks[ref_label],
                    risk,
                )

                delong_rows.append({
                    "reference_model": ref_label,
                    "comparison_model": label,
                    "auc_reference": auc_ref,
                    "auc_comparison": auc_model,
                    "delta_auc": auc_ref - auc_model,
                    "z": z,
                    "p_value": p,
                    "se_diff": se_diff,
                })

            delong_df = pd.DataFrame(delong_rows)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-validated ROC curves")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "roc": results,
        "oof_risks": oof_risks,
        "delong": delong_df,
    }

def subset_df_to_split_ids(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    split_csv: str,
    *,
    subject_id_col: str = "Subject ID",
    split_id_col: str = "index",
    outer_fold=None,
):
    """
    Restrict df/duration/event to the unique IDs present in split_csv.
    If outer_fold is provided, first filter split_csv to that outer_fold.
    """
    split_csv = Path(split_csv)

    df = df.copy()
    duration = duration.loc[df.index].copy()
    event = event.loc[df.index].copy()

    if subject_id_col not in df.columns:
        raise RuntimeError(f"'{subject_id_col}' not found in df columns.")

    df[subject_id_col] = df[subject_id_col].map(_norm_id)

    split_df = pd.read_csv(split_csv).copy()
    if outer_fold is not None:
        if "outer_fold" not in split_df.columns:
            raise RuntimeError(
                "outer_fold was provided, but split CSV has no 'outer_fold' column."
            )
        split_df = split_df.loc[split_df["outer_fold"] == outer_fold].copy()

    split_df[split_id_col] = split_df[split_id_col].map(_norm_id)

    split_ids = pd.Index(split_df[split_id_col].dropna().unique())
    keep_mask = df[subject_id_col].isin(split_ids)

    df_sub = df.loc[keep_mask].copy()
    duration_sub = duration.loc[df_sub.index].copy()
    event_sub = event.loc[df_sub.index].copy()

    missing_ids = sorted(set(split_ids) - set(df_sub[subject_id_col].dropna().tolist()))
    if missing_ids:
        raise RuntimeError(
            "Some split IDs were not found in df.\n"
            f"Examples: {missing_ids[:10]}"
        )

    return df_sub, duration_sub, event_sub

def visualize_ttrisk_groups(split_csv):

    df = pd.read_csv(split_csv)

    folds = sorted(df["fold"].unique())
    n_folds = len(folds)

    fig, axes = plt.subplots(1, n_folds, figsize=(18, 4), sharey=True, dpi=300)

    for fold in range(1, n_folds + 1):
        ax = axes[fold - 1]
        fold_df = df[df["fold"] == fold]
        rg = fold_df["risk_group"].values
        counts = np.bincount(rg)

        ax.hist(
            fold_df["risk_group"],
            bins=range(fold_df["risk_group"].max() + 2),
            edgecolor="black"
        )

        labels = [f"Group {i}: {c}" for i, c in enumerate(counts)]
        handles = [mpatches.Patch(color="none", label=lab) for lab in labels]

        ax.legend(handles=handles, loc="upper right")

        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Risk Group")

    axes[0].set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def _compute_midrank(x):
    """Midranks for DeLong."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        # ranks are 1-based
        mid = 0.5 * (i + j - 1) + 1
        T[i:j] = mid
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def _fast_delong(preds_sorted_transposed, label_1_count):
    """
    Fast DeLong implementation.
    preds_sorted_transposed: shape (k, n) where k = #models (2), n = total samples
    label_1_count: number of positives
    Returns: aucs (k,), covariance matrix (k,k)
    """
    m = int(label_1_count)
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]

    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]

    tx = np.vstack([_compute_midrank(pos[i]) for i in range(k)])
    ty = np.vstack([_compute_midrank(neg[i]) for i in range(k)])
    tz = np.vstack([_compute_midrank(preds_sorted_transposed[i]) for i in range(k)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01, bias=False)
    sy = np.cov(v10, bias=False)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov

def delong_roc_test(y_true, pred_a, pred_b):

    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(float)
    pred_b = np.asarray(pred_b).astype(float)

    assert y_true.shape == pred_a.shape == pred_b.shape

    # DeLong expects positives first after sorting by label
    order = np.argsort(-y_true)  # positives (1) first
    y = y_true[order]
    preds = np.vstack([pred_a[order], pred_b[order]])

    m = int(y.sum())

    aucs, cov = _fast_delong(preds, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se_diff = float(np.sqrt(max(var_diff, 1e-18)))
    z = (auc_a - auc_b) / se_diff
    p = 2 * stats.norm.sf(abs(z))
    return p, float(z), auc_a, auc_b, se_diff

def cox_kfold_oof_risk_from_split_csv(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    split_csv: str,
    subject_id_col: str = "Subject ID",
    split_id_col: str = "index",
    fold_col: str = "fold",
    split_col: str = "split",
    penalizer: float = 1.0,
    l1_ratio: float = 0.0,
    preprocess_kwargs=None,
):

    preprocess_kwargs = preprocess_kwargs or {}
    split_csv = Path(split_csv)

    df = df.copy()
    duration = duration.loc[df.index].copy()
    event = event.loc[df.index].astype(int).copy()

    if subject_id_col not in df.columns:
        raise RuntimeError(f"'{subject_id_col}' not found in df columns.")

    df[subject_id_col] = df[subject_id_col].map(_norm_id)

    split_df = pd.read_csv(split_csv)
    needed = {fold_col, split_col, split_id_col}
    missing_cols = needed - set(split_df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Split CSV is missing columns: {sorted(missing_cols)}. "
            f"Found: {list(split_df.columns)}"
        )

    split_df = split_df.copy()
    split_df[split_id_col] = split_df[split_id_col].map(_norm_id)
    split_df[split_col] = split_df[split_col].astype(str).str.strip().str.lower()

    dup_mask = df[subject_id_col].duplicated(keep=False)

    id_to_idx = pd.Series(df.index.values, index=df[subject_id_col].values)

    oof = pd.Series(np.nan, index=df.index, dtype=float)

    for fold in sorted(split_df[fold_col].dropna().unique()):
        fold_rows = split_df.loc[split_df[fold_col] == fold]

        train_ids = fold_rows.loc[fold_rows[split_col] == "train", split_id_col].dropna().tolist()
        test_ids  = fold_rows.loc[fold_rows[split_col] == "test",  split_id_col].dropna().tolist()

        if len(train_ids) == 0 or len(test_ids) == 0:
            raise RuntimeError(f"Fold {fold} has empty train or test IDs.")

        missing_train = sorted(set(train_ids) - set(id_to_idx.index))
        missing_test  = sorted(set(test_ids) - set(id_to_idx.index))
        if missing_train or missing_test:
            raise RuntimeError(
                f"Fold {fold} contains IDs not present in df.\n"
                f"Missing train IDs (first 10): {missing_train[:10]}\n"
                f"Missing test IDs (first 10): {missing_test[:10]}"
            )

        tr_idx = id_to_idx.loc[train_ids].to_numpy()
        te_idx = id_to_idx.loc[test_ids].to_numpy()

        Xtr_raw = df.loc[tr_idx, feature_cols]
        Xte_raw = df.loc[te_idx, feature_cols]

        pre = CoxPreprocessor(**preprocess_kwargs).fit(Xtr_raw)
        Xtr = pre.transform(Xtr_raw)
        Xte = pre.transform(Xte_raw)

        train_df = Xtr.copy()
        train_df["duration"] = duration.loc[tr_idx].values
        train_df["event"] = event.loc[tr_idx].values

        test_X = Xte.copy()

        cph = fit_cox(train_df, penalizer=penalizer, l1_ratio=l1_ratio)

        risk = cph.predict_partial_hazard(test_X).values.reshape(-1)
        oof.loc[te_idx] = risk

    if oof.isna().any():
        missing_n = int(oof.isna().sum())
        raise RuntimeError(f"OOF prediction contains {missing_n} missing rows. Check split CSV coverage.")

    return oof.loc[df.index].to_numpy(dtype=float)

# --- append to cox_utils.py ---

def save_cox_results(
    out_dir: str,
    cph: CoxPHFitter,
    model_df: pd.DataFrame,
    feature_cols_final,
    config: dict,
    cv_cidxs = None,
    cv_aucs = None
):
    """
    Save:
      - cox_summary.csv (lifelines summary)
      - cox_coef.csv (coef, HR, CI)
      - model_df.parquet (features + duration/event)   [optional big but useful]
      - risk_scores.csv (risk score per patient row)
      - config.json
      - cv_cindex.csv (if provided)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) model summary
    summary = cph.summary.copy()
    summary.to_csv(out / "cox_summary.csv", index=True)

    # 2) cleaner coef table
    coef = summary.copy()
    # lifelines already provides exp(coef), coef lower/upper 95%
    keep_cols = [c for c in [
        "coef", "exp(coef)", "se(coef)", "z", "p",
        "coef lower 95%", "coef upper 95%",
        "exp(coef) lower 95%", "exp(coef) upper 95%",
    ] if c in coef.columns]
    coef = coef[keep_cols].reset_index().rename(columns={"index": "feature"})
    coef.to_csv(out / "cox_coef.csv", index=False)

    # 3) risk scores (in-sample)
    X = model_df.drop(columns=["duration", "event"])
    risk = cph.predict_partial_hazard(X).values.reshape(-1)

    surv_180 = cph.predict_survival_function(X, times=[180]).T.iloc[:, 0].values
    risk_180 = 1.0 - surv_180

    risk_df = pd.DataFrame({
        "row_id": model_df.index,
        "risk_score": risk,
        "pred_surv_180": surv_180,
        "pred_risk_180": risk_180,
        "duration": model_df["duration"].values,
        "event": model_df["event"].values,
    })
    # tertiles for KM plots later
    rs = risk_df["risk_score"].astype(float)

    # drop NaNs
    valid = rs.dropna()
    nuniq = valid.nunique()

    # default
    risk_df["risk_tertile"] = np.nan

    if nuniq < 2:
        # impossible to split
        risk_df.loc[rs.notna(), "risk_tertile"] = "mid"  # or "all"
    else:
        # Try tertiles, but allow duplicate edges to be dropped
        try:
            bins = pd.qcut(valid, 3, duplicates="drop")
        except ValueError:
            bins = None

        if bins is not None and bins.cat.categories.size == 3:
            # 3 bins succeeded
            risk_df.loc[valid.index, "risk_tertile"] = pd.qcut(
                valid, 3, labels=["low", "mid", "high"], duplicates="drop"
            )
        else:
            # Fall back to 2 bins, still with duplicates="drop"
            bins2 = pd.qcut(valid, 2, duplicates="drop")
            k2 = bins2.cat.categories.size

            if k2 == 2:
                risk_df.loc[valid.index, "risk_tertile"] = pd.qcut(
                    valid, 2, labels=["low", "high"], duplicates="drop"
                )
            else:
                # still can't split due to ties
                risk_df.loc[valid.index, "risk_tertile"] = "mid"
    risk_df.to_csv(out / "risk_scores.csv", index=False)
    y_all = model_df["event"].astype(int).values
    auc_in_sample = np.nan
    if np.unique(y_all).size >= 2:
        auc_in_sample = float(roc_auc_score(y_all, risk))
   
    # 4) save model_df for reproducibility
    model_df.to_parquet(out / "cox_model_df.parquet")

    # 5) save config (incl. resolved features)
    cfg = dict(config)
    cfg["n_rows_model_df"] = int(len(model_df))
    cfg["auc_in_sample"] = auc_in_sample
    cfg["n_features_model_df"] = int(model_df.shape[1] - 2)
    cfg["concordance_index_in_sample"] = float(cph.concordance_index_)
    cfg["feature_cols_final"] = list(feature_cols_final)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 6) save CV results
    if cv_cidxs is not None:
        cv_df = pd.DataFrame({"fold_cindex": cv_cidxs})
        cv_df.loc["mean", "fold_cindex"] = cv_cidxs.mean()
        cv_df.loc["std", "fold_cindex"] = cv_cidxs.std(ddof=1) if len(cv_cidxs) > 1 else 0.0
        cv_df.to_csv(out / "cv_cindex.csv", index=True)
    if cv_aucs is not None:
        cv_auc_df = pd.DataFrame({"fold_auc": cv_aucs})
        cv_auc_df.loc["mean", "fold_auc"] = np.nanmean(cv_aucs)
        cv_auc_df.loc["std", "fold_auc"] = np.nanstd(cv_aucs, ddof=1) if np.isfinite(cv_aucs).sum() > 1 else 0.0
        cv_auc_df.to_csv(out / "cv_auc.csv", index=True)

    print("\n[Saved results to]", out.resolve())
    print(" - cox_summary.csv")
    print(" - cox_coef.csv")
    print(" - risk_scores.csv")
    print(" - cox_model_df.parquet")
    print(" - config.json")
    if cv_cidxs is not None:
        print(" - cv_cindex.csv")
    if cv_aucs is not None:
        print(" - cv_auc.csv")
