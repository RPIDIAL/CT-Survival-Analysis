from importlib import import_module
from pathlib import Path
import sys

from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from cox_utils import CoxPreprocessor, make_strata


def _import_real_xgboost():
    """
    Import the actual third-party xgboost package even though this repo also has
    a local directory named `xgboost/`.
    """
    module = import_module("xgboost")
    if hasattr(module, "DMatrix"):
        return module

    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    removed = []

    for bad_path in ("", str(current_dir), str(repo_root)):
        while bad_path in sys.path:
            sys.path.remove(bad_path)
            removed.append(bad_path)

    sys.modules.pop("xgboost", None)
    try:
        module = import_module("xgboost")
    finally:
        for path in reversed(removed):
            sys.path.insert(0, path)

    if not hasattr(module, "DMatrix"):
        raise ImportError(
            "Imported `xgboost`, but it does not look like the third-party "
            "package. Check your Python environment and import path."
        )
    return module


xgb = _import_real_xgboost()


def _make_survival_labels(duration: pd.Series, event: pd.Series) -> np.ndarray:
    """
    For XGBoost survival:cox, censored samples are encoded as negative times.
    """
    y = duration.astype(float).to_numpy(copy=True)
    y[event.astype(int).to_numpy() == 0] *= -1.0
    return y


def _with_default_xgb_params(params: dict | None) -> dict:
    params = dict(params or {})
    params.setdefault("objective", "survival:cox")
    params.setdefault("eval_metric", "cox-nloglik")
    params.setdefault("eta", 0.05)
    params.setdefault("max_depth", 3)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", 0.8)
    params.setdefault("min_child_weight", 1.0)
    params.setdefault("reg_lambda", 1.0)
    params.setdefault("reg_alpha", 0.0)
    params.setdefault("tree_method", "hist")
    return params


def _balanced_outer_folds(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    *,
    n_splits: int,
    seed: int,
    risk_groups: int = 3,
):
    strata = make_strata(duration.loc[df.index], event.loc[df.index], risk_groups=risk_groups)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(skf.split(df, strata.to_numpy(dtype=np.int64)), 1):
        yield fold, df.index[tr], df.index[te]


def _balanced_train_valid_split(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    *,
    seed: int,
    risk_groups: int = 3,
    n_splits: int = 5,
):
    if len(df) < n_splits:
        raise ValueError(f"Need at least {n_splits} training rows for inner validation split.")

    strata = make_strata(duration.loc[df.index], event.loc[df.index], risk_groups=risk_groups)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_pos, va_pos = next(skf.split(df, strata.to_numpy(dtype=np.int64)))
    return df.index[tr_pos], df.index[va_pos]


def _numeric_feature_view(X: pd.DataFrame) -> pd.DataFrame:

    return X.apply(pd.to_numeric, errors="coerce")


def _rank_features_by_univariate_cindex(
    X_train: pd.DataFrame,
    duration_train: pd.Series,
    event_train: pd.Series,
    feature_cols,
    *,
    min_non_missing: float = 0.50,
    min_unique: int = 5,
):
    X_num = _numeric_feature_view(X_train.loc[:, feature_cols])
    scores = {}
    kept_numeric = {}

    for col in feature_cols:
        s = X_num[col]
        non_missing = float(s.notna().mean())
        if non_missing < min_non_missing:
            continue

        s = s.replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            continue

        s = s.fillna(s.median())
        if s.nunique(dropna=False) < min_unique:
            continue

        ci = concordance_index(
            duration_train.values,
            -s.to_numpy(dtype=float),
            event_train.values,
        )
        scores[col] = abs(float(ci) - 0.5)
        kept_numeric[col] = s.to_numpy(dtype=float)

    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
    return ranked, scores, kept_numeric


def _prune_correlated_features(
    ranked_cols,
    numeric_values: dict,
    *,
    corr_threshold: float,
    top_k: int,
):
    selected = []

    for col in ranked_cols:
        x = numeric_values[col]
        too_similar = False

        for kept in selected:
            y = numeric_values[kept]
            corr = np.corrcoef(x, y)[0, 1]
            if np.isfinite(corr) and abs(corr) >= corr_threshold:
                too_similar = True
                break

        if not too_similar:
            selected.append(col)

        if len(selected) >= top_k:
            break

    return selected


def select_xgb_features_for_fold(
    X_train: pd.DataFrame,
    duration_train: pd.Series,
    event_train: pd.Series,
    feature_cols,
    *,
    enabled: bool = True,
    top_k: int = 50,
    corr_threshold: float = 0.95,
    min_non_missing: float = 0.50,
    min_unique: int = 5,
    always_keep=None,
):

    feature_cols = list(dict.fromkeys(feature_cols))
    always_keep = [c for c in (always_keep or []) if c in feature_cols]

    if not enabled or len(feature_cols) <= top_k:
        return feature_cols, {}

    ranked, scores, numeric_values = _rank_features_by_univariate_cindex(
        X_train,
        duration_train,
        event_train,
        feature_cols,
        min_non_missing=min_non_missing,
        min_unique=min_unique,
    )

    if not ranked:
        return feature_cols, {}

    filtered_ranked = [c for c in ranked if c not in always_keep]
    selected = list(always_keep)
    selected.extend(
        _prune_correlated_features(
            filtered_ranked,
            numeric_values,
            corr_threshold=corr_threshold,
            top_k=max(top_k - len(selected), 0),
        )
    )

    if not selected:
        selected = ranked[:top_k]

    return selected, scores

def fit_xgb_survival_cox(
    X_train: pd.DataFrame,
    duration_train: pd.Series,
    event_train: pd.Series,
    *,
    params: dict = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    X_valid: pd.DataFrame = None,
    duration_valid: pd.Series = None,
    event_valid: pd.Series = None,
    verbose_eval: bool = False,
):

    params = _with_default_xgb_params(params)

    dtrain = xgb.DMatrix(
        X_train,
        label=_make_survival_labels(duration_train, event_train),
    )

    evals = [(dtrain, "train")]
    if X_valid is not None:
        dvalid = xgb.DMatrix(
            X_valid,
            label=_make_survival_labels(duration_valid, event_valid),
        )
        evals.append((dvalid, "valid"))
    else:
        dvalid = None

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
        verbose_eval=verbose_eval,
    )

    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        best_iter = num_boost_round - 1

    return booster, best_iter


def fit_xgb_survival_cox_full_data(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    *,
    preprocess_kwargs=None,
    xgb_params=None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    seed: int = 42,
    risk_groups: int = 3,
    feature_filter_kwargs=None,
):

    preprocess_kwargs = preprocess_kwargs or {}
    xgb_params = _with_default_xgb_params(xgb_params)
    feature_filter_kwargs = feature_filter_kwargs or {}

    df = df.copy()
    duration = duration.loc[df.index].astype(float)
    event = event.loc[df.index].astype(int)
    feature_cols = list(dict.fromkeys(feature_cols))

    selected_feature_cols, filter_scores = select_xgb_features_for_fold(
        df.loc[:, feature_cols],
        duration,
        event,
        feature_cols,
        **feature_filter_kwargs,
    )

    tr_idx, va_idx = _balanced_train_valid_split(
        df,
        duration,
        event,
        seed=seed,
        risk_groups=risk_groups,
    )

    pre_es = CoxPreprocessor(**preprocess_kwargs).fit(df.loc[tr_idx, selected_feature_cols])
    Xtr = pre_es.transform(df.loc[tr_idx, selected_feature_cols])
    Xva = pre_es.transform(df.loc[va_idx, selected_feature_cols])

    _, best_iter = fit_xgb_survival_cox(
        Xtr,
        duration.loc[tr_idx],
        event.loc[tr_idx],
        X_valid=Xva,
        duration_valid=duration.loc[va_idx],
        event_valid=event.loc[va_idx],
        params=xgb_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    final_rounds = max(int(best_iter) + 1, 1)
    pre_full = CoxPreprocessor(**preprocess_kwargs).fit(df.loc[:, selected_feature_cols])
    X_full = pre_full.transform(df.loc[:, selected_feature_cols])

    dfull = xgb.DMatrix(X_full, label=_make_survival_labels(duration, event))
    booster = xgb.train(
        params=xgb_params,
        dtrain=dfull,
        num_boost_round=final_rounds,
        evals=[(dfull, "train")],
        verbose_eval=False,
    )
    booster.set_attr(best_iteration=str(final_rounds - 1))

    risk = booster.predict(xgb.DMatrix(X_full))
    return {
        "booster": booster,
        "preprocessor": pre_full,
        "selected_feature_cols_raw": selected_feature_cols,
        "X_full": X_full,
        "risk": risk,
        "best_iteration": final_rounds - 1,
        "filter_scores": filter_scores,
    }


def compute_xgb_feature_contributions(
    booster,
    X: pd.DataFrame,
):
    contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    feature_names = list(X.columns)
    contrib_df = pd.DataFrame(
        contrib[:, :-1],
        columns=feature_names,
        index=X.index,
    )
    bias = pd.Series(contrib[:, -1], index=X.index, name="bias")

    mean_abs = contrib_df.abs().mean().sort_values(ascending=False)
    mean_signed = contrib_df.mean().reindex(mean_abs.index)

    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    gain_s = pd.Series(gain, dtype=float).reindex(feature_names).fillna(0.0)
    weight_s = pd.Series(weight, dtype=float).reindex(feature_names).fillna(0.0)

    summary = pd.DataFrame({
        "feature": mean_abs.index,
        "mean_abs_contribution": mean_abs.values,
        "mean_signed_contribution": mean_signed.values,
        "gain": gain_s.reindex(mean_abs.index).values,
        "split_count": weight_s.reindex(mean_abs.index).values,
    })
    return contrib_df, bias, summary


def save_xgb_feature_contributions(
    out_dir: str,
    booster,
    X: pd.DataFrame,
    *,
    top_n: int = 30,
    row_metadata: pd.DataFrame | None = None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    contrib_df, bias, summary = compute_xgb_feature_contributions(booster, X)

    summary.to_csv(out / "xgb_feature_contribution_summary.csv", index=False)

    per_patient = contrib_df.copy()
    per_patient["bias"] = bias.values
    if row_metadata is not None:
        per_patient = pd.concat(
            [row_metadata.reset_index(drop=True), per_patient.reset_index(drop=True)],
            axis=1,
        )
    per_patient.to_parquet(out / "xgb_feature_contributions_per_patient.parquet", index=False)

    top_summary = summary.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, 0.3 * len(top_summary))))
    plt.barh(top_summary["feature"], top_summary["mean_abs_contribution"])
    plt.xlabel("Mean |contribution| to XGBoost risk score")
    plt.title(f"Top {min(top_n, len(summary))} XGBoost feature contributions")
    plt.tight_layout()
    plt.savefig(out / "xgb_feature_contribution_top.png", dpi=200)
    plt.close()

    print("\n[Saved XGBoost contribution artifacts to]", out.resolve())
    print(" - xgb_feature_contribution_summary.csv")
    print(" - xgb_feature_contributions_per_patient.parquet")
    print(" - xgb_feature_contribution_top.png")

def xgb_kfold_cindex(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    *,
    n_splits: int = 5,
    seed: int = 42,
    preprocess_kwargs=None,
    xgb_params=None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    risk_groups: int = 3,
    feature_filter_kwargs=None,
):
    preprocess_kwargs = preprocess_kwargs or {}
    xgb_params = xgb_params or {}
    feature_filter_kwargs = feature_filter_kwargs or {}
    cidxs = []

    df = df.copy()
    duration = duration.loc[df.index].astype(float)
    event = event.loc[df.index].astype(int)

    for fold, tr_idx, te_idx in _balanced_outer_folds(
        df,
        duration,
        event,
        n_splits=n_splits,
        seed=seed,
        risk_groups=risk_groups,
    ):
        fold_feature_cols, fold_scores = select_xgb_features_for_fold(
            df.loc[tr_idx, feature_cols],
            duration.loc[tr_idx],
            event.loc[tr_idx],
            feature_cols,
            **feature_filter_kwargs,
        )

        Xte_raw = df.loc[te_idx, fold_feature_cols]

        tr_sub, va_sub = _balanced_train_valid_split(
            df.loc[tr_idx],
            duration.loc[tr_idx],
            event.loc[tr_idx],
            seed=seed + fold,
            risk_groups=risk_groups,
        )

        # Fit preprocessing on the actual fitting subset only to avoid
        # validation leakage through imputation/scaling statistics.
        pre = CoxPreprocessor(**preprocess_kwargs).fit(df.loc[tr_sub, fold_feature_cols])
        Xtr2 = pre.transform(df.loc[tr_sub, fold_feature_cols])
        Xva = pre.transform(df.loc[va_sub, fold_feature_cols])
        Xte = pre.transform(Xte_raw)

        if fold_scores:
            top_preview = ", ".join(fold_feature_cols[:5])
            print(
                f"Fold {fold}: selected {len(fold_feature_cols)}/{len(feature_cols)} "
                f"raw features for XGBoost. Top examples: {top_preview}"
            )

        booster, best_iter = fit_xgb_survival_cox(
            Xtr2, duration.loc[tr_sub], event.loc[tr_sub],
            X_valid=Xva, duration_valid=duration.loc[va_sub], event_valid=event.loc[va_sub],
            params=xgb_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        risk = booster.predict(xgb.DMatrix(Xte), iteration_range=(0, best_iter + 1))
        ci = concordance_index(duration.loc[te_idx].values, -risk, event.loc[te_idx].values)
        cidxs.append(ci)
        print(f"Fold {fold}: C-index = {ci:.4f}")

    cidxs = np.asarray(cidxs, dtype=float)
    print(f"\n{n_splits}-fold mean C-index = {cidxs.mean():.4f} ± {cidxs.std(ddof=1):.4f}")
    return cidxs

def xgb_kfold_auc(
    df: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    feature_cols,
    *,
    n_splits: int = 5,
    seed: int = 42,
    preprocess_kwargs=None,
    xgb_params=None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    risk_groups: int = 3,
    feature_filter_kwargs=None,
):

    preprocess_kwargs = preprocess_kwargs or {}
    xgb_params = xgb_params or {}
    feature_filter_kwargs = feature_filter_kwargs or {}
    aucs = []

    df = df.copy()
    duration = duration.loc[df.index].astype(float)
    event = event.loc[df.index].astype(int)

    for fold, tr_idx, te_idx in _balanced_outer_folds(
        df,
        duration,
        event,
        n_splits=n_splits,
        seed=seed,
        risk_groups=risk_groups,
    ):
        fold_feature_cols, _ = select_xgb_features_for_fold(
            df.loc[tr_idx, feature_cols],
            duration.loc[tr_idx],
            event.loc[tr_idx],
            feature_cols,
            **feature_filter_kwargs,
        )

        Xte_raw = df.loc[te_idx, fold_feature_cols]

        tr_sub, va_sub = _balanced_train_valid_split(
            df.loc[tr_idx],
            duration.loc[tr_idx],
            event.loc[tr_idx],
            seed=seed + fold,
            risk_groups=risk_groups,
        )

        pre = CoxPreprocessor(**preprocess_kwargs).fit(df.loc[tr_sub, fold_feature_cols])
        Xtr2 = pre.transform(df.loc[tr_sub, fold_feature_cols])
        Xva = pre.transform(df.loc[va_sub, fold_feature_cols])
        Xte = pre.transform(Xte_raw)

        booster, best_iter = fit_xgb_survival_cox(
            Xtr2, duration.loc[tr_sub], event.loc[tr_sub],
            X_valid=Xva, duration_valid=duration.loc[va_sub], event_valid=event.loc[va_sub],
            params=xgb_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        risk = booster.predict(xgb.DMatrix(Xte), iteration_range=(0, best_iter + 1))
        y = event.loc[te_idx].values.astype(int)

        if np.unique(y).size < 2:
            print(f"Fold {fold}: Event AUC = NaN (only one class in test fold)")
            aucs.append(np.nan)
            continue

        auc = roc_auc_score(y, risk)
        aucs.append(auc)
        print(f"Fold {fold}: Event AUC = {auc:.4f}")

    aucs = np.asarray(aucs, dtype=float)
    valid = np.isfinite(aucs)
    if valid.any():
        print(
            f"\n{n_splits}-fold mean Event AUC = "
            f"{np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}"
        )
    else:
        print("\nAll folds Event AUC are NaN.")
    return aucs
