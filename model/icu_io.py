
import re
import numpy as np
import pandas as pd


DEFAULT_MISSING_TOKENS = {
    "", " ", "  ", "NA", "N/A", "na", "n/a", "NULL", "null", "None", "none",
    ".", "-", "--", "Unknown", "unknown", "#REF!"
}


def _norm_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    if s.lower().startswith("unnamed"):
        return None
    s = re.sub(r"\.0$", "", s)
    return s


def make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
    return out


def build_columns_from_header_rows(raw: pd.DataFrame, header_rows: int = 3, joiner: str = " | ") -> list:
    header_parts = [raw.iloc[i].tolist() for i in range(header_rows)]
    col_names = []
    ncol = raw.shape[1]
    for j in range(ncol):
        parts = []
        for i in range(header_rows):
            cell = _norm_cell(header_parts[i][j] if j < len(header_parts[i]) else None)
            if cell:
                parts.append(cell)
        name = joiner.join(parts) if parts else f"COL_{j+1}"
        col_names.append(name)
    return make_unique(col_names)


def resolve_first_match(columns, name: str):

    cols = list(columns)
    if name in cols:
        return name
    key = str(name).lower()
    for c in cols:
        if key in str(c).lower():
            return c
    return None


def _looks_like_int(x) -> bool:
    try:
        if pd.isna(x):
            return False
        s = str(x).strip()
        if s == "":
            return False
        _ = int(float(s))
        return True
    except Exception:
        return False


def find_first_patient_row_by_case(df: pd.DataFrame, case_col: str) -> int:
    for i, v in enumerate(df[case_col].tolist()):
        if _looks_like_int(v):
            return i
    raise RuntimeError(f"Cannot find first patient row using case column: {case_col}")


def coerce_datetime(s: pd.Series, date_format: str = "%m/%d/%Y") -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, format=date_format, errors="coerce")


def parse_binary_01(x):

    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1", "alive", "yes", "y", "true"}:
        return 1.0
    if s in {"0", "dead", "deceased", "no", "n", "false"}:
        return 0.0
    try:
        v = float(s)
        if v in (0.0, 1.0):
            return v
    except Exception:
        pass
    return np.nan


def load_metadata_xlsx(
    xlsx_path: str,
    sheet_name=0,
    header_rows: int = 3,
    joiner: str = " | ",
    missing_tokens=DEFAULT_MISSING_TOKENS,
) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    col_names = build_columns_from_header_rows(raw, header_rows=header_rows, joiner=joiner)

    df = raw.iloc[header_rows:].copy()
    df.columns = col_names
    df.reset_index(drop=True, inplace=True)

    # normalize missing tokens
    df.replace(to_replace=r"^\s*$", value=np.nan, regex=True, inplace=True)
    df.replace(list(missing_tokens), np.nan, inplace=True)
    # avoid pandas downcast warning / keep stable dtypes
    df = df.infer_objects(copy=False)

    return df


def trim_patient_rows(df: pd.DataFrame, case_col_name: str = "Case #"):
    case_col = resolve_first_match(df.columns, case_col_name)
    if case_col is None:
        raise RuntimeError(f"Cannot find case column using name: {case_col_name}")

    start_idx = find_first_patient_row_by_case(df, case_col)
    df2 = df.iloc[start_idx:].copy()
    df2 = df2[df2[case_col].apply(_looks_like_int)].copy()
    df2.reset_index(drop=True, inplace=True)
    return df2, case_col


def build_6m_survival_cohort(
    df: pd.DataFrame,
    admitted_col_name: str,
    death_col_name: str,
    surv6m_col_name: str,
    date_format: str = "%Y/%m/%d",
    censor_days: int = 180,
    save_filter_info = False,
    subject_id_label = "Subject ID"
):

    adm_col = resolve_first_match(df.columns, admitted_col_name)
    dod_col = resolve_first_match(df.columns, death_col_name)
    s6m_col = resolve_first_match(df.columns, surv6m_col_name)
    filter_stats = {}

    admitted = coerce_datetime(df[adm_col], date_format)
    death_date = coerce_datetime(df[dod_col], date_format)
    surv6m = df[s6m_col].apply(parse_binary_01)  # 1=alive, 0=dead, NaN=unknown

    # exclusions
    mask_known = surv6m.notna()                           # drop unknown survival
    mask_dead_missing_dod = (surv6m == 0) & (death_date.isna())
    mask_adm_ok = admitted.notna()

    keep = mask_known & (~mask_dead_missing_dod) & mask_adm_ok

    print("total rows:", len(df))
    print("drop unknown surv6m:", (~mask_known).sum())
    print("drop dead missing DoD:", mask_dead_missing_dod.sum())
    print("drop missing admitted:", (~mask_adm_ok).sum())
    print("kept after first filter:", keep.sum())

    if save_filter_info:
        filter_stats = {
            "Missing 6mo Survival": df.loc[~mask_known, subject_id_label].tolist(),
            "Dead Missing DoD": df.loc[mask_dead_missing_dod, subject_id_label].tolist(),
            "Missing Admitted Date": df.loc[~mask_adm_ok, subject_id_label].tolist(),
        }

    df_kept = df.loc[keep].copy()

    admitted2 = admitted.loc[keep]
    death2 = death_date.loc[keep]
    surv2 = surv6m.loc[keep]

    event = (surv2 == 0).astype(int)  # dead=1
    duration = pd.Series(np.nan, index=df_kept.index, dtype=float)

    dead_idx = df_kept.index[event == 1]
    alive_idx = df_kept.index[event == 0]

    duration.loc[dead_idx] = (death2.loc[dead_idx] - admitted2.loc[dead_idx]).dt.total_seconds() / (3600 * 24)
    duration.loc[alive_idx] = float(censor_days)

    # guards
    bad = duration.isna() | (duration <= 0)
    bad = bad | ((event == 1) & (duration > (censor_days + 1)))  # should not exceed 180d if label correct

    print("bad duration isna:", duration.isna().sum())
    print("bad duration <= 0:", (duration <= 0).sum())
    print("bad dead > censor:", ((event == 1) & (duration > (censor_days + 1))).sum())
    print("total bad:", bad.sum())

    if save_filter_info:
        filter_stats["Duration NaN"] = df_kept.loc[duration.isna(), subject_id_label].tolist()
        filter_stats["Duration <= 0"] = df_kept.loc[duration <= 0, subject_id_label].tolist()
        filter_stats["Dead > Censor"] = df_kept.loc[((event == 1) & (duration > (censor_days + 1))), subject_id_label].tolist()

    df_kept = df_kept.loc[~bad].copy()
    duration = duration.loc[df_kept.index].copy()
    event = event.loc[df_kept.index].copy()

    info = {
        "admitted_col": adm_col,
        "death_col": dod_col,
        "surv6m_col": s6m_col,
        "n": int(len(df_kept)),
        "events": int(event.sum()),
        "censored": int((1 - event).sum()),
    }
    return df_kept, duration, event, info, filter_stats
