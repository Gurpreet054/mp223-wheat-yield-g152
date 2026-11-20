import numpy as np
import pandas as pd


def detect_data_issues(train_df: pd.DataFrame, later_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    issues = {}

    # 1. Leaking column (present in train, absent in holdout, not the target)
    extra_cols = set(train_df.columns) - set(holdout_df.columns) - {"yield_qtl_per_acre"}
    if extra_cols:
        leak_col = sorted(extra_cols)[0]
        issues["leaking_column"] = {"present": True, "column": leak_col}
    else:
        issues["leaking_column"] = {"present": False, "column": None}

    # 2. Unit mixup (Tehsil West fertilizer_kg is 1000x larger — grams instead of kg)
    grp_medians = train_df.groupby("tehsil")["fertilizer_kg"].median()
    ratio = grp_medians.max() / grp_medians.min()
    if ratio > 500:
        issues["unit_mixup"] = {
            "present": True,
            "column": "fertilizer_kg",
            "group_value": str(grp_medians.idxmax()),
            "factor": 1000,
        }
    else:
        issues["unit_mixup"] = {"present": False, "column": None, "group_value": None, "factor": None}

    # 3. Sentinel value (-999 in soil_ph)
    sentinel_count = int((train_df["soil_ph"] == -999).sum())
    if sentinel_count > 0:
        issues["sentinel_value"] = {
            "present": True,
            "column": "soil_ph",
            "value": -999,
            "count": sentinel_count,
        }
    else:
        issues["sentinel_value"] = {"present": False, "column": None, "value": None, "count": None}

    # 4. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    issues["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None,
    }

    # 5. Stuck sensor (longest consecutive identical run sorted by date)
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    num_cols = ["rainfall_mm", "fertilizer_kg", "sowing_delay_days", "irrigation_count", "soil_ph", "farm_acres"]
    stuck_found = False
    for col in num_cols:
        s = sorted_df[col]
        groups = (s != s.shift()).cumsum()
        run_lengths = s.groupby(groups).size()
        if run_lengths.max() >= 30:
            max_grp = run_lengths.idxmax()
            rows = sorted_df[groups == max_grp]
            issues["stuck_sensor"] = {
                "present": True,
                "column": col,
                "start_date": str(rows["date"].min()),
                "end_date": str(rows["date"].max()),
            }
            stuck_found = True
            break
    if not stuck_found:
        issues["stuck_sensor"] = {"present": False, "column": None, "start_date": None, "end_date": None}

    # 6. Target typos (10x too big)
    typo_count = int((train_df["yield_qtl_per_acre"] > 50).sum())
    issues["target_typos"] = {
        "present": typo_count > 0,
        "count": typo_count if typo_count > 0 else None,
    }

    # 7. Spelling variants
    spelling_issue = False
    for col in ["seed_variety", "tehsil"]:
        if set(train_df[col].unique()) != set(later_df[col].unique()):
            spelling_issue = True
            issues["spelling_variants"] = {
                "present": True,
                "column": col,
                "true_levels": int(later_df[col].nunique()),
            }
            break
    if not spelling_issue:
        issues["spelling_variants"] = {"present": False, "column": None, "true_levels": None}

    return issues


def clean_train_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "yield_from_mandi_receipt" in df.columns:
        df = df.drop(columns=["yield_from_mandi_receipt"])
    mask_west = (df["tehsil"] == "Tehsil West") & (df["fertilizer_kg"] > 500)
    df.loc[mask_west, "fertilizer_kg"] = df.loc[mask_west, "fertilizer_kg"] / 1000.0
    valid_ph_median = df.loc[df["soil_ph"] != -999, "soil_ph"].median()
    df.loc[df["soil_ph"] == -999, "soil_ph"] = valid_ph_median
    if "yield_qtl_per_acre" in df.columns:
        mask_typo = df["yield_qtl_per_acre"] > 50
        df.loc[mask_typo, "yield_qtl_per_acre"] = df.loc[mask_typo, "yield_qtl_per_acre"] / 10.0
    df = df.drop_duplicates().reset_index(drop=True)
    return df
