"""
preprocess.py

Clinical preprocessing for SEER-like breast cancer registry data.

Input:
    - export.csv or export_demo.csv

Output:
    - preprocessed_breast_cancer.csv

Main principles:
    1. Read all raw columns as string to preserve medical codes such as '0014', '027', '00'.
    2. Do not treat all Unknown / Blank / No-Unknown values the same way.
    3. Merge variables that describe the same clinical concept but use different coding eras:
       - grade_2018 + grade_thru_2017 -> grade_unified
       - surgery_1998_2022 + surgery_2023 -> surgery_code_unified
    4. Do not use Vital status or Survival months.
       They are removed from the processed output.
    5. The target variable is:
       - survive_after_5: 1 = survive after 5 years, 0 = not survive after 5 years
    6. This script produces a cleaned clinical feature table, not one-hot encoded final model matrix.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CANDIDATES = [
    Path("/Users/minhdt/Desktop/ML Breast/Preprocess/final_demo.csv"),
]

OUTPUT_PATH = Path("preprocessed_breast_cancer.csv")


RAW_TO_CLEAN_COLUMNS = {
    "Year of diagnosis": "diagnosis_year_raw",
    "Sex": "sex_raw",
    "Age recode with <1 year olds and 90+": "age_group_raw",
    "Race recode (W, B, AI, API)": "race_raw",
    "Derived Summary Grade 2018 (2018+)": "grade_2018_raw",
    "Grade Recode (thru 2017)": "grade_thru_2017_raw",
    "Laterality": "laterality_raw",
    "Combined Summary Stage with Expanded Regional Codes (2004+)": "summary_stage_raw",
    "Tumor Size Over Time Recode (1988+)": "tumor_size_raw",
    "Regional nodes positive (1988+)": "nodes_positive_raw",
    "Regional nodes examined (1988+)": "nodes_examined_raw",
    "ER Status Recode Breast Cancer (1990+)": "er_status_raw",
    "PR Status Recode Breast Cancer (1990+)": "pr_status_raw",
    "Derived HER2 Recode (2010+)": "her2_status_raw",
    "RX Summ--Surg Prim Site (1998-2022)": "surgery_1998_2022_raw",
    "RX Summ--Surg Prim Site 2023 (2023+)": "surgery_2023_raw",
    "Chemotherapy recode (yes, no/unk)": "chemotherapy_raw",
    "Radiation recode": "radiation_raw",
    "survive_after_5": "survive_after_5_raw",
}


UNKNOWN_LIKE_VALUES = {
    "",
    "Blank(s)",
    "Unknown",
    "Borderline/Unknown",
    "No/Unknown",
    "None/Unknown",
    "Recode not available",
    "Unknown/unstaged/unspecified/DCO",
    "Unknown or size unreasonable (includes any tumor sizes 401-989)",
    "Recommended, unknown if administered",
}


# ============================================================
# UTILITY
# ============================================================

def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    candidates = ", ".join(str(p) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(f"Cannot find input file. Expected one of: {candidates}")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load CSV as string to preserve registry codes."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in RAW_TO_CLEAN_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing expected columns:\n" + "\n".join(f"- {col}" for col in missing)
        )

    return df.rename(columns=RAW_TO_CLEAN_COLUMNS)


def is_blank_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "Blank(s)"}


# ============================================================
# ATTRIBUTE-SPECIFIC PREPROCESSING
# ============================================================

def preprocess_year(s: pd.Series) -> pd.DataFrame:
    year = pd.to_numeric(s, errors="coerce").astype("Int64")

    era = pd.Series("unknown", index=s.index, dtype="object")
    era = era.mask(year < 2010, "pre_2010")
    era = era.mask((year >= 2010) & (year <= 2017), "2010_2017")
    era = era.mask((year >= 2018) & (year <= 2022), "2018_2022")
    era = era.mask(year >= 2023, "2023_plus")

    return pd.DataFrame({
        "diagnosis_year": year,
        "diagnosis_era": era,
    })


def parse_age_midpoint(value: Any) -> float:
    value = str(value).strip()

    if value == "90+ years":
        return 92.0

    match = re.search(r"(\d+)-(\d+)", value)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2

    if value == "00 years":
        return 0.0

    return np.nan


def preprocess_age(s: pd.Series) -> pd.DataFrame:
    age_midpoint = s.apply(parse_age_midpoint)

    return pd.DataFrame({
        "age_group": s,
        "age_midpoint": age_midpoint,
        "age_90_plus_flag": (s == "90+ years").astype(int),
        "age_unknown_flag": age_midpoint.isna().astype(int),
    })


def preprocess_race(s: pd.Series) -> pd.DataFrame:
    race = s.replace({"": "Unknown", "Blank(s)": "Unknown"})

    return pd.DataFrame({
        "race": race,
        "race_unknown_flag": (race == "Unknown").astype(int),
    })


def grade_from_text(value: Any) -> tuple[float, str]:
    text = str(value).strip()

    if text in {"", "Blank(s)", "Unknown", "Grade cannot be assessed; Unknown"}:
        return np.nan, "unknown"

    grade_map = {
        "Site-specific grade system category (1)": (1, "grade_1_well_or_low"),
        "Site-specific grade system category (2)": (2, "grade_2_moderate"),
        "Site-specific grade system category (3)": (3, "grade_3_poor_or_high"),
        "Site-specific grade system category (4)": (4, "grade_4_undifferentiated"),
        "Well differentiated; Grade I": (1, "grade_1_well_or_low"),
        "Moderately differentiated; Grade II": (2, "grade_2_moderate"),
        "Poorly differentiated; Grade III": (3, "grade_3_poor_or_high"),
        "Undifferentiated; anaplastic; Grade IV": (4, "grade_4_undifferentiated"),
        "Well differentiated": (1, "grade_1_well_or_low"),
        "Low grade": (1, "grade_1_well_or_low"),
        "Moderately differentiated": (2, "grade_2_moderate"),
        "Poorly differentiated": (3, "grade_3_poor_or_high"),
        "High grade": (3, "grade_3_poor_or_high"),
        "Undifferentiated and anaplastic": (4, "grade_4_undifferentiated"),
    }

    if text in grade_map:
        num, group = grade_map[text]
        return float(num), group

    return np.nan, "other_or_uncertain"


def preprocess_grade(df: pd.DataFrame) -> pd.DataFrame:
    g2018 = df["grade_2018_raw"]
    gold = df["grade_thru_2017_raw"]

    unified_text = []
    source = []

    for new_val, old_val in zip(g2018, gold):
        if not is_blank_like(new_val):
            unified_text.append(new_val)
            source.append("grade_2018_plus")
        elif not is_blank_like(old_val):
            unified_text.append(old_val)
            source.append("grade_thru_2017")
        else:
            unified_text.append("Unknown")
            source.append("unknown")

    parsed = [grade_from_text(x) for x in unified_text]
    grade_num = [p[0] for p in parsed]
    grade_group = [p[1] for p in parsed]

    return pd.DataFrame({
        "grade_unified_raw": unified_text,
        "grade_source": source,
        "grade_unified_num": grade_num,
        "grade_group": grade_group,
        "grade_unknown_flag": pd.isna(pd.Series(grade_num)).astype(int),
    }, index=df.index)


def preprocess_laterality(s: pd.Series) -> pd.DataFrame:
    mapping = {
        "Left - origin of primary": "left",
        "Right - origin of primary": "right",
        "Paired site, but no information concerning laterality": "paired_site_laterality_unknown",
        "Only one side - side unspecified": "one_side_unspecified",
        "Bilateral, single primary": "bilateral_single_primary",
    }

    laterality = s.map(mapping).fillna("unknown")

    return pd.DataFrame({
        "laterality": laterality,
        "laterality_unknown_flag": laterality.isin([
            "paired_site_laterality_unknown",
            "one_side_unspecified",
            "unknown",
        ]).astype(int),
    })


def stage_features(value: Any) -> dict[str, Any]:
    text = str(value).strip()

    base = {
        "stage_group": "unknown",
        "stage_ordinal": np.nan,
        "stage_in_situ_flag": 0,
        "stage_localized_flag": 0,
        "stage_regional_flag": 0,
        "stage_distant_flag": 0,
        "stage_unknown_flag": 0,
        "regional_direct_extension_flag": 0,
        "regional_lymph_node_involved_flag": 0,
    }

    if text in {"", "Blank(s)", "Unknown/unstaged/unspecified/DCO"}:
        base["stage_unknown_flag"] = 1
        return base

    if text == "In situ":
        base.update({
            "stage_group": "in_situ",
            "stage_ordinal": 0,
            "stage_in_situ_flag": 1,
        })
    elif text == "Localized only":
        base.update({
            "stage_group": "localized",
            "stage_ordinal": 1,
            "stage_localized_flag": 1,
        })
    elif text == "Regional by direct extension only":
        base.update({
            "stage_group": "regional",
            "stage_ordinal": 2,
            "stage_regional_flag": 1,
            "regional_direct_extension_flag": 1,
        })
    elif text == "Regional lymph nodes involved only":
        base.update({
            "stage_group": "regional",
            "stage_ordinal": 2,
            "stage_regional_flag": 1,
            "regional_lymph_node_involved_flag": 1,
        })
    elif text == "Regional by both direct extension and lymph node involvement":
        base.update({
            "stage_group": "regional",
            "stage_ordinal": 2,
            "stage_regional_flag": 1,
            "regional_direct_extension_flag": 1,
            "regional_lymph_node_involved_flag": 1,
        })
    elif text == "Distant site(s)/node(s) involved":
        base.update({
            "stage_group": "distant",
            "stage_ordinal": 3,
            "stage_distant_flag": 1,
        })
    else:
        base["stage_unknown_flag"] = 1

    return base


def preprocess_stage(s: pd.Series) -> pd.DataFrame:
    return s.apply(stage_features).apply(pd.Series)


def tumor_size_features(value: Any) -> dict[str, Any]:
    text = str(value).strip()

    result = {
        "tumor_size_mm": np.nan,
        "tumor_size_group": "unknown",
        "tumor_size_unknown_flag": 0,
        "tumor_size_special_flag": 0,
        "tumor_size_microscopic_focus_flag": 0,
        "tumor_size_no_primary_evidence_flag": 0,
    }

    if text in {"", "Blank(s)"} or "Unknown or size unreasonable" in text:
        result["tumor_size_unknown_flag"] = 1
        return result

    if text == "Tumor Size Not Consistent Over Time or Not Applicable for this Site":
        result["tumor_size_group"] = "not_consistent_or_not_applicable"
        result["tumor_size_special_flag"] = 1
        return result

    if text.startswith("990"):
        result["tumor_size_group"] = "microscopic_focus"
        result["tumor_size_special_flag"] = 1
        result["tumor_size_microscopic_focus_flag"] = 1
        return result

    if text.startswith("998"):
        result["tumor_size_group"] = "site_specific_code"
        result["tumor_size_special_flag"] = 1
        return result

    if text.startswith("000"):
        result["tumor_size_mm"] = 0.0
        result["tumor_size_group"] = "no_evidence_of_primary_tumor"
        result["tumor_size_no_primary_evidence_flag"] = 1
        return result

    if text.isdigit():
        size = int(text)
        if 1 <= size <= 400:
            result["tumor_size_mm"] = float(size)

            if size <= 10:
                group = "01_10_mm"
            elif size <= 20:
                group = "11_20_mm"
            elif size <= 50:
                group = "21_50_mm"
            else:
                group = "gt_50_mm"

            result["tumor_size_group"] = group
            return result

    result["tumor_size_unknown_flag"] = 1
    return result


def preprocess_tumor_size(s: pd.Series) -> pd.DataFrame:
    out = s.apply(tumor_size_features).apply(pd.Series)
    out["log1p_tumor_size_mm"] = np.log1p(out["tumor_size_mm"])
    return out


def parse_node_code(value: Any, column_type: str) -> dict[str, Any]:
    text = str(value).strip()

    result = {
        f"{column_type}_count": np.nan,
        f"{column_type}_special_code": "none",
        f"{column_type}_unknown_flag": 0,
    }

    if not text.isdigit():
        result[f"{column_type}_special_code"] = "non_numeric_or_blank"
        result[f"{column_type}_unknown_flag"] = 1
        return result

    code = int(text)

    if 0 <= code <= 90:
        result[f"{column_type}_count"] = float(code)
        return result

    if 95 <= code <= 99:
        result[f"{column_type}_special_code"] = text
        result[f"{column_type}_unknown_flag"] = 1
        return result

    result[f"{column_type}_special_code"] = "other_out_of_range"
    result[f"{column_type}_unknown_flag"] = 1
    return result


def preprocess_nodes(df: pd.DataFrame) -> pd.DataFrame:
    pos = (
        df["nodes_positive_raw"]
        .apply(lambda x: parse_node_code(x, "nodes_positive"))
        .apply(pd.Series)
    )

    examined = (
        df["nodes_examined_raw"]
        .apply(lambda x: parse_node_code(x, "nodes_examined"))
        .apply(pd.Series)
    )

    out = pd.concat([pos, examined], axis=1)

    out["nodes_positive_any_flag"] = (out["nodes_positive_count"] > 0).astype(int)
    out["nodes_examined_any_flag"] = (out["nodes_examined_count"] > 0).astype(int)

    out["nodes_positive_ratio"] = np.where(
        out["nodes_examined_count"] > 0,
        out["nodes_positive_count"] / out["nodes_examined_count"],
        np.nan,
    )

    out["nodes_ratio_unknown_flag"] = out["nodes_positive_ratio"].isna().astype(int)

    return out


def receptor_features(s: pd.Series, prefix: str) -> pd.DataFrame:
    clean = s.replace({
        "": "Unknown",
        "Blank(s)": "Unknown",
        "Borderline/Unknown": "Unknown",
        "Recode not available": "Not available",
    })

    return pd.DataFrame({
        f"{prefix}_status": clean,
        f"{prefix}_positive_flag": (clean == "Positive").astype(int),
        f"{prefix}_negative_flag": (clean == "Negative").astype(int),
        f"{prefix}_unknown_flag": (clean == "Unknown").astype(int),
        f"{prefix}_not_available_flag": (clean == "Not available").astype(int),
    })


def preprocess_receptors(df: pd.DataFrame) -> pd.DataFrame:
    er = receptor_features(df["er_status_raw"], "er")
    pr = receptor_features(df["pr_status_raw"], "pr")
    her2 = receptor_features(df["her2_status_raw"], "her2")

    out = pd.concat([er, pr, her2], axis=1)

    out["hr_positive_flag"] = (
        (out["er_positive_flag"] == 1) | (out["pr_positive_flag"] == 1)
    ).astype(int)

    out["triple_negative_flag"] = (
        (out["er_negative_flag"] == 1)
        & (out["pr_negative_flag"] == 1)
        & (out["her2_negative_flag"] == 1)
    ).astype(int)

    out["receptor_unknown_or_unavailable_count"] = (
        out["er_unknown_flag"]
        + out["er_not_available_flag"]
        + out["pr_unknown_flag"]
        + out["pr_not_available_flag"]
        + out["her2_unknown_flag"]
        + out["her2_not_available_flag"]
    )

    return out


def preprocess_surgery(df: pd.DataFrame) -> pd.DataFrame:
    year = pd.to_numeric(df["diagnosis_year_raw"], errors="coerce")
    old_code = df["surgery_1998_2022_raw"]
    new_code = df["surgery_2023_raw"]

    unified_code = []
    source = []

    for y, old, new in zip(year, old_code, new_code):
        if pd.notna(y) and int(y) >= 2023:
            unified_code.append(new if not is_blank_like(new) else "Unknown")
            source.append("surgery_2023_plus")
        else:
            unified_code.append(old if not is_blank_like(old) else "Unknown")
            source.append("surgery_1998_2022")

    code_series = pd.Series(unified_code, index=df.index, dtype="object")

    no_surgery_codes = {"00", "A000"}
    unknown_codes = {"", "Blank(s)", "Unknown", "99", "A990"}

    surgery_group = pd.Series("surgery_recorded", index=df.index, dtype="object")
    surgery_group = surgery_group.mask(code_series.isin(no_surgery_codes), "no_surgery")
    surgery_group = surgery_group.mask(code_series.isin(unknown_codes), "surgery_unknown")

    return pd.DataFrame({
        "surgery_code_unified": code_series,
        "surgery_code_source": source,
        "surgery_group": surgery_group,
        "surgery_recorded_flag": (surgery_group == "surgery_recorded").astype(int),
        "no_surgery_flag": (surgery_group == "no_surgery").astype(int),
        "surgery_unknown_flag": (surgery_group == "surgery_unknown").astype(int),
    })


def preprocess_chemotherapy(s: pd.Series) -> pd.DataFrame:
    clean = s.replace({"": "No/Unknown", "Blank(s)": "No/Unknown"})

    return pd.DataFrame({
        "chemotherapy_status": clean,
        "chemotherapy_yes_flag": (clean == "Yes").astype(int),
        "chemotherapy_no_or_unknown_flag": (clean == "No/Unknown").astype(int),
    })


def preprocess_radiation(s: pd.Series) -> pd.DataFrame:
    group_map = {
        "Beam radiation": "radiation_given_beam",
        "Radioactive implants (includes brachytherapy) (1988+)": "radiation_given_implants",
        "Combination of beam with implants or isotopes": "radiation_given_combination",
        "Radioisotopes (1988+)": "radiation_given_radioisotopes",
        "Radiation, NOS  method or source not specified": "radiation_given_nos",
        "Refused (1988+)": "radiation_refused",
        "Recommended, unknown if administered": "radiation_recommended_unknown",
        "None/Unknown": "radiation_none_or_unknown",
    }

    group = s.map(group_map).fillna("radiation_unknown")

    radiation_given = group.isin([
        "radiation_given_beam",
        "radiation_given_implants",
        "radiation_given_combination",
        "radiation_given_radioisotopes",
        "radiation_given_nos",
    ])

    return pd.DataFrame({
        "radiation_group": group,
        "radiation_given_flag": radiation_given.astype(int),
        "radiation_refused_flag": (group == "radiation_refused").astype(int),
        "radiation_uncertain_flag": group.isin([
            "radiation_none_or_unknown",
            "radiation_recommended_unknown",
            "radiation_unknown",
        ]).astype(int),
    })


def preprocess_survive_after_5(df: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(df["survive_after_5_raw"], errors="coerce").astype("Int64")

    invalid_values = target.dropna()[~target.dropna().isin([0, 1])]

    if len(invalid_values) > 0:
        raise ValueError(
            "Cột survive_after_5 chỉ được chứa 0 hoặc 1. "
            f"Phát hiện giá trị không hợp lệ: {invalid_values.unique().tolist()}"
        )

    return pd.DataFrame({
        "survive_after_5": target,
    }, index=df.index)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)

    parts = [
        preprocess_year(df["diagnosis_year_raw"]),
        preprocess_age(df["age_group_raw"]),
        preprocess_race(df["race_raw"]),
        preprocess_grade(df),
        preprocess_laterality(df["laterality_raw"]),
        preprocess_stage(df["summary_stage_raw"]),
        preprocess_tumor_size(df["tumor_size_raw"]),
        preprocess_nodes(df),
        preprocess_receptors(df),
        preprocess_surgery(df),
        preprocess_chemotherapy(df["chemotherapy_raw"]),
        preprocess_radiation(df["radiation_raw"]),
        preprocess_survive_after_5(df),
    ]

    processed = pd.concat(parts, axis=1)

    # Keep raw values that are useful for auditing/debugging.
    # Vital status and Survival months are intentionally not kept.
    audit_cols = [
        "grade_2018_raw",
        "grade_thru_2017_raw",
        "summary_stage_raw",
        "tumor_size_raw",
        "nodes_positive_raw",
        "nodes_examined_raw",
        "surgery_1998_2022_raw",
        "surgery_2023_raw",
    ]

    processed = pd.concat([processed, df[audit_cols]], axis=1)

    return processed


def print_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame, input_path: Path) -> None:
    print("=" * 100)
    print("PREPROCESSING SUMMARY")
    print("=" * 100)
    print(f"Input file : {input_path.resolve()}")
    print(f"Raw shape  : {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    print(f"Output file: {OUTPUT_PATH.resolve()}")
    print(f"Output shape: {processed_df.shape[0]} rows x {processed_df.shape[1]} columns")

    print("\nTarget distribution:")
    print(
        processed_df["survive_after_5"]
        .value_counts(dropna=False)
        .rename(index={
            0: "Not survive after 5 years",
            1: "Survive after 5 years",
        })
    )

    print("\nKey missing / unknown flags:")
    flags = [
        "grade_unknown_flag",
        "stage_unknown_flag",
        "tumor_size_unknown_flag",
        "nodes_positive_unknown_flag",
        "nodes_examined_unknown_flag",
        "her2_not_available_flag",
        "surgery_unknown_flag",
        "chemotherapy_no_or_unknown_flag",
        "radiation_uncertain_flag",
    ]

    for col in flags:
        if col in processed_df.columns:
            rate = processed_df[col].mean() * 100
            print(f"- {col}: {rate:.2f}%")

    forbidden_cols = [
        "event_dead",
        "survival_months_int",
        "survival_months_unknown_flag",
        "alive_censored_flag",
        "vital_status_raw",
        "survival_months_raw",
    ]

    existing_forbidden_cols = [
        col for col in forbidden_cols
        if col in processed_df.columns
    ]

    print("\nLeakage check:")
    if existing_forbidden_cols:
        print("WARNING: These forbidden columns still exist:")
        for col in existing_forbidden_cols:
            print(f"- {col}")
    else:
        print("- OK: Vital status and Survival months are not in processed output.")

    print("\nImportant warning:")
    print("- survive_after_5 is the target variable, not an input feature.")
    print("- Vital status and Survival months are intentionally not used in the processed output.")
    print("- treatment variables can cause temporal leakage if prediction time is before treatment.")
    print("- perform imputation / one-hot encoding inside the train/test pipeline, not before splitting.")


def main() -> None:
    input_path = find_input_file()
    raw_df = load_raw_data(input_path)
    processed_df = preprocess(raw_df)

    processed_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print_summary(raw_df, processed_df, input_path)


if __name__ == "__main__":
    main()