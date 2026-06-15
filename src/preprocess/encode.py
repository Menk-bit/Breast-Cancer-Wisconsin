from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = DATA_DIR / "preprocessed_breast_cancer.csv"

OUTPUT_TREE_PATH = DATA_DIR / "model_ready_tree.csv"
OUTPUT_SCALED_PATH = DATA_DIR / "model_ready_scaled.csv"


# =========================================================
# FEATURE WHITELIST
# =========================================================
# Chỉ các cột trong whitelist này mới được đưa vào model-ready.
# Không tự động encode toàn bộ object columns.

CONTINUOUS_NUMERIC_FEATURES = [
    # Time / demographic
    "diagnosis_year",
    "age_midpoint",

    # Tumor biology / stage
    "grade_unified_num",
    "stage_ordinal",

    # Tumor burden
    "tumor_size_mm",
    "log1p_tumor_size_mm",

    # Lymph nodes
    "nodes_positive_count",
    "nodes_examined_count",
    "nodes_positive_ratio",

    # Biomarker summary
    "receptor_unknown_or_unavailable_count",
]


BINARY_FLAG_FEATURES = [
    # Age
    "age_90_plus_flag",
    "age_unknown_flag",

    # Grade
    "grade_unknown_flag",

    # Laterality
    "laterality_unknown_flag",

    # Stage
    "stage_in_situ_flag",
    "stage_localized_flag",
    "stage_regional_flag",
    "stage_distant_flag",
    "stage_unknown_flag",
    "regional_direct_extension_flag",
    "regional_lymph_node_involved_flag",

    # Tumor size
    "tumor_size_unknown_flag",
    "tumor_size_special_flag",
    "tumor_size_microscopic_focus_flag",
    "tumor_size_no_primary_evidence_flag",

    # Nodes
    "nodes_positive_unknown_flag",
    "nodes_examined_unknown_flag",
    "nodes_positive_any_flag",
    "nodes_examined_any_flag",
    "nodes_ratio_unknown_flag",

    # ER
    "er_positive_flag",
    "er_negative_flag",
    "er_unknown_flag",
    "er_not_available_flag",

    # PR
    "pr_positive_flag",
    "pr_negative_flag",
    "pr_unknown_flag",
    "pr_not_available_flag",

    # HER2
    "her2_positive_flag",
    "her2_negative_flag",
    "her2_unknown_flag",
    "her2_not_available_flag",

    # Derived receptor subtype
    "hr_positive_flag",
    "triple_negative_flag",

    # Surgery summary
    "surgery_recorded_flag",
    "no_surgery_flag",
    "surgery_unknown_flag",

    # Chemotherapy
    "chemotherapy_yes_flag",
    "chemotherapy_no_or_unknown_flag",

    # Radiation summary
    "radiation_given_flag",
    "radiation_refused_flag",
    "radiation_uncertain_flag",
]


# Chỉ one-hot những categorical thật sự cần, số nhóm ít, dễ giải thích.
# Không one-hot raw tumor size, raw surgery code, raw survival months.
CATEGORICAL_ONEHOT_FEATURES = [
    "diagnosis_era",
    "race",
    "laterality",
]


# Target mới.
# Đây là output target cho bài toán sống sau 5 năm.
TARGET_COLUMNS = [
    "survive_after_5",
]


# Các cột luôn bị cấm đưa vào feature hoặc output model-ready.
# Vì preprocess mới đã bỏ Vital status và Survival months, các cột này bình thường sẽ không tồn tại.
# Nhưng vẫn giữ check để tránh vô tình chạy nhầm file cũ.
FORBIDDEN_EXACT_COLUMNS = [
    # Old outcome / leakage columns
    "event_dead",
    "survival_months",
    "Survival months",
    "survival_months_int",
    "survival_months_unknown_flag",
    "survival_months_raw",
    "vital_status",
    "vital_status_raw",
    "Vital status recode (study cutoff used)",
    "alive_censored_flag",

    # Not useful in this sample
    "sex",
    "sex_raw",
    "sex_unknown_flag",
]


# Các prefix raw one-hot không bao giờ được phép xuất hiện trong model-ready.
FORBIDDEN_OUTPUT_PREFIXES = [
    "survival_months_raw_",
    "survival_months_int_",
    "survival_months_unknown_flag_",
    "vital_status_raw_",
    "event_dead_",

    "tumor_size_raw_",
    "grade_2018_raw_",
    "grade_thru_2017_raw_",
    "summary_stage_raw_",
    "surgery_1998_2022_raw_",
    "surgery_2023_raw_",
    "surgery_code_unified_",
    "grade_unified_raw_",
    "age_group_",
]


# =========================================================
# LOAD
# =========================================================

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {path.resolve()}")

    df = pd.read_csv(path)

    print("=" * 100)
    print("LOADED DATA")
    print("=" * 100)
    print(f"Input file: {path.resolve()}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


# =========================================================
# COLUMN SELECTION
# =========================================================

def keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def report_missing_columns(df: pd.DataFrame, columns: list[str], group_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]

    if missing:
        print(f"\nMissing columns in group [{group_name}] - skipped:")
        for col in missing:
            print(f"  - {col}")


def select_columns(df: pd.DataFrame):
    continuous_cols = keep_existing_columns(df, CONTINUOUS_NUMERIC_FEATURES)
    binary_cols = keep_existing_columns(df, BINARY_FLAG_FEATURES)
    categorical_cols = keep_existing_columns(df, CATEGORICAL_ONEHOT_FEATURES)
    target_cols = keep_existing_columns(df, TARGET_COLUMNS)

    report_missing_columns(df, CONTINUOUS_NUMERIC_FEATURES, "continuous numeric")
    report_missing_columns(df, BINARY_FLAG_FEATURES, "binary flags")
    report_missing_columns(df, CATEGORICAL_ONEHOT_FEATURES, "categorical one-hot")
    report_missing_columns(df, TARGET_COLUMNS, "target columns")

    if "survive_after_5" not in target_cols:
        raise ValueError(
            "Không tìm thấy target column 'survive_after_5' trong input.\n"
            "Hãy kiểm tra lại clean.py hoặc file preprocessed_breast_cancer.csv."
        )

    selected_feature_cols = continuous_cols + binary_cols + categorical_cols

    forbidden_selected = [
        col for col in selected_feature_cols
        if col in FORBIDDEN_EXACT_COLUMNS
    ]

    if forbidden_selected:
        raise ValueError(
            "Có cột forbidden bị chọn nhầm vào feature:\n"
            + "\n".join(forbidden_selected)
        )

    return continuous_cols, binary_cols, categorical_cols, target_cols


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_continuous_features(df: pd.DataFrame, continuous_cols: list[str]):
    if not continuous_cols:
        return pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)

    continuous_raw = df[continuous_cols].copy()

    for col in continuous_cols:
        continuous_raw[col] = pd.to_numeric(continuous_raw[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")

    continuous_imputed = pd.DataFrame(
        imputer.fit_transform(continuous_raw),
        columns=continuous_cols,
        index=df.index
    )

    scaler = StandardScaler()

    continuous_scaled = pd.DataFrame(
        scaler.fit_transform(continuous_imputed),
        columns=continuous_cols,
        index=df.index
    )

    return continuous_imputed, continuous_scaled


def prepare_binary_features(df: pd.DataFrame, binary_cols: list[str]):
    if not binary_cols:
        return pd.DataFrame(index=df.index)

    binary_df = df[binary_cols].copy()

    for col in binary_cols:
        binary_df[col] = pd.to_numeric(binary_df[col], errors="coerce")
        binary_df[col] = binary_df[col].fillna(0)

        # Ép về 0/1 an toàn.
        binary_df[col] = (binary_df[col] > 0).astype(int)

    return binary_df


def prepare_categorical_features(df: pd.DataFrame, categorical_cols: list[str]):
    if not categorical_cols:
        return pd.DataFrame(index=df.index)

    categorical_df = df[categorical_cols].copy()

    for col in categorical_cols:
        categorical_df[col] = categorical_df[col].astype("string")
        categorical_df[col] = categorical_df[col].fillna("Unknown")
        categorical_df[col] = categorical_df[col].str.strip()
        categorical_df[col] = categorical_df[col].replace({
            "": "Unknown",
            "nan": "Unknown",
            "NaN": "Unknown",
            "None": "Unknown",
            "Blank(s)": "Unknown",
        })

    onehot_df = pd.get_dummies(
        categorical_df,
        columns=categorical_cols,
        dummy_na=False,
        drop_first=False,
        dtype=int
    )

    return onehot_df


def prepare_targets(df: pd.DataFrame, target_cols: list[str]):
    target_df = df[target_cols].copy()

    target_df["survive_after_5"] = pd.to_numeric(
        target_df["survive_after_5"],
        errors="coerce"
    )

    missing_count = target_df["survive_after_5"].isna().sum()

    if missing_count > 0:
        raise ValueError(
            f"Cột survive_after_5 có {missing_count} giá trị bị thiếu hoặc không parse được."
        )

    target_df["survive_after_5"] = target_df["survive_after_5"].astype(int)

    invalid_values = sorted(
        target_df.loc[
            ~target_df["survive_after_5"].isin([0, 1]),
            "survive_after_5"
        ].unique()
    )

    if invalid_values:
        raise ValueError(
            "Cột survive_after_5 chỉ được chứa 0 hoặc 1.\n"
            f"Giá trị không hợp lệ: {invalid_values}"
        )

    return target_df


# =========================================================
# SAFETY CHECKS
# =========================================================

def check_forbidden_output_columns(df: pd.DataFrame, file_name: str):
    bad_cols = []

    for col in df.columns:
        if col in FORBIDDEN_EXACT_COLUMNS:
            bad_cols.append(col)

        for prefix in FORBIDDEN_OUTPUT_PREFIXES:
            if col.startswith(prefix):
                bad_cols.append(col)

    bad_cols = sorted(set(bad_cols))

    if bad_cols:
        preview = "\n".join(bad_cols[:80])
        raise ValueError(
            f"{file_name} vẫn chứa forbidden/raw/leakage columns:\n"
            f"{preview}\n"
            f"Tổng số cột lỗi: {len(bad_cols)}"
        )


def check_target_position(df: pd.DataFrame, file_name: str):
    if "survive_after_5" not in df.columns:
        raise ValueError(f"{file_name} không có cột target survive_after_5.")

    if df.columns[-1] != "survive_after_5":
        raise ValueError(
            f"{file_name}: cột cuối cùng phải là survive_after_5, "
            f"nhưng hiện tại là {df.columns[-1]}"
        )


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


# =========================================================
# REPORT
# =========================================================

def print_report(
    original_df,
    continuous_cols,
    binary_cols,
    categorical_cols,
    target_cols,
    categorical_encoded,
    tree_ready,
    scaled_ready
):
    print("\n" + "=" * 100)
    print("ENCODING REPORT - COMPACT / IMPORTANT FEATURES ONLY")
    print("=" * 100)

    print(f"Original shape: {original_df.shape[0]} rows x {original_df.shape[1]} columns")

    print("\nSelected feature groups:")
    print(f"- Continuous numeric features: {len(continuous_cols)}")
    print(f"- Binary flag features:        {len(binary_cols)}")
    print(f"- Categorical columns:         {len(categorical_cols)}")
    print(f"- One-hot columns generated:   {categorical_encoded.shape[1]}")
    print(f"- Target column kept:          {target_cols}")

    if continuous_cols:
        print("\nContinuous numeric features:")
        for col in continuous_cols:
            print(f"  - {col}")

    if binary_cols:
        print("\nBinary flag features:")
        for col in binary_cols:
            print(f"  - {col}")

    if categorical_cols:
        print("\nCategorical columns one-hot encoded:")
        for col in categorical_cols:
            print(f"  - {col}")

    print("\nExplicitly NOT encoded / NOT kept:")
    print("- Vital status recode")
    print("- Survival months")
    print("- event_dead")
    print("- survival_months_int")
    print("- survival_months_unknown_flag")
    print("- alive_censored_flag")
    print("- raw tumor size values")
    print("- raw surgery codes")
    print("- raw grade columns")
    print("- raw summary stage columns")
    print("- age_group raw one-hot")
    print("- sex, because dataset is 100% Female in your sample")

    print("\nOutput shapes:")
    print(f"- {OUTPUT_TREE_PATH.name}:   {tree_ready.shape[0]} rows x {tree_ready.shape[1]} columns")
    print(f"- {OUTPUT_SCALED_PATH.name}: {scaled_ready.shape[0]} rows x {scaled_ready.shape[1]} columns")

    print("\nTarget distribution:")
    print(
        tree_ready["survive_after_5"]
        .value_counts(dropna=False)
        .rename(index={
            0: "Not survive after 5 years",
            1: "Survive after 5 years",
        })
    )

    print("\nImportant note:")
    print("- Tree file: continuous numeric is imputed but not scaled.")
    print("- Scaled file: continuous numeric is imputed + StandardScaler.")
    print("- Binary flags and one-hot columns remain 0/1 in both files.")
    print("- survive_after_5 is the only target column.")
    print("- Vital status and Survival months are not encoded or kept in the output files.")


# =========================================================
# MAIN
# =========================================================

def main():
    df = load_data(INPUT_PATH)

    continuous_cols, binary_cols, categorical_cols, target_cols = select_columns(df)

    continuous_tree, continuous_scaled = prepare_continuous_features(
        df,
        continuous_cols
    )

    binary_features = prepare_binary_features(
        df,
        binary_cols
    )

    categorical_encoded = prepare_categorical_features(
        df,
        categorical_cols
    )

    target_df = prepare_targets(
        df,
        target_cols
    )

    # Tree-ready:
    # continuous numeric imputed, binary 0/1, one-hot 0/1
    tree_features = pd.concat(
        [continuous_tree, binary_features, categorical_encoded],
        axis=1
    )

    # Scaled-ready:
    # continuous numeric scaled, binary 0/1, one-hot 0/1
    scaled_features = pd.concat(
        [continuous_scaled, binary_features, categorical_encoded],
        axis=1
    )

    # Target được để cuối file.
    tree_ready = pd.concat([tree_features, target_df], axis=1)
    scaled_ready = pd.concat([scaled_features, target_df], axis=1)

    tree_ready = drop_duplicate_columns(tree_ready)
    scaled_ready = drop_duplicate_columns(scaled_ready)

    check_forbidden_output_columns(tree_ready, OUTPUT_TREE_PATH.name)
    check_forbidden_output_columns(scaled_ready, OUTPUT_SCALED_PATH.name)

    check_target_position(tree_ready, OUTPUT_TREE_PATH.name)
    check_target_position(scaled_ready, OUTPUT_SCALED_PATH.name)

    tree_ready.to_csv(OUTPUT_TREE_PATH, index=False, encoding="utf-8-sig")
    scaled_ready.to_csv(OUTPUT_SCALED_PATH, index=False, encoding="utf-8-sig")

    print_report(
        original_df=df,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols,
        categorical_encoded=categorical_encoded,
        tree_ready=tree_ready,
        scaled_ready=scaled_ready
    )

    print("\n" + "=" * 100)
    print("FILES CREATED")
    print("=" * 100)
    print(f"Saved: {OUTPUT_TREE_PATH.resolve()}")
    print(f"Saved: {OUTPUT_SCALED_PATH.resolve()}")


if __name__ == "__main__":
    main()
