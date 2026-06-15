import pandas as pd


# =========================================================
# CONFIG
# =========================================================

TREE_INPUT = "/Users/minhdt/Desktop/ML Breast/Preprocess/model_ready_tree.csv"
SCALED_INPUT = "/Users/minhdt/Desktop/ML Breast/Preprocess/model_ready_scaled.csv"

TARGET_COLS = [
    "event_dead",
    "survival_months_int",
    "survival_months_unknown_flag",
]

# Các cột này không bao giờ dùng làm feature khi train Dead/Alive
LEAKAGE_COLS = [
    "event_dead",
    "survival_months_int",
    "survival_months_unknown_flag",
]


# =========================================================
# SET 1: ALL VALID FEATURES
# =========================================================
# Không cần khai báo thủ công.
# Sẽ lấy toàn bộ cột trong model_ready trừ target/outcome.


# =========================================================
# SET 2: CHART-SELECTED FEATURES
# =========================================================
# Dựa chủ yếu trên biểu đồ:
# - Top filter-based
# - Top model-based
# - Overlap giữa 2 cách
# - Nhóm attribute có score cao

CHART_SELECTED_FEATURES = [
    # Age / diagnosis era
    "age_midpoint",
    "diagnosis_year",
    "diagnosis_era_pre_2010",
    "diagnosis_era_2010_2017",
    "diagnosis_era_2018_2022",
    "diagnosis_era_2023_plus",

    # HER2 / receptor
    "her2_not_available_flag",
    "her2_negative_flag",
    "receptor_unknown_or_unavailable_count",
    "triple_negative_flag",
    "hr_positive_flag",
    "er_positive_flag",
    "pr_positive_flag",

    # Stage
    "stage_localized_flag",
    "stage_regional_flag",
    "stage_unknown_flag",
    "stage_ordinal",
    "regional_direct_extension_flag",

    # Lymph nodes
    "nodes_examined_any_flag",
    "nodes_examined_count",
    "nodes_positive_unknown_flag",
    "nodes_ratio_unknown_flag",

    # Tumor size
    "tumor_size_microscopic_focus_flag",
    "tumor_size_special_flag",

    # Surgery / radiation
    "surgery_recorded_flag",
    "no_surgery_flag",
    "surgery_unknown_flag",
    "radiation_given_flag",
    "radiation_uncertain_flag",

    # Race / laterality
    "race_Unknown",
    "race_Black",
    "laterality_unknown_flag",
    "laterality_paired_site_laterality_unknown",
]


# =========================================================
# SET 3: CLINICAL-SELECTED FEATURES
# =========================================================
# Dựa trên:
# - Age
# - Grade
# - Stage
# - Tumor size
# - Lymph nodes
# - ER / PR / HER2
# - Receptor subtype
#
# Cố tình bỏ:
# - diagnosis_year
# - diagnosis_era_*
# - surgery_*
# - radiation_*
# - chemotherapy_*
#
# Vì các nhóm đó dễ phản ánh follow-up bias, coding era, hoặc thông tin sau chẩn đoán.

CLINICAL_SELECTED_FEATURES = [
    # Demographic
    "age_midpoint",

    # Grade
    "grade_unified_num",
    "grade_unknown_flag",

    # Stage
    "stage_ordinal",
    "stage_localized_flag",
    "stage_regional_flag",
    "stage_distant_flag",
    "stage_unknown_flag",
    "regional_direct_extension_flag",
    "regional_lymph_node_involved_flag",

    # Tumor size
    "log1p_tumor_size_mm",
    "tumor_size_unknown_flag",
    "tumor_size_special_flag",
    "tumor_size_microscopic_focus_flag",

    # Lymph nodes
    "nodes_positive_count",
    "nodes_examined_count",
    "nodes_positive_ratio",
    "nodes_positive_any_flag",
    "nodes_examined_any_flag",
    "nodes_positive_unknown_flag",
    "nodes_ratio_unknown_flag",

    # ER / PR
    "er_positive_flag",
    "er_unknown_flag",
    "pr_positive_flag",
    "pr_unknown_flag",

    # HER2
    "her2_positive_flag",
    "her2_negative_flag",
    "her2_unknown_flag",
    "her2_not_available_flag",

    # Derived receptor subtype
    "hr_positive_flag",
    "triple_negative_flag",
    "receptor_unknown_or_unavailable_count",

    # Race - giữ hạn chế, vì model-based có chọn race_Black/race_Unknown
    "race_Black",
    "race_Unknown",
]


# =========================================================
# FUNCTIONS
# =========================================================

def get_existing_targets(df):
    return [col for col in TARGET_COLS if col in df.columns]


def get_all_valid_features(df):
    return [col for col in df.columns if col not in LEAKAGE_COLS]


def make_selected_file(input_path, output_path, selected_features=None):
    df = pd.read_csv(input_path)

    target_cols = get_existing_targets(df)

    if selected_features is None:
        # All valid features
        feature_cols = get_all_valid_features(df)
        missing_features = []
    else:
        feature_cols = [col for col in selected_features if col in df.columns]
        missing_features = [col for col in selected_features if col not in df.columns]

    output_df = df[feature_cols + target_cols].copy()
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"Created: {output_path}")
    print(f"Input file: {input_path}")
    print(f"Selected features: {len(feature_cols)}")
    print(f"Targets kept: {target_cols}")

    if missing_features:
        print("Missing features skipped:")
        for col in missing_features:
            print(f"  - {col}")


def main():
    # =====================================================
    # 1. ALL VALID FEATURES
    # =====================================================

    make_selected_file(
        input_path=TREE_INPUT,
        output_path="model_ready_tree_all_valid.csv",
        selected_features=None
    )

    make_selected_file(
        input_path=SCALED_INPUT,
        output_path="model_ready_scaled_all_valid.csv",
        selected_features=None
    )

    # =====================================================
    # 2. CHART-SELECTED FEATURES
    # =====================================================

    make_selected_file(
        input_path=TREE_INPUT,
        output_path="model_ready_tree_chart_selected.csv",
        selected_features=CHART_SELECTED_FEATURES
    )

    make_selected_file(
        input_path=SCALED_INPUT,
        output_path="model_ready_scaled_chart_selected.csv",
        selected_features=CHART_SELECTED_FEATURES
    )

    # =====================================================
    # 3. CLINICAL-SELECTED FEATURES
    # =====================================================

    make_selected_file(
        input_path=TREE_INPUT,
        output_path="model_ready_tree_clinical_selected.csv",
        selected_features=CLINICAL_SELECTED_FEATURES
    )

    make_selected_file(
        input_path=SCALED_INPUT,
        output_path="model_ready_scaled_clinical_selected.csv",
        selected_features=CLINICAL_SELECTED_FEATURES
    )

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Created 6 files:")
    print("- model_ready_tree_all_valid.csv")
    print("- model_ready_scaled_all_valid.csv")
    print("- model_ready_tree_chart_selected.csv")
    print("- model_ready_scaled_chart_selected.csv")
    print("- model_ready_tree_clinical_selected.csv")
    print("- model_ready_scaled_clinical_selected.csv")


if __name__ == "__main__":
    main()