from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# CONFIG
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
TREE_FILE = REPO_ROOT / "data" / "model_ready_tree.csv"
SCALED_FILE = REPO_ROOT / "data" / "model_ready_scaled.csv"

OUTPUT_DIR = Path("feature_selection_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "survive_after_5"

LEAKAGE_COLS = [
    "survive_after_5",
    "survival_months_int",
    "survival_months_unknown_flag",
]

RANDOM_STATE = 42

# Bật / tắt 2 hướng feature selection ở đây
RUN_FILTER_BASED = True
RUN_MODEL_BASED = True

# Chọn bao nhiêu feature top đầu
TOP_K = 20

# Loại feature gần như không thay đổi
VARIANCE_THRESHOLD = 0.0001


# =========================================================
# BASIC UTILS
# =========================================================

def load_data():
    tree_df = pd.read_csv(TREE_FILE)
    scaled_df = pd.read_csv(SCALED_FILE)

    y = tree_df[TARGET_COL].astype(int)

    X_tree = tree_df.drop(columns=LEAKAGE_COLS, errors="ignore")
    X_scaled = scaled_df.drop(columns=LEAKAGE_COLS, errors="ignore")

    # Hai file phải cùng feature columns
    common_features = [col for col in X_tree.columns if col in X_scaled.columns]
    X_tree = X_tree[common_features]
    X_scaled = X_scaled[common_features]

    X_tree = X_tree.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_tree, X_scaled, y


def remove_low_variance_features(X_tree, X_scaled):
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(X_tree)

    kept_features = X_tree.columns[selector.get_support()].tolist()
    removed_features = X_tree.columns[~selector.get_support()].tolist()

    pd.DataFrame({
        "removed_feature": removed_features
    }).to_csv(
        OUTPUT_DIR / "removed_low_variance_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return X_tree[kept_features], X_scaled[kept_features], removed_features


def normalize_score(series):
    series = series.fillna(0)

    if series.max() == series.min():
        return series * 0

    scaler = MinMaxScaler()
    return pd.Series(
        scaler.fit_transform(series.values.reshape(-1, 1)).ravel(),
        index=series.index
    )


def infer_attribute_group(feature):
    name = feature.lower()

    if "diagnosis" in name:
        return "Diagnosis year / era"

    if "age" in name:
        return "Age"

    if "race" in name:
        return "Race"

    if "grade" in name:
        return "Grade"

    if "laterality" in name:
        return "Laterality"

    if "stage" in name or "regional_direct" in name or "regional_lymph" in name:
        return "Stage"

    if "tumor_size" in name or "log1p_tumor" in name:
        return "Tumor size"

    if "nodes" in name:
        return "Lymph nodes"

    if name.startswith("er_"):
        return "ER status"

    if name.startswith("pr_"):
        return "PR status"

    if "her2" in name:
        return "HER2 status"

    if "hr_positive" in name or "triple_negative" in name or "receptor" in name:
        return "Receptor subtype"

    if "surgery" in name:
        return "Surgery"

    if "chemo" in name:
        return "Chemotherapy"

    if "radiation" in name:
        return "Radiation"

    return "Other"


# =========================================================
# FILTER-BASED FEATURE SELECTION
# Correlation + Mutual Information
# =========================================================

def run_filter_based_selection(X_tree, y):
    rows = []

    for col in X_tree.columns:
        corr = np.corrcoef(X_tree[col], y)[0, 1]

        if np.isnan(corr):
            corr = 0.0

        rows.append({
            "feature": col,
            "correlation": corr,
            "correlation_abs": abs(corr),
        })

    corr_df = pd.DataFrame(rows)

    mi_scores = mutual_info_classif(
        X_tree,
        y,
        random_state=RANDOM_STATE,
        discrete_features="auto"
    )

    mi_df = pd.DataFrame({
        "feature": X_tree.columns,
        "mutual_info": mi_scores
    })

    result = corr_df.merge(mi_df, on="feature", how="left")

    result["correlation_norm"] = normalize_score(result["correlation_abs"])
    result["mutual_info_norm"] = normalize_score(result["mutual_info"])

    # Filter-based score: không train model, chỉ dựa trên thống kê
    result["filter_score"] = (
        0.50 * result["correlation_norm"]
        + 0.50 * result["mutual_info_norm"]
    )

    result["attribute_group"] = result["feature"].apply(infer_attribute_group)

    result = result.sort_values("filter_score", ascending=False)
    result["filter_rank"] = np.arange(1, len(result) + 1)

    result.to_csv(
        OUTPUT_DIR / "filter_based_feature_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    selected = result.head(min(TOP_K, len(result)))

    selected.to_csv(
        OUTPUT_DIR / f"selected_top{len(selected)}_filter_based.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result, selected


# =========================================================
# MODEL-BASED FEATURE SELECTION
# L1 Logistic + Random Forest Importance
# =========================================================

def run_model_based_selection(X_tree, X_scaled, y):
    # -------------------------
    # L1 Logistic Regression
    # -------------------------
    l1 = LogisticRegression(
        solver="saga",
        C=0.1,
        l1_ratio=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    )

    l1.fit(X_scaled, y)

    l1_df = pd.DataFrame({
        "feature": X_scaled.columns,
        "l1_coef": l1.coef_[0],
        "l1_coef_abs": np.abs(l1.coef_[0]),
        "l1_selected": np.abs(l1.coef_[0]) > 1e-8
    })

    # -------------------------
    # Random Forest Importance
    # -------------------------
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X_tree, y)

    rf_df = pd.DataFrame({
        "feature": X_tree.columns,
        "rf_importance": rf.feature_importances_
    })

    result = l1_df.merge(rf_df, on="feature", how="left")

    result["l1_norm"] = normalize_score(result["l1_coef_abs"])
    result["rf_norm"] = normalize_score(result["rf_importance"])

    # Model-based score: dùng 2 model phụ để đo importance
    result["model_based_score"] = (
        0.50 * result["l1_norm"]
        + 0.50 * result["rf_norm"]
    )

    result["attribute_group"] = result["feature"].apply(infer_attribute_group)

    result = result.sort_values("model_based_score", ascending=False)
    result["model_based_rank"] = np.arange(1, len(result) + 1)

    result.to_csv(
        OUTPUT_DIR / "model_based_feature_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    selected = result.head(min(TOP_K, len(result)))

    selected.to_csv(
        OUTPUT_DIR / f"selected_top{len(selected)}_model_based.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result, selected


# =========================================================
# COMPARISON
# =========================================================

def compare_filter_and_model_based(filter_result, model_result):
    if filter_result is None or model_result is None:
        return None

    comparison = filter_result[[
        "feature",
        "filter_rank",
        "filter_score",
        "correlation_abs",
        "mutual_info",
        "attribute_group"
    ]].merge(
        model_result[[
            "feature",
            "model_based_rank",
            "model_based_score",
            "l1_coef_abs",
            "rf_importance"
        ]],
        on="feature",
        how="outer"
    )

    comparison["rank_gap"] = (
        comparison["filter_rank"] - comparison["model_based_rank"]
    ).abs()

    comparison = comparison.sort_values(
        ["filter_rank", "model_based_rank"],
        ascending=True
    )

    comparison.to_csv(
        OUTPUT_DIR / "filter_vs_model_based_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    filter_top = set(filter_result.head(min(TOP_K, len(filter_result)))["feature"])
    model_top = set(model_result.head(min(TOP_K, len(model_result)))["feature"])

    overlap = sorted(filter_top & model_top)
    filter_only = sorted(filter_top - model_top)
    model_only = sorted(model_top - filter_top)

    overlap_df = pd.DataFrame({
        "group": (
            ["overlap"] * len(overlap)
            + ["filter_only"] * len(filter_only)
            + ["model_only"] * len(model_only)
        ),
        "feature": overlap + filter_only + model_only
    })

    overlap_df.to_csv(
        OUTPUT_DIR / "selected_feature_overlap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return comparison


# =========================================================
# PLOTS
# =========================================================

def save_barplot(df, score_col, rank_col, title, output_name):
    top_df = df.head(min(TOP_K, len(df))).copy()
    top_df = top_df.sort_values(score_col, ascending=True)

    plt.figure(figsize=(10, max(5, 0.35 * len(top_df))))
    sns.barplot(
        data=top_df,
        x=score_col,
        y="feature"
    )

    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300)
    plt.close()


def save_group_plot(result_df, score_col, output_name, title):
    group_df = (
        result_df
        .groupby("attribute_group")
        .agg(
            max_score=(score_col, "max"),
            mean_score=(score_col, "mean"),
            n_features=("feature", "count")
        )
        .reset_index()
        .sort_values("max_score", ascending=False)
    )

    group_df.to_csv(
        OUTPUT_DIR / output_name.replace(".png", ".csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plot_df = group_df.sort_values("max_score", ascending=True)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=plot_df,
        x="max_score",
        y="attribute_group"
    )

    plt.title(title)
    plt.xlabel("Max score")
    plt.ylabel("Attribute group")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300)
    plt.close()


def save_overlap_plot():
    overlap_path = OUTPUT_DIR / "selected_feature_overlap.csv"

    if not overlap_path.exists():
        return

    overlap_df = pd.read_csv(overlap_path)

    count_df = (
        overlap_df
        .groupby("group")
        .size()
        .reset_index(name="n_features")
    )

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=count_df,
        x="group",
        y="n_features"
    )

    plt.title("Overlap between filter-based and model-based selected features")
    plt.xlabel("Selection group")
    plt.ylabel("Number of features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_selection_overlap.png", dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    X_tree, X_scaled, y = load_data()

    X_tree, X_scaled, removed_features = remove_low_variance_features(
        X_tree,
        X_scaled
    )

    filter_result = None
    model_result = None

    if RUN_FILTER_BASED:
        filter_result, filter_selected = run_filter_based_selection(X_tree, y)

        save_barplot(
            filter_result,
            score_col="filter_score",
            rank_col="filter_rank",
            title=f"Top {min(TOP_K, len(filter_result))} filter-based selected features",
            output_name="plot_filter_based_top_features.png"
        )

        save_group_plot(
            filter_result,
            score_col="filter_score",
            output_name="plot_filter_based_attribute_groups.png",
            title="Filter-based feature importance by attribute group"
        )

    if RUN_MODEL_BASED:
        model_result, model_selected = run_model_based_selection(
            X_tree,
            X_scaled,
            y
        )

        save_barplot(
            model_result,
            score_col="model_based_score",
            rank_col="model_based_rank",
            title=f"Top {min(TOP_K, len(model_result))} model-based selected features",
            output_name="plot_model_based_top_features.png"
        )

        save_group_plot(
            model_result,
            score_col="model_based_score",
            output_name="plot_model_based_attribute_groups.png",
            title="Model-based feature importance by attribute group"
        )

    if RUN_FILTER_BASED and RUN_MODEL_BASED:
        compare_filter_and_model_based(filter_result, model_result)
        save_overlap_plot()

    print("=" * 80)
    print("FEATURE SELECTION DONE")
    print("=" * 80)
    print(f"Usable features after low-variance filter: {X_tree.shape[1]}")
    print(f"Removed low-variance features: {len(removed_features)}")
    print(f"RUN_FILTER_BASED: {RUN_FILTER_BASED}")
    print(f"RUN_MODEL_BASED:  {RUN_MODEL_BASED}")
    print(f"TOP_K: {TOP_K}")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")

    if RUN_FILTER_BASED:
        print(f"- selected_top{min(TOP_K, len(filter_result))}_filter_based.csv")

    if RUN_MODEL_BASED:
        print(f"- selected_top{min(TOP_K, len(model_result))}_model_based.csv")

    if RUN_FILTER_BASED and RUN_MODEL_BASED:
        print("- filter_vs_model_based_comparison.csv")
        print("- selected_feature_overlap.csv")


if __name__ == "__main__":
    main()
