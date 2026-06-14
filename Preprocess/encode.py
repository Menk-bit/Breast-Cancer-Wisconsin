from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = Path("preprocessed_breast_cancer.csv")

OUTPUT_TREE_PATH = Path("model_ready_tree.csv")
OUTPUT_SCALED_PATH = Path("model_ready_scaled.csv")

# Các cột outcome giữ lại ở cuối file, nhưng không dùng làm feature để encode/scale.
TARGET_CANDIDATES = [
    "event_dead",
    "survival_months",
    "survival_months_int",
    "survival_months_unknown_flag",
]

# Các cột không nên đưa vào feature nếu còn tồn tại trong file clinical.
# Phòng trường hợp preprocess_clinical vẫn giữ một số raw outcome/raw text.
FORCE_DROP_FROM_FEATURES = [
    "vital_status",
    "vital_status_raw",
    "Vital status recode (study cutoff used)",
    "Survival months",
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
# COLUMN SPLIT
# =========================================================

def split_feature_and_target(df: pd.DataFrame):
    """
    Tách feature và target.

    Với classification Dead/Alive:
    - y = event_dead
    - survival_months không được đưa vào X vì là outcome time.

    Với survival analysis:
    - dùng event_dead + survival_months làm outcome riêng.
    """

    target_cols = [col for col in TARGET_CANDIDATES if col in df.columns]

    feature_df = df.copy()

    # Bỏ target khỏi feature.
    feature_df = feature_df.drop(columns=target_cols, errors="ignore")

    # Bỏ các cột raw outcome nếu có.
    feature_df = feature_df.drop(columns=FORCE_DROP_FROM_FEATURES, errors="ignore")

    target_df = df[target_cols].copy() if target_cols else pd.DataFrame(index=df.index)

    return feature_df, target_df, target_cols


def detect_column_types(feature_df: pd.DataFrame):
    """
    Tách numeric và categorical.

    Numeric:
    - int, float, bool

    Categorical:
    - object/category/string
    """

    categorical_cols = feature_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_cols = feature_df.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    # Nếu có cột không rơi vào 2 nhóm trên, ép sang categorical cho an toàn.
    known_cols = set(categorical_cols) | set(numeric_cols)
    other_cols = [col for col in feature_df.columns if col not in known_cols]

    if other_cols:
        print("\nWarning: Một số cột có dtype lạ, sẽ xử lý như categorical:")
        for col in other_cols:
            print(f"- {col}: {feature_df[col].dtype}")
        categorical_cols.extend(other_cols)

    return numeric_cols, categorical_cols


# =========================================================
# ENCODING
# =========================================================

def clean_categorical_values(feature_df: pd.DataFrame, categorical_cols):
    """
    Chuẩn hóa categorical trước one-hot.
    Không xóa Unknown/Blank vì trong dữ liệu y tế chúng có ý nghĩa riêng.
    """

    categorical_df = feature_df[categorical_cols].copy()

    for col in categorical_cols:
        categorical_df[col] = categorical_df[col].astype("string")
        categorical_df[col] = categorical_df[col].fillna("Unknown")
        categorical_df[col] = categorical_df[col].str.strip()

        # Chuẩn hóa các ô rỗng thành Unknown.
        categorical_df[col] = categorical_df[col].replace({
            "": "Unknown",
            "nan": "Unknown",
            "NaN": "Unknown",
            "None": "Unknown",
        })

    return categorical_df


def encode_features(feature_df: pd.DataFrame, numeric_cols, categorical_cols):
    """
    Tạo 2 bản feature:

    1. tree_features:
       - numeric impute median
       - categorical one-hot
       - không scale

    2. scaled_features:
       - numeric impute median + StandardScaler
       - categorical one-hot
       - dùng cho KNN / Logistic Regression
    """

    # -------------------------
    # Numeric imputation
    # -------------------------
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="median")

        numeric_imputed = pd.DataFrame(
            numeric_imputer.fit_transform(feature_df[numeric_cols]),
            columns=numeric_cols,
            index=feature_df.index
        )
    else:
        numeric_imputed = pd.DataFrame(index=feature_df.index)

    # -------------------------
    # Categorical one-hot
    # -------------------------
    if categorical_cols:
        categorical_clean = clean_categorical_values(feature_df, categorical_cols)

        categorical_encoded = pd.get_dummies(
            categorical_clean,
            columns=categorical_cols,
            dummy_na=False,
            drop_first=False,
            dtype=int
        )
    else:
        categorical_encoded = pd.DataFrame(index=feature_df.index)

    # -------------------------
    # Tree-ready: không scale
    # -------------------------
    tree_features = pd.concat(
        [numeric_imputed, categorical_encoded],
        axis=1
    )

    # -------------------------
    # Scaled-ready: scale numeric, giữ one-hot 0/1
    # -------------------------
    if numeric_cols:
        scaler = StandardScaler()

        numeric_scaled = pd.DataFrame(
            scaler.fit_transform(numeric_imputed),
            columns=numeric_cols,
            index=feature_df.index
        )
    else:
        numeric_scaled = pd.DataFrame(index=feature_df.index)

    scaled_features = pd.concat(
        [numeric_scaled, categorical_encoded],
        axis=1
    )

    return tree_features, scaled_features


# =========================================================
# FINALIZE
# =========================================================

def attach_targets(feature_ready: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gắn target vào cuối file output để tiện train.
    Khi train classification, nhớ tách:
        y = event_dead
        X = toàn bộ cột trừ event_dead, survival_months
    """

    if target_df.empty:
        return feature_ready

    return pd.concat([feature_ready, target_df], axis=1)


def print_report(
    original_df,
    feature_df,
    target_cols,
    numeric_cols,
    categorical_cols,
    tree_ready,
    scaled_ready
):
    print("\n" + "=" * 100)
    print("ENCODING REPORT")
    print("=" * 100)

    print(f"Original shape: {original_df.shape[0]} rows x {original_df.shape[1]} columns")
    print(f"Feature columns before encoding: {feature_df.shape[1]}")
    print(f"Target/outcome columns kept: {target_cols}")

    print("\nColumn type split:")
    print(f"- Numeric columns: {len(numeric_cols)}")
    print(f"- Categorical columns: {len(categorical_cols)}")

    if numeric_cols:
        print("\nNumeric columns:")
        for col in numeric_cols:
            print(f"  - {col}")

    if categorical_cols:
        print("\nCategorical columns one-hot encoded:")
        for col in categorical_cols:
            print(f"  - {col}")

    print("\nOutput shapes:")
    print(f"- model_ready_tree.csv:   {tree_ready.shape[0]} rows x {tree_ready.shape[1]} columns")
    print(f"- model_ready_scaled.csv: {scaled_ready.shape[0]} rows x {scaled_ready.shape[1]} columns")

    print("\nImportant note:")
    print("- model_ready_tree.csv dùng cho Random Forest / Decision Tree / XGBoost.")
    print("- model_ready_scaled.csv dùng cho KNN / Logistic Regression / SVM.")
    print("- survival_months không được dùng làm feature nếu bài toán là predict Dead/Alive.")


# =========================================================
# MAIN
# =========================================================

def main():
    df = load_data(INPUT_PATH)

    feature_df, target_df, target_cols = split_feature_and_target(df)

    numeric_cols, categorical_cols = detect_column_types(feature_df)

    tree_features, scaled_features = encode_features(
        feature_df=feature_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    tree_ready = attach_targets(tree_features, target_df)
    scaled_ready = attach_targets(scaled_features, target_df)

    tree_ready.to_csv(OUTPUT_TREE_PATH, index=False, encoding="utf-8-sig")
    scaled_ready.to_csv(OUTPUT_SCALED_PATH, index=False, encoding="utf-8-sig")

    print_report(
        original_df=df,
        feature_df=feature_df,
        target_cols=target_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
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