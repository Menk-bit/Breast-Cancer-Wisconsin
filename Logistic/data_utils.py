import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "M": 1,
    "B": 0,
    "Malignant": 1,
    "Benign": 0,
}


def load_data(csv_path):
    return pd.read_csv(csv_path)


def map_target(series):
    mapped = series.map(LABEL_MAP)
    numeric = pd.to_numeric(series, errors="coerce")

    y = mapped.where(mapped.notna(), numeric)

    if y.isna().any():
        invalid_values = series[y.isna()].unique()
        raise ValueError(f"Có nhãn diagnosis không hợp lệ: {invalid_values}")

    if not y.isin([0, 1]).all():
        invalid_values = series[~y.isin([0, 1])].unique()
        raise ValueError(f"diagnosis chỉ được nhận giá trị 0/1 hoặc B/M: {invalid_values}")

    return y.astype(int)


def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if "diagnosis" not in df.columns:
        raise ValueError("Không tìm thấy cột diagnosis.")

    y = map_target(df["diagnosis"])

    X = df.drop(columns=["diagnosis"])
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("Không còn feature số sau tiền xử lý.")

    feature_names = X.columns.tolist()

    return X, y, feature_names


def split_train_test(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )