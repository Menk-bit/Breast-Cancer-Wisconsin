import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class StandardScalerScratch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0

        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler chưa được fit.")

        X = np.asarray(X, dtype=float)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def load_data(csv_path):
    return pd.read_csv(csv_path)


def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if "diagnosis" not in df.columns:
        raise ValueError("Không tìm thấy cột diagnosis.")

    df["diagnosis"] = df["diagnosis"].map(
        {
            "M": 1,
            "B": 0,
            "Malignant": 1,
            "Benign": 0,
        }
    )

    if df["diagnosis"].isna().any():
        invalid_values = df.loc[df["diagnosis"].isna(), "diagnosis"].unique()
        raise ValueError(f"Có nhãn diagnosis không hợp lệ: {invalid_values}")

    X = df.drop(columns=["diagnosis"])
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("Không còn feature số sau tiền xử lý.")

    y = df["diagnosis"].astype(int)
    feature_names = X.columns.tolist()

    return X, y, feature_names


def split_train_test(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def select_features_by_correlation(X_train, y_train, threshold):
    corr_with_target = X_train.corrwith(y_train).fillna(0)

    corr_table = pd.DataFrame(
        {
            "feature": corr_with_target.index,
            "corr_with_target": corr_with_target.values,
            "abs_corr": corr_with_target.abs().values,
        }
    ).sort_values("abs_corr", ascending=False)

    selected_features = corr_table.loc[
        corr_table["abs_corr"] >= threshold,
        "feature",
    ].tolist()

    if not selected_features:
        selected_features = [corr_table.iloc[0]["feature"]]

    return selected_features, corr_table


def get_selected_features(X_train, y_train, feature_mode, corr_threshold):
    if feature_mode == "full":
        return X_train.columns.tolist(), None

    if feature_mode == "correlation":
        return select_features_by_correlation(
            X_train,
            y_train,
            threshold=corr_threshold,
        )

    raise ValueError("feature_mode chỉ được là 'full' hoặc 'correlation'.")


def prepare_data(X_train, X_valid, selected_features, use_scaling):
    X_train_selected = X_train[selected_features].to_numpy(dtype=float)
    X_valid_selected = X_valid[selected_features].to_numpy(dtype=float)

    if not use_scaling:
        return X_train_selected, X_valid_selected, None

    scaler = StandardScalerScratch()

    X_train_prepared = scaler.fit_transform(X_train_selected)
    X_valid_prepared = scaler.transform(X_valid_selected)

    return X_train_prepared, X_valid_prepared, scaler
