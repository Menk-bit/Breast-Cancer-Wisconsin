from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# =========================
# CONFIG
# =========================

CSV_PATH = REPO_ROOT / "data" / "preprocessed_breast_cancer.csv"

# Nếu file của bạn đang tên export.csv như trong ảnh thì đổi dòng trên thành:
# CSV_PATH = Path("export.csv")

MAX_VALUES_PER_COLUMN = None
# None  -> in toàn bộ giá trị unique của mỗi cột
# 50    -> chỉ in top 50 giá trị xuất hiện nhiều nhất mỗi cột


# =========================
# LOAD DATA
# =========================

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path.resolve()}")

    # Đọc toàn bộ dưới dạng string để không mất mã như 0014, 027, 00
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )

    # Strip khoảng trắng đầu/cuối trong cell dạng text
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    return df


# =========================
# BASIC STATISTICS
# =========================

def print_dataset_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("DATASET OVERVIEW")
    print("=" * 100)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:02d}. {col}")

    print("\nDataFrame info:")
    print(df.info())


def print_describe(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("DESCRIBE ALL COLUMNS")
    print("=" * 100)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 80
    ):
        print(df.describe(include="all").T)


def print_missing_like_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("MISSING / UNKNOWN-LIKE VALUE SUMMARY")
    print("=" * 100)

    missing_like_values = {
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

    rows = []

    for col in df.columns:
        s = df[col]

        true_null_count = s.isna().sum()
        empty_string_count = (s == "").sum()
        missing_like_count = s.isin(missing_like_values).sum()

        rows.append({
            "column": col,
            "n_rows": len(s),
            "n_unique": s.nunique(dropna=False),
            "true_null_count": true_null_count,
            "empty_string_count": empty_string_count,
            "missing_unknown_like_count": missing_like_count,
            "missing_unknown_like_percent": round(missing_like_count / len(s) * 100, 2),
        })

    summary = pd.DataFrame(rows)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 80
    ):
        print(summary)


def print_column_value_counts(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("VALUE COUNTS FOR EACH COLUMN")
    print("=" * 100)

    for i, col in enumerate(df.columns, start=1):
        print("\n" + "-" * 100)
        print(f"{i:02d}. COLUMN: {col}")
        print("-" * 100)

        s = df[col]
        counts = s.value_counts(dropna=False)

        if MAX_VALUES_PER_COLUMN is not None:
            counts = counts.head(MAX_VALUES_PER_COLUMN)

        value_count_df = counts.reset_index()
        value_count_df.columns = ["value", "count"]
        value_count_df["percent"] = (value_count_df["count"] / len(df) * 100).round(2)

        print(f"Number of unique values: {s.nunique(dropna=False)}")

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 220,
            "display.max_colwidth", 120
        ):
            print(value_count_df.to_string(index=False))


def print_sample_rows(df: pd.DataFrame, n: int = 5) -> None:
    print("\n" + "=" * 100)
    print(f"FIRST {n} ROWS")
    print("=" * 100)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 80
    ):
        print(df.head(n))


# =========================
# MAIN
# =========================

def main() -> None:
    df = load_data(CSV_PATH)

    print_dataset_overview(df)
    print_sample_rows(df, n=5)
    print_describe(df)
    print_missing_like_summary(df)
    print_column_value_counts(df)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
