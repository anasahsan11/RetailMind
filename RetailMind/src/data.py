import pandas as pd
from .config import DATA_PATH, TARGET_COLUMN, ID_COLUMNS


def load_dataset(path=DATA_PATH):
    """Load the Telco CSV and perform minimal, deterministic type cleanup."""
    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    # TotalCharges contains blank strings in the original Telco dataset.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(str).str.strip()
    X = df.drop(columns=[TARGET_COLUMN] + ID_COLUMNS, errors="ignore")
    return X, y, df


def inspect_dataset(df):
    """Print a compact schema, target-balance, and missing-value report."""
    print("\\n--- Dataset shape ---")
    print(df.shape)
    print("\\n--- Schema ---")
    print(df.dtypes)
    print("\\n--- Target balance ---")
    print(df["Churn"].value_counts(dropna=False))
    print("\\n--- Missing values ---")
    print(df.isna().sum().sort_values(ascending=False))
