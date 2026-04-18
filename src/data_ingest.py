from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "malicious_urls.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "phishing_binary.csv"


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find dataset at {RAW_PATH}.\n"
            "Make sure the Kaggle CSV has been placed in data/raw/malicious_urls.csv."
        )

    df = pd.read_csv(RAW_PATH)
    df.columns = [column.strip().lower() for column in df.columns]
    return df


def make_binary_phish_vs_benign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only benign and phishing rows and map them to:
    0 = benign
    1 = phishing
    """
    url_col_candidates = ["url", "urls", "link", "website"]
    label_col_candidates = ["type", "label", "class", "category", "result"]

    url_col = next((column for column in url_col_candidates if column in df.columns), None)
    label_col = next((column for column in label_col_candidates if column in df.columns), None)

    if url_col is None or label_col is None:
        raise ValueError(
            "Could not find URL and label columns.\n"
            f"Columns found: {list(df.columns)}\n"
            "Expected something similar to url + type/label."
        )

    output = df[[url_col, label_col]].copy()
    output = output.rename(columns={url_col: "url", label_col: "label_raw"})

    output["label_raw"] = output["label_raw"].astype(str).str.strip().str.lower()
    output = output[output["label_raw"].isin(["benign", "phishing"])].copy()
    output["label"] = output["label_raw"].map({"benign": 0, "phishing": 1}).astype(int)

    output = output.drop(columns=["label_raw"]).dropna()
    output["url"] = output["url"].astype(str).str.strip()
    output = output[output["url"] != ""]
    output = output.drop_duplicates(subset=["url", "label"]).reset_index(drop=True)

    return output


if __name__ == "__main__":
    raw_df = load_raw()
    print("Loaded raw dataset:", raw_df.shape)
    print("Raw columns:", list(raw_df.columns))

    binary_df = make_binary_phish_vs_benign(raw_df)
    print("\nBinary dataset shape:", binary_df.shape)
    print(binary_df.head())

    print("\nClass distribution (0 = benign, 1 = phishing):")
    print(binary_df["label"].value_counts().sort_index())

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    binary_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {PROCESSED_PATH}")
