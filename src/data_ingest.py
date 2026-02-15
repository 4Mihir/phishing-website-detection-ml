from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/malicious_urls.csv")


def load_raw() -> pd.DataFrame:
    """Load the raw Kaggle CSV file."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find dataset at {RAW_PATH}.\n"
            "Make sure you extracted the Kaggle zip and placed the CSV here:\n"
            "  data/raw/malicious_urls.csv"
        )

    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip().lower() for c in df.columns]  # normalise headers
    return df


def make_binary_phish_vs_benign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only benign + phishing rows and map to:
    0 = benign
    1 = phishing
    """
    url_col_candidates = ["url", "urls", "link", "website"]
    label_col_candidates = ["type", "label", "class", "category", "result"]

    url_col = next((c for c in url_col_candidates if c in df.columns), None)
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if url_col is None or label_col is None:
        raise ValueError(
            "Could not find URL/label columns.\n"
            f"Columns found: {list(df.columns)}\n"
            "Expected something like: url + type/label"
        )

    out = df[[url_col, label_col]].copy()
    out = out.rename(columns={url_col: "url", label_col: "label_raw"})

    out["label_raw"] = out["label_raw"].astype(str).str.strip().str.lower()

    out = out[out["label_raw"].isin(["benign", "phishing"])].copy()
    out["label"] = out["label_raw"].map({"benign": 0, "phishing": 1}).astype(int)

    out = out.drop(columns=["label_raw"]).dropna()
    out["url"] = out["url"].astype(str).str.strip()
    out = out[out["url"] != ""]

    return out.reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw()
    print("✅ Loaded raw shape:", df.shape)
    print("Raw columns:", list(df.columns))

    binary = make_binary_phish_vs_benign(df)
    print("\n✅ Binary dataset shape:", binary.shape)
    print(binary.head())

    print("\nClass distribution (0=benign, 1=phishing):")
    print(binary["label"].value_counts())

    from pathlib import Path

    processed_path = Path("data/processed/phishing_binary.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    binary.to_csv(processed_path, index=False)
    print(f"\nSaved cleaned dataset to: {processed_path}")

