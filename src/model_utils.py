from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

try:
    from src.features import extract_features, get_registered_domain, normalise_url
except ImportError:
    from features import extract_features, get_registered_domain, normalise_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DATA_PATH = DATA_DIR / "processed" / "phishing_binary.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "random_forest.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

CLASS_LABELS = {
    0: "LEGITIMATE",
    1: "PHISHING",
}


def load_processed_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run src/data_ingest.py first."
        )

    return pd.read_csv(path)


def augment_with_benign_root_domains(df: pd.DataFrame) -> pd.DataFrame:
    """
    The source dataset contains many benign deep links but very few homepage
    URLs. Adding benign registered domains improves generalisation for real
    websites such as bbc.co.uk or google.com.
    """
    benign_domains = (
        df.loc[df["label"] == 0, "url"]
        .map(get_registered_domain)
        .loc[lambda series: series != ""]
        .drop_duplicates()
    )

    synthetic_benign = pd.DataFrame(
        {
            "url": benign_domains,
            "label": 0,
        }
    )

    return pd.concat([df, synthetic_benign], ignore_index=True)


def build_training_data(
    df: pd.DataFrame | None = None,
    augment_benign_roots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    working_df = load_processed_dataset() if df is None else df.copy()

    if augment_benign_roots:
        working_df = augment_with_benign_root_domains(working_df)

    working_df = working_df.reset_index(drop=True)
    X = extract_features(working_df)
    y = working_df["label"].astype(int)

    return working_df, X, y


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src/train.py first."
        )

    return load(model_path)


def probability_to_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    if probability >= 0.5:
        return "Elevated"
    if probability >= 0.2:
        return "Low"
    return "Very Low"


def normalise_urls(urls: Iterable[str]) -> list[str]:
    cleaned_urls = [str(url).strip() for url in urls if str(url).strip()]
    if not cleaned_urls:
        raise ValueError("Provide at least one non-empty URL.")

    return cleaned_urls


def predict_urls(urls: Iterable[str], model=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_urls = normalise_urls(urls)
    input_df = pd.DataFrame({"url": cleaned_urls})

    X_input = extract_features(input_df)
    active_model = load_model() if model is None else model

    predictions = active_model.predict(X_input).astype(int)
    if hasattr(active_model, "predict_proba"):
        probabilities = active_model.predict_proba(X_input)[:, 1]
    else:
        probabilities = predictions.astype(float)

    output = input_df.copy()
    output["normalised_url"] = output["url"].map(normalise_url)
    output["prediction"] = predictions
    output["label"] = output["prediction"].map(CLASS_LABELS)
    output["phishing_probability"] = probabilities
    output["risk_level"] = output["phishing_probability"].map(probability_to_risk_level)

    return output, X_input


def predict_url(url: str, model=None) -> tuple[dict[str, object], pd.Series]:
    predictions_df, features_df = predict_urls([url], model=model)
    return predictions_df.iloc[0].to_dict(), features_df.iloc[0]
