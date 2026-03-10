from pathlib import Path

import pandas as pd
from joblib import load

from features import extract_features


def inspect_urls(urls):
    model_path = Path("models/random_forest.joblib")
    data_path = Path("data/processed/phishing_binary.csv")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    model = load(model_path)

    input_df = pd.DataFrame({"url": urls})
    X = extract_features(input_df)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    output = input_df.copy()
    output["prediction"] = preds
    output["phishing_probability"] = probs

    print("\n=== INPUT FEATURES ===")
    print(X)

    print("\n=== PREDICTIONS ===")
    print(output)

    # Compare with a few known benign samples from the dataset
    df = pd.read_csv(data_path)
    benign_samples = df[df["label"] == 0].head(5).copy()
    X_benign = extract_features(benign_samples)
    benign_preds = model.predict(X_benign)
    benign_probs = model.predict_proba(X_benign)[:, 1]

    benign_samples["prediction"] = benign_preds
    benign_samples["phishing_probability"] = benign_probs

    print("\n=== KNOWN BENIGN DATASET SAMPLES ===")
    print(benign_samples[["url", "label", "prediction", "phishing_probability"]])


if __name__ == "__main__":
    urls_to_test = [
        "https://www.bbc.co.uk",
        "https://www.google.com",
        "https://github.com",
        "https://www.microsoft.com",
        "http://secure-login-paypal-account.com",
    ]
    inspect_urls(urls_to_test)