import pandas as pd

try:
    from src.model_utils import load_model, load_processed_dataset, predict_urls
except ImportError:
    from model_utils import load_model, load_processed_dataset, predict_urls


def inspect_urls(urls):
    model = load_model()
    predictions_df, features_df = predict_urls(urls, model=model)

    print("\n=== INPUT FEATURES ===")
    print(features_df)

    print("\n=== PREDICTIONS ===")
    print(predictions_df[["url", "label", "phishing_probability", "risk_level"]])

    df = load_processed_dataset()
    benign_samples = df[df["label"] == 0].head(5).copy()
    benign_predictions, _ = predict_urls(benign_samples["url"].tolist(), model=model)

    comparison_df = pd.DataFrame(
        {
            "url": benign_samples["url"].values,
            "label": benign_samples["label"].values,
            "prediction": benign_predictions["label"].values,
            "phishing_probability": benign_predictions["phishing_probability"].values,
            "risk_level": benign_predictions["risk_level"].values,
        }
    )

    print("\n=== KNOWN BENIGN DATASET SAMPLES ===")
    print(comparison_df)


if __name__ == "__main__":
    urls_to_test = [
        "https://www.bbc.co.uk",
        "https://www.google.com",
        "https://github.com",
        "https://www.microsoft.com",
        "http://secure-login-paypal-account.com",
    ]
    inspect_urls(urls_to_test)
