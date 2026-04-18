import argparse

try:
    from src.model_utils import predict_urls
except ImportError:
    from model_utils import predict_urls


def print_prediction(url: str, label: str, probability: float, risk_level: str) -> None:
    print("\n=== Prediction Result ===")
    print(f"URL: {url}")
    print(f"Prediction: {label}")
    print(f"Phishing Probability: {probability:.4f}")
    print(f"Risk Level: {risk_level}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether one or more URLs are phishing or legitimate."
    )
    parser.add_argument("urls", nargs="+", help='One or more URLs, e.g. "https://example.com"')
    args = parser.parse_args()

    predictions_df, _ = predict_urls(args.urls)

    for row in predictions_df.to_dict(orient="records"):
        print_prediction(
            url=row["url"],
            label=row["label"],
            probability=row["phishing_probability"],
            risk_level=row["risk_level"],
        )


if __name__ == "__main__":
    main()
