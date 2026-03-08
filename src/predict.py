from pathlib import Path
import sys

import pandas as pd
from joblib import load

from features import extract_features


def predict_url(url: str) -> None:
    model_path = Path("models/random_forest.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run train.py first."
        )

    model = load(model_path)

    input_df = pd.DataFrame({"url": [url]})
    X_input = extract_features(input_df)

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"

    print("\n=== Prediction Result ===")
    print(f"URL: {url}")
    print(f"Prediction: {label}")
    print(f"Phishing Probability: {probability:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "https://example.com"')
        sys.exit(1)

    url = sys.argv[1]
    predict_url(url)