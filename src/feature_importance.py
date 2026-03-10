from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

from features import extract_features


def main() -> None:
    data_path = Path("data/processed/phishing_binary.csv")
    model_path = Path("models/random_forest.joblib")
    output_path = Path("reports/feature_importance.csv")
    figure_path = Path("reports/feature_importance.png")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}. Run data_ingest.py first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Random Forest model not found at {model_path}. Run train.py first."
        )

    df = pd.read_csv(data_path)
    X = extract_features(df)

    model = load(model_path)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)

    print("Top 10 feature importances:")
    print(importance_df.head(10))

    top_n = 10
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)

    print(f"\nSaved feature importance table to: {output_path}")
    print(f"Saved feature importance plot to: {figure_path}")


if __name__ == "__main__":
    main()