import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.model_utils import REPORTS_DIR, build_training_data, load_model
except ImportError:
    from model_utils import REPORTS_DIR, build_training_data, load_model


def main() -> None:
    _, X, _ = build_training_data()
    model = load_model()

    output_path = REPORTS_DIR / "feature_importance.csv"
    figure_path = REPORTS_DIR / "feature_importance.png"

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)

    print("Top 10 feature importances:")
    print(importance_df.head(10))

    top_features = importance_df.head(10)

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
