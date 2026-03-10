from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_features


def plot_model_comparison(results_path: Path, output_dir: Path) -> None:
    results_df = pd.read_csv(results_path)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["f1_score"])
    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.title("Model Comparison by F1-score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_f1.png", dpi=300)
    plt.close()


def plot_roc_curve(model, X_test, y_test, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_random_forest.png", dpi=300)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, output_dir: Path) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_random_forest.png", dpi=300)
    plt.close()


def main() -> None:
    data_path = Path("data/processed/phishing_binary.csv")
    results_path = Path("reports/model_results.csv")
    model_path = Path("models/random_forest.joblib")
    output_dir = Path("reports/plots")

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    if not results_path.exists():
        raise FileNotFoundError(f"Model results file not found at {results_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Random Forest model not found at {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = extract_features(df)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = load(model_path)

    plot_model_comparison(results_path, output_dir)
    plot_roc_curve(model, X_test, y_test, output_dir)
    plot_confusion_matrix(model, X_test, y_test, output_dir)

    print(f"Saved plots to: {output_dir}")
    print("- model_comparison_f1.png")
    print("- roc_curve_random_forest.png")
    print("- confusion_matrix_random_forest.png")


if __name__ == "__main__":
    main()