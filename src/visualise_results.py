import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

try:
    from src.model_utils import (
        REPORTS_DIR,
        build_training_data,
        load_model,
        split_training_data,
    )
except ImportError:
    from model_utils import REPORTS_DIR, build_training_data, load_model, split_training_data


def plot_model_comparison(results_path, output_dir) -> None:
    results_df = pd.read_csv(results_path).sort_values(by="f1_score", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["f1_score"])
    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.title("Model Comparison by F1-score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_f1.png", dpi=300)
    plt.close()


def plot_roc_curve(model, X_test, y_test, output_dir) -> None:
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_random_forest.png", dpi=300)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, output_dir) -> None:
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
    results_path = REPORTS_DIR / "model_results.csv"
    output_dir = REPORTS_DIR / "plots"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Model results file not found at {results_path}. Run src/train.py first."
        )

    _, X, y = build_training_data()
    _, X_test, _, y_test = split_training_data(X, y)
    model = load_model()

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(results_path, output_dir)
    plot_roc_curve(model, X_test, y_test, output_dir)
    plot_confusion_matrix(model, X_test, y_test, output_dir)

    print(f"Saved plots to: {output_dir}")
    print("- model_comparison_f1.png")
    print("- roc_curve_random_forest.png")
    print("- confusion_matrix_random_forest.png")


if __name__ == "__main__":
    main()
