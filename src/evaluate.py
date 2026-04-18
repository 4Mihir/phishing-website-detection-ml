import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.model_utils import (
        REPORTS_DIR,
        build_training_data,
        load_model,
        split_training_data,
    )
except ImportError:
    from model_utils import REPORTS_DIR, build_training_data, load_model, split_training_data


def main() -> None:
    _, X, y = build_training_data()
    _, X_test, _, y_test = split_training_data(X, y)

    model = load_model()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "test_rows": int(len(y_test)),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / "evaluation_summary.json"
    report_path = REPORTS_DIR / "classification_report.csv"
    confusion_path = REPORTS_DIR / "confusion_matrix.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, digits=4, output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(report_path)

    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual_legitimate", "actual_phishing"],
        columns=["pred_legitimate", "pred_phishing"],
    )
    confusion_df.to_csv(confusion_path)

    print("Saved evaluation summary to:", summary_path)
    print("Saved classification report to:", report_path)
    print("Saved confusion matrix to:", confusion_path)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        if key == "test_rows":
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: {value:.4f}")


if __name__ == "__main__":
    main()
