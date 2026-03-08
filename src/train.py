from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from features import extract_features


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 50}")
    print(f"MODEL: {name}")
    print(f"{'=' * 50}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


def main():
    processed_path = Path("data/processed/phishing_binary.csv")

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. "
            "Run data_ingest.py first."
        )

    df = pd.read_csv(processed_path)

    print("Loaded processed dataset:", df.shape)

    X = extract_features(df)
    y = df["label"]

    print("Feature matrix shape:", X.shape)
    print("Label shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

        safe_name = name.lower().replace(" ", "_")
        model_path = models_dir / f"{safe_name}.joblib"
        dump(model, model_path)
        print(f"Saved model to: {model_path}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    results_path = Path("reports/model_results.csv")
    results_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved model comparison results to: {results_path}")
    print("\nModel Comparison Table:")
    print(results_df)


if __name__ == "__main__":
    main()