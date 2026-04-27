import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.model_utils import PROJECT_ROOT, predict_urls
except ImportError:
    from model_utils import PROJECT_ROOT, predict_urls


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "manual_url_test_set.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "manual_url_test_results.csv"
REQUIRED_COLUMNS = {"url", "expected_label"}


def load_manual_test_set(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manual URL test set not found at {path}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Manual URL test set is missing required columns: {sorted(missing_columns)}"
        )

    df["expected_label"] = df["expected_label"].astype(str).str.strip().str.upper()
    valid_labels = {"LEGITIMATE", "PHISHING"}
    invalid_labels = sorted(set(df["expected_label"]) - valid_labels)
    if invalid_labels:
        raise ValueError(f"Unexpected labels found in manual URL test set: {invalid_labels}")

    return df


def build_results_table(manual_df: pd.DataFrame) -> pd.DataFrame:
    predictions_df, _ = predict_urls(manual_df["url"].tolist())

    results_df = manual_df.copy()
    results_df["predicted_label"] = predictions_df["label"].values
    results_df["phishing_probability"] = predictions_df["phishing_probability"].round(6).values
    results_df["risk_level"] = predictions_df["risk_level"].values
    results_df["is_correct"] = (
        results_df["expected_label"] == results_df["predicted_label"]
    )
    results_df["is_borderline"] = results_df["phishing_probability"].between(0.4, 0.6)

    return results_df


def build_overall_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    total = len(results_df)
    correct = int(results_df["is_correct"].sum())
    accuracy = correct / total if total else 0.0
    borderline_count = int(results_df["is_borderline"].sum())
    false_positives = int(
        (
            (results_df["expected_label"] == "LEGITIMATE")
            & (results_df["predicted_label"] == "PHISHING")
        ).sum()
    )
    false_negatives = int(
        (
            (results_df["expected_label"] == "PHISHING")
            & (results_df["predicted_label"] == "LEGITIMATE")
        ).sum()
    )

    return pd.DataFrame(
        [
            {"metric": "total_urls", "value": total},
            {"metric": "correct_predictions", "value": correct},
            {"metric": "manual_accuracy", "value": round(accuracy, 6)},
            {"metric": "borderline_predictions", "value": borderline_count},
            {"metric": "false_positives", "value": false_positives},
            {"metric": "false_negatives", "value": false_negatives},
        ]
    )


def build_label_breakdown(results_df: pd.DataFrame) -> pd.DataFrame:
    label_breakdown = (
        results_df.groupby("expected_label", as_index=False)
        .agg(
            total=("url", "size"),
            correct=("is_correct", "sum"),
            borderline=("is_borderline", "sum"),
            average_probability=("phishing_probability", "mean"),
        )
    )
    label_breakdown["accuracy"] = (
        label_breakdown["correct"] / label_breakdown["total"]
    )
    return label_breakdown


def build_category_breakdown(results_df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in results_df.columns:
        return pd.DataFrame()

    category_breakdown = (
        results_df.groupby("category", as_index=False)
        .agg(
            total=("url", "size"),
            correct=("is_correct", "sum"),
            borderline=("is_borderline", "sum"),
        )
    )
    category_breakdown["accuracy"] = (
        category_breakdown["correct"] / category_breakdown["total"]
    )
    return category_breakdown


def split_error_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    false_positives = results_df[
        (results_df["expected_label"] == "LEGITIMATE")
        & (results_df["predicted_label"] == "PHISHING")
    ]
    false_negatives = results_df[
        (results_df["expected_label"] == "PHISHING")
        & (results_df["predicted_label"] == "LEGITIMATE")
    ]
    return false_positives, false_negatives


def save_overview_chart(
    label_breakdown: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_lookup = {
        row["metric"]: float(row["value"]) for _, row in overall_metrics.iterrows()
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].bar(
        label_breakdown["expected_label"],
        label_breakdown["accuracy"],
        color=["#4c78a8", "#f58518"],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Expected Label")
    for index, value in enumerate(label_breakdown["accuracy"]):
        axes[0].text(index, value + 0.03, f"{value:.2f}", ha="center")

    error_labels = ["False Positives", "False Negatives", "Borderline"]
    error_values = [
        metric_lookup["false_positives"],
        metric_lookup["false_negatives"],
        metric_lookup["borderline_predictions"],
    ]
    axes[1].bar(error_labels, error_values, color=["#e45756", "#72b7b2", "#54a24b"])
    axes[1].set_ylabel("Count")
    axes[1].set_title("Manual Benchmark Error Profile")
    for index, value in enumerate(error_values):
        axes[1].text(index, value + 0.2, f"{int(value)}", ha="center")
    axes[1].tick_params(axis="x", rotation=10)

    fig.suptitle("Manual URL Benchmark Overview", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_report_ready_note(
    overall_metrics: pd.DataFrame,
    label_breakdown: pd.DataFrame,
    false_positives: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_lookup = {
        row["metric"]: float(row["value"]) for _, row in overall_metrics.iterrows()
    }
    legitimate_accuracy = label_breakdown.loc[
        label_breakdown["expected_label"] == "LEGITIMATE", "accuracy"
    ].iloc[0]
    phishing_accuracy = label_breakdown.loc[
        label_breakdown["expected_label"] == "PHISHING", "accuracy"
    ].iloc[0]

    lines = [
        "# Manual URL Benchmark Notes",
        "",
        "Suggested report/poster wording:",
        "",
        (
            f"- A curated manual benchmark of {int(metric_lookup['total_urls'])} URLs "
            f"was used to assess real-world behaviour outside the training split."
        ),
        (
            f"- The model achieved {metric_lookup['manual_accuracy']:.2%} overall manual-benchmark "
            f"accuracy, with {legitimate_accuracy:.2%} accuracy on legitimate URLs and "
            f"{phishing_accuracy:.2%} accuracy on phishing-style URLs."
        ),
        (
            f"- The benchmark produced {int(metric_lookup['false_positives'])} false positives, "
            f"{int(metric_lookup['false_negatives'])} false negatives, and "
            f"{int(metric_lookup['borderline_predictions'])} borderline predictions."
        ),
    ]

    if not false_positives.empty:
        lines.extend(
            [
                "",
                "Observed false positives:",
                *[
                    f"- {row.url} ({row.phishing_probability:.4f})"
                    for row in false_positives.itertuples()
                ],
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_analysis_artifacts(results_df: pd.DataFrame, output_path: Path) -> None:
    output_dir = output_path.parent
    overall_metrics = build_overall_metrics(results_df)
    label_breakdown = build_label_breakdown(results_df)
    category_breakdown = build_category_breakdown(results_df)
    false_positives, false_negatives = split_error_tables(results_df)
    borderline_df = results_df[results_df["is_borderline"]]

    overall_metrics.to_csv(output_dir / "manual_url_test_summary.csv", index=False)
    label_breakdown.to_csv(output_dir / "manual_url_test_breakdown_by_label.csv", index=False)

    if not category_breakdown.empty:
        category_breakdown.to_csv(
            output_dir / "manual_url_test_breakdown_by_category.csv",
            index=False,
        )

    false_positives.to_csv(output_dir / "manual_url_test_false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "manual_url_test_false_negatives.csv", index=False)
    borderline_df.to_csv(output_dir / "manual_url_test_borderline_cases.csv", index=False)

    save_overview_chart(
        label_breakdown=label_breakdown,
        overall_metrics=overall_metrics,
        output_path=output_dir / "manual_url_test_overview.png",
    )
    save_report_ready_note(
        overall_metrics=overall_metrics,
        label_breakdown=label_breakdown,
        false_positives=false_positives,
        output_path=output_dir / "manual_url_test_notes.md",
    )


def print_summary(results_df: pd.DataFrame) -> None:
    overall_metrics = build_overall_metrics(results_df)
    label_breakdown = build_label_breakdown(results_df)
    category_breakdown = build_category_breakdown(results_df)
    false_positives, false_negatives = split_error_tables(results_df)
    borderline_df = results_df[results_df["is_borderline"]]

    metric_lookup = {
        row["metric"]: float(row["value"]) for _, row in overall_metrics.iterrows()
    }

    print("\n=== Manual URL Test Summary ===")
    print(f"Total URLs tested: {int(metric_lookup['total_urls'])}")
    print(f"Correct predictions: {int(metric_lookup['correct_predictions'])}")
    print(f"Manual test accuracy: {metric_lookup['manual_accuracy']:.4f}")
    print(f"Borderline predictions (0.40-0.60): {int(metric_lookup['borderline_predictions'])}")

    print("\nBreakdown by expected label:")
    print(
        label_breakdown.to_string(
            index=False,
            formatters={
                "average_probability": "{:.4f}".format,
                "accuracy": "{:.4f}".format,
            },
        )
    )

    if not category_breakdown.empty:
        print("\nBreakdown by category:")
        print(
            category_breakdown.to_string(
                index=False,
                formatters={"accuracy": "{:.4f}".format},
            )
        )

    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")

    if not false_positives.empty:
        print("\nFalse positives:")
        print(
            false_positives[
                ["url", "phishing_probability", "risk_level", "note"]
            ].to_string(index=False)
        )

    if not false_negatives.empty:
        print("\nFalse negatives:")
        print(
            false_negatives[
                ["url", "phishing_probability", "risk_level", "note"]
            ].to_string(index=False)
        )

    borderline_df = results_df[results_df["is_borderline"]]
    if not borderline_df.empty:
        print("\nBorderline predictions:")
        print(
            borderline_df[
                ["url", "expected_label", "predicted_label", "phishing_probability"]
            ].to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the saved phishing model against a curated manual URL test set."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the CSV file containing URLs and expected labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the CSV results.",
    )
    args = parser.parse_args()

    manual_df = load_manual_test_set(args.input)
    results_df = build_results_table(manual_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    save_analysis_artifacts(results_df, args.output)

    print_summary(results_df)
    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
