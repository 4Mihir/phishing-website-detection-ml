import json

import pandas as pd
import streamlit as st

from src.model_utils import REPORTS_DIR, load_model, predict_url


EXAMPLE_URLS = [
    ("BBC", "https://www.bbc.co.uk"),
    ("Google", "https://www.google.com"),
    ("GitHub", "https://github.com"),
    ("Microsoft", "https://www.microsoft.com"),
    ("Suspicious PayPal-style URL", "http://secure-login-paypal-account.com"),
]

DEMO_FLOW = [
    "Start with a trusted homepage such as BBC or Google to show a low-risk result.",
    "Explain that the model uses URL features such as length, hyphens, subdomains, and suspicious words.",
    "Switch to the PayPal-style example to show how the phishing probability rises sharply.",
    "Close by noting that this is a URL-based detector, so it supports decision-making rather than replacing human judgement.",
]


st.set_page_config(
    page_title="Phishing Website Detector",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #f3ede2;
            --bg-bottom: #dce8ef;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-solid: #f8faf8;
            --ink: #142433;
            --muted: #4f6473;
            --accent: #0d6f6d;
            --accent-dark: #0a4f4d;
            --danger: #8a2d2d;
            --safe: #1d6f42;
            --border: rgba(20, 36, 51, 0.12);
            --control-bg: #121722;
            --control-text: #ffffff;
        }

        html, body, [class*="css"] {
            font-family: Georgia, "Times New Roman", serif;
        }

        .stApp {
            background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .hero-card, .panel-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 50px rgba(20, 36, 51, 0.08);
        }

        .hero-title {
            font-size: 2.4rem;
            line-height: 1.1;
            margin: 0 0 0.4rem 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin: 0;
        }

        .stTextInput label, .stTextInput label p {
            color: var(--ink) !important;
            font-weight: 700 !important;
        }

        .stTextInput input {
            background: var(--control-bg) !important;
            color: var(--control-text) !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
        }

        .stTextInput input::placeholder {
            color: rgba(255, 255, 255, 0.68) !important;
        }

        .stButton button, .stFormSubmitButton button {
            background: var(--control-bg) !important;
            color: var(--control-text) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            font-weight: 700 !important;
        }

        .stButton button:hover, .stFormSubmitButton button:hover {
            background: var(--accent-dark) !important;
            color: var(--control-text) !important;
            border-color: rgba(255, 255, 255, 0.34) !important;
        }

        .stButton button p, .stFormSubmitButton button p {
            color: var(--control-text) !important;
        }

        .result-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.95rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.75rem;
        }

        .result-safe {
            background: rgba(29, 111, 66, 0.12);
            color: var(--safe);
            border: 1px solid rgba(29, 111, 66, 0.18);
        }

        .result-danger {
            background: rgba(138, 45, 45, 0.12);
            color: var(--danger);
            border: 1px solid rgba(138, 45, 45, 0.18);
        }

        .small-note {
            color: var(--muted);
            font-size: 0.95rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-solid);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
        }

        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stAlert"] {
            border: 1px solid var(--border);
        }

        div[data-testid="stAlert"] * {
            color: var(--ink) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_model_results():
    path = REPORTS_DIR / "model_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_evaluation_summary():
    path = REPORTS_DIR / "evaluation_summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_feature_importance():
    path = REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def set_example_url(url: str) -> None:
    st.session_state["url_input"] = url


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Demo URLs")
        for label, url in EXAMPLE_URLS:
            if st.button(label, use_container_width=True):
                set_example_url(url)

        st.markdown("---")
        st.markdown("## Run Locally")
        st.code("streamlit run app.py")
        st.markdown(
            "This app uses lexical and structural URL features only. "
            "It does not inspect full web page content."
        )

        st.markdown("---")
        st.markdown("## Suggested Demo Flow")
        for index, step in enumerate(DEMO_FLOW, start=1):
            st.markdown(f"{index}. {step}")


def render_model_snapshot() -> None:
    results_df = load_model_results()
    summary = load_evaluation_summary()
    feature_importance_df = load_feature_importance()

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Model Snapshot")

    if summary is not None:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Accuracy", f"{summary['accuracy']:.3f}")
        metric_cols[1].metric("Recall", f"{summary['recall']:.3f}")
        metric_cols[2].metric("F1-score", f"{summary['f1_score']:.3f}")

    if results_df is not None:
        st.markdown("### Saved Model Comparison")
        st.dataframe(
            results_df.round(4),
            hide_index=True,
            use_container_width=True,
        )

    if feature_importance_df is not None:
        st.markdown("### Top Feature Importance")
        chart_df = feature_importance_df.head(10).set_index("feature")
        st.bar_chart(chart_df)

    st.markdown(
        '<p class="small-note">The random forest model is used for the live prediction view.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_panel() -> None:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Phishing Website Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">A Streamlit showcase for machine-learning based URL risk assessment.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    with st.form("prediction_form"):
        st.text_input(
            "Enter a URL",
            key="url_input",
            placeholder="https://www.bbc.co.uk",
        )
        submitted = st.form_submit_button("Analyse URL", use_container_width=True)

    if submitted:
        try:
            prediction, feature_row = predict_url(st.session_state["url_input"], model=get_model())
            st.session_state["latest_prediction"] = prediction
            st.session_state["latest_features"] = feature_row.to_dict()
            st.session_state["latest_error"] = None
        except Exception as exc:
            st.session_state["latest_error"] = str(exc)

    if st.session_state.get("latest_error"):
        st.error(st.session_state["latest_error"])
        return

    prediction = st.session_state.get("latest_prediction")
    features = st.session_state.get("latest_features")
    if not prediction or not features:
        st.info("Choose a demo URL from the sidebar or enter your own URL to begin.")
        return

    probability = float(prediction["phishing_probability"])
    label = prediction["label"]
    risk_level = prediction["risk_level"]
    badge_class = "result-danger" if label == "PHISHING" else "result-safe"

    st.markdown(
        f'<div class="result-badge {badge_class}">{label}</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Prediction", label)
    metric_cols[1].metric("Phishing Probability", f"{probability:.1%}")
    metric_cols[2].metric("Risk Level", risk_level)

    st.progress(min(max(probability, 0.0), 1.0))
    st.write(f"Normalised URL: `{prediction['normalised_url']}`")

    if label == "PHISHING":
        st.error(
            "The model sees a phishing pattern in this URL. Treat it as suspicious and verify it carefully."
        )
    elif probability >= 0.35:
        st.warning(
            "The model predicts legitimate, but the score is close enough to the decision boundary that you should still verify the destination."
        )
    else:
        st.success("The model predicts this URL as legitimate with a low phishing probability.")

    st.markdown("### Extracted Feature Values")
    feature_df = pd.DataFrame(
        [{"feature": feature_name, "value": feature_value} for feature_name, feature_value in features.items()]
    )
    st.dataframe(feature_df, hide_index=True, use_container_width=True)

    with st.expander("Presenter Talking Points"):
        st.markdown(
            "- The model uses engineered URL features such as hostname length, subdomain count, hyphen count, suspicious keywords, and special-character patterns.\n"
            "- The phishing probability helps explain confidence, not just the final class label.\n"
            "- The training pipeline was adjusted to reduce false positives on legitimate homepage URLs.\n"
            "- This is a practical URL-screening tool for early warning, not a replacement for full browser or content-based analysis."
        )


def main() -> None:
    inject_styles()
    render_sidebar()

    left_col, right_col = st.columns([1.35, 1.0], gap="large")

    with left_col:
        render_prediction_panel()

    with right_col:
        render_model_snapshot()


if __name__ == "__main__":
    main()
