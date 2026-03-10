from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from src.features import extract_features


MODEL_PATH = Path("models/random_forest.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )
    return load(MODEL_PATH)


def predict_url(url: str):
    model = load_model()
    input_df = pd.DataFrame({"url": [url]})
    X_input = extract_features(input_df)

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    return label, probability


st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")

st.title("🛡️ Phishing URL Detector")
st.write("Enter a URL to predict whether it is phishing or legitimate.")

url = st.text_input("URL", placeholder="https://example.com")

if st.button("Check URL"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        try:
            label, probability = predict_url(url)

            st.subheader("Prediction Result")
            st.write(f"**URL:** {url}")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Phishing probability:** {probability:.4f}")

            if label == "PHISHING":
                st.error("This URL appears suspicious.")
            else:
                st.success("This URL appears legitimate.")
        except Exception as exc:
            st.error(f"Error: {exc}")