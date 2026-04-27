# Phishing Website Detection Using Machine Learning

Final Year Project  
BSc Computer Science  
University of West London

## Overview

This project detects phishing websites from URL-based features using supervised machine learning. It includes:

- dataset ingestion and binary label preparation
- feature engineering for lexical and structural URL signals
- training and evaluation for multiple machine learning models
- a Streamlit application for live prediction and project demonstration

## Project Structure

- `data/` - raw and processed datasets
- `models/` - trained machine learning models
- `reports/` - saved metrics, plots, and evaluation outputs
- `src/` - data preparation, feature extraction, training, evaluation, and prediction scripts
- `app.py` - Streamlit showcase application

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## End-to-End Workflow

1. Prepare the binary phishing dataset:

```powershell
python src/data_ingest.py
```

2. Train the models:

```powershell
python src/train.py
```

3. Generate evaluation artifacts:

```powershell
python src/evaluate.py
python src/feature_importance.py
python src/visualise_results.py
```

4. Run CLI predictions:

```powershell
python src/predict.py "https://www.bbc.co.uk"
python src/predict.py "https://www.google.com" "https://github.com"
```

You can also run the curated manual benchmark:

```powershell
python src/manual_url_test.py
```

This benchmark uses a curated 50-URL test set containing both legitimate and phishing-style examples.
It also generates summary CSV files, a benchmark overview chart, and report-ready notes in `reports/`.

5. Launch the Streamlit application:

```powershell
streamlit run app.py
```

## Streamlit Showcase Notes

The Streamlit app is designed for demonstration use and includes:

- quick demo URLs in the sidebar
- live phishing probability scoring
- extracted feature values for explanation
- saved model metrics and feature importance charts

This makes it suitable for both technical explanation and a live walkthrough.

For presentation notes and a suggested live demo script, see `SHOWCASE_GUIDE.md`.

## Example URLs

Legitimate:

- `https://www.bbc.co.uk`
- `https://www.google.com`
- `https://github.com`
- `https://www.microsoft.com`

Suspicious:

- `http://secure-login-paypal-account.com`

## Notes

- The model uses URL features only. It does not inspect full page content, HTML, or screenshots.
- The training pipeline includes synthetic benign root-domain examples to reduce false positives on legitimate homepage URLs.
- The saved Streamlit metrics depend on the latest run of the training and evaluation scripts.
