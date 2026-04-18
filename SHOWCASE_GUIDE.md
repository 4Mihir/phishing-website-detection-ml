# Streamlit Showcase Guide

## Goal

Present the project as a practical machine learning tool that helps identify suspicious URLs using engineered features from the address itself.

## Suggested 3-5 Minute Demo Script

### 1. Opening

"This project is a phishing website detector built with Python, scikit-learn, and Streamlit. The idea is to analyse the structure of a URL and classify it as either legitimate or phishing based on learned patterns."

### 2. Explain the input and output

"The user enters a URL, the model extracts features such as length, number of hyphens, subdomains, suspicious keywords, and other lexical signals, then the app returns both a prediction and a phishing probability."

### 3. Show a legitimate example first

Use one of:

- `https://www.bbc.co.uk`
- `https://www.google.com`
- `https://github.com`
- `https://www.microsoft.com`

Say:

"I start with a well-known legitimate homepage so we can see what a low-risk result looks like. The model predicts legitimate and gives a relatively low phishing probability."

### 4. Show the suspicious example

Use:

- `http://secure-login-paypal-account.com`

Say:

"Now I switch to a phishing-style domain. Even without visiting the website, the URL already contains strong warning signs such as brand impersonation, multiple suspicious keywords, and unusual structure. The model recognises those patterns and raises the phishing probability significantly."

### 5. Close with the value of the project

"The main value of this system is early warning. It gives a fast first-pass assessment from the URL alone, which can support safer browsing and help flag suspicious links before a user clicks them."

## Safe Demo Order

1. Launch the Streamlit app.
2. Click `BBC` in the sidebar.
3. Point out the prediction, probability, and extracted feature values.
4. Click `Google` or `GitHub` to show consistency across legitimate homepages.
5. Click `Suspicious PayPal-style URL`.
6. Highlight the probability jump and explain why it changed.
7. Finish by showing the model snapshot panel on the right.

## What To Point At On Screen

- `Prediction`: the final class decision.
- `Phishing Probability`: useful for explaining borderline cases.
- `Risk Level`: a more user-friendly summary for non-technical audiences.
- `Extracted Feature Values`: evidence that the model is using structured URL characteristics rather than guessing.
- `Model Snapshot`: shows that the model was trained and evaluated, not hand-coded.

## Good Talking Points

- "This model is feature-based, so it is fast and lightweight."
- "It does not need to download the entire website to make a first-pass judgement."
- "The probability output is useful because security decisions are often not just yes or no."
- "One improvement I made was reducing false positives on legitimate homepage URLs."
- "This is a support tool, not a guarantee of absolute safety."

## If You Get Asked About Limitations

Use something like:

"A limitation is that this version uses URL-based features only. That makes it efficient and explainable, but it does not inspect HTML content, screenshots, SSL certificates, or page behaviour. A future version could combine URL features with content-based detection for better real-world coverage."

## If You Get Asked What You Improved

Use something like:

"I improved the project by fixing a false-positive issue in the feature engineering, standardising the training and prediction pipeline, adding evaluation outputs, and making the Streamlit app more suitable for demonstration."

## Backup Plan If The Live Demo Feels Slow

- Start the app before the presentation begins.
- Keep one legitimate URL already loaded in the input box.
- Use the sidebar buttons rather than typing everything manually.
- If needed, show the saved model metrics and feature importance panel while the audience is looking at the screen.

## URLs To Avoid In A Live Demo

- Random unknown phishing links from the internet.
- URLs that require explanation of unrelated topics.
- Anything you would not feel comfortable typing in front of an assessor.

Use the included suspicious example instead, because it is clearly formatted like a phishing URL without needing to visit a real malicious page.
