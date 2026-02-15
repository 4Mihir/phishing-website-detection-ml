import re
from urllib.parse import urlparse
import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account",
    "update", "bank", "paypal", "signin"
]


def has_ip_address(url: str) -> int:
    # Check if URL contains an IP address
    ip_pattern = r"(http[s]?://)?(\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(ip_pattern, url)))


def count_digits(url: str) -> int:
    return sum(c.isdigit() for c in url)


def count_special_chars(url: str) -> int:
    return sum(not c.isalnum() for c in url)


def contains_suspicious_keyword(url: str) -> int:
    url_lower = url.lower()
    return int(any(keyword in url_lower for keyword in SUSPICIOUS_KEYWORDS))


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    features["url_length"] = df["url"].apply(len)
    features["num_dots"] = df["url"].apply(lambda x: x.count("."))
    features["num_digits"] = df["url"].apply(count_digits)
    features["num_special_chars"] = df["url"].apply(count_special_chars)
    features["has_ip"] = df["url"].apply(has_ip_address)
    features["has_https"] = df["url"].apply(lambda x: int(x.startswith("https")))
    features["suspicious_keyword"] = df["url"].apply(contains_suspicious_keyword)

    return features
