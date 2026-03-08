import re
from urllib.parse import urlparse

import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "paypal",
    "signin",
]

SHORTENING_SERVICES = [
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "ow.ly",
    "t.co",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "cutt.ly",
    "tiny.cc",
]


def normalise_url(url: str) -> str:
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def safe_urlparse(url: str):
    """
    Parse URL safely. If parsing fails, return an empty parsed result.
    """
    try:
        return urlparse(normalise_url(url))
    except ValueError:
        return urlparse("http://invalid-url")


def has_ip_address(url: str) -> int:
    ip_pattern = r"(http[s]?://)?(\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(ip_pattern, str(url))))


def count_digits(url: str) -> int:
    return sum(char.isdigit() for char in str(url))


def count_special_chars(url: str) -> int:
    return sum(not char.isalnum() for char in str(url))


def contains_suspicious_keyword(url: str) -> int:
    url_lower = str(url).lower()
    return int(any(keyword in url_lower for keyword in SUSPICIOUS_KEYWORDS))


def uses_shortening_service(url: str) -> int:
    url_lower = str(url).lower()
    return int(any(service in url_lower for service in SHORTENING_SERVICES))


def get_hostname(url: str) -> str:
    parsed = safe_urlparse(url)
    return parsed.netloc.lower()


def get_path(url: str) -> str:
    parsed = safe_urlparse(url)
    return parsed.path


def get_query(url: str) -> str:
    parsed = safe_urlparse(url)
    return parsed.query


def count_subdomains(url: str) -> int:
    hostname = get_hostname(url)
    parts = hostname.split(".")
    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    features = pd.DataFrame()

    features["url_length"] = df["url"].apply(lambda x: len(str(x)))
    features["hostname_length"] = df["url"].apply(lambda x: len(get_hostname(x)))
    features["path_length"] = df["url"].apply(lambda x: len(get_path(x)))
    features["query_length"] = df["url"].apply(lambda x: len(get_query(x)))

    features["num_dots"] = df["url"].apply(lambda x: str(x).count("."))
    features["num_hyphens"] = df["url"].apply(lambda x: str(x).count("-"))
    features["num_underscores"] = df["url"].apply(lambda x: str(x).count("_"))
    features["num_slashes"] = df["url"].apply(lambda x: str(x).count("/"))
    features["num_question_marks"] = df["url"].apply(lambda x: str(x).count("?"))
    features["num_equals"] = df["url"].apply(lambda x: str(x).count("="))
    features["num_at_symbols"] = df["url"].apply(lambda x: str(x).count("@"))
    features["num_percent"] = df["url"].apply(lambda x: str(x).count("%"))

    features["num_digits"] = df["url"].apply(count_digits)
    features["num_special_chars"] = df["url"].apply(count_special_chars)
    features["num_subdomains"] = df["url"].apply(count_subdomains)

    features["has_ip"] = df["url"].apply(has_ip_address)
    features["has_https"] = df["url"].apply(
        lambda x: int(str(x).lower().startswith("https"))
    )
    features["suspicious_keyword"] = df["url"].apply(contains_suspicious_keyword)
    features["shortening_service"] = df["url"].apply(uses_shortening_service)

    return features