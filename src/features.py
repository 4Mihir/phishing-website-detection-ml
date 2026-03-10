import re
from urllib.parse import urlparse

import pandas as pd
import tldextract


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
    try:
        return urlparse(normalise_url(url))
    except ValueError:
        return urlparse("http://invalid-url")


def get_hostname(url: str) -> str:
    parsed = safe_urlparse(url)
    hostname = parsed.netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def get_path(url: str) -> str:
    parsed = safe_urlparse(url)
    return parsed.path


def get_query(url: str) -> str:
    parsed = safe_urlparse(url)
    return parsed.query


def get_lexical_url(url: str) -> str:
    """
    Build a consistent lexical representation:
    hostname (without www) + path + query
    This avoids scheme-related skew.
    """
    hostname = get_hostname(url)
    path = get_path(url)
    query = get_query(url)

    if query:
        return f"{hostname}{path}?{query}"
    return f"{hostname}{path}"


def has_ip_address(url: str) -> int:
    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(ip_pattern, get_lexical_url(url))))


def count_digits(text: str) -> int:
    return sum(char.isdigit() for char in str(text))


def count_special_chars(text: str) -> int:
    return sum(not char.isalnum() for char in str(text))


def contains_suspicious_keyword(url: str) -> int:
    text = get_lexical_url(url).lower()
    return int(any(keyword in text for keyword in SUSPICIOUS_KEYWORDS))


def uses_shortening_service(url: str) -> int:
    text = get_hostname(url).lower()
    return int(any(service in text for service in SHORTENING_SERVICES))


def count_subdomains(url: str) -> int:
    hostname = get_hostname(url)
    extracted = tldextract.extract(hostname)

    subdomain = extracted.subdomain.strip().lower()
    if not subdomain:
        return 0

    parts = [part for part in subdomain.split(".") if part]
    return len(parts)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    lexical_urls = df["url"].apply(get_lexical_url)

    features["url_length"] = lexical_urls.apply(len)
    features["hostname_length"] = df["url"].apply(lambda x: len(get_hostname(x)))
    features["path_length"] = df["url"].apply(lambda x: len(get_path(x)))
    features["query_length"] = df["url"].apply(lambda x: len(get_query(x)))

    features["num_dots"] = lexical_urls.apply(lambda x: x.count("."))
    features["num_hyphens"] = lexical_urls.apply(lambda x: x.count("-"))
    features["num_underscores"] = lexical_urls.apply(lambda x: x.count("_"))
    features["num_slashes"] = lexical_urls.apply(lambda x: x.count("/"))
    features["num_question_marks"] = lexical_urls.apply(lambda x: x.count("?"))
    features["num_equals"] = lexical_urls.apply(lambda x: x.count("="))
    features["num_at_symbols"] = lexical_urls.apply(lambda x: x.count("@"))
    features["num_percent"] = lexical_urls.apply(lambda x: x.count("%"))

    features["num_digits"] = lexical_urls.apply(count_digits)
    features["num_special_chars"] = lexical_urls.apply(count_special_chars)
    features["num_subdomains"] = df["url"].apply(count_subdomains)

    features["has_ip"] = df["url"].apply(has_ip_address)
    features["suspicious_keyword"] = df["url"].apply(contains_suspicious_keyword)
    features["shortening_service"] = df["url"].apply(uses_shortening_service)

    return features