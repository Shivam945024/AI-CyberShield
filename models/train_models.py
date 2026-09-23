import os
import re
import joblib
import pandas as pd

from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# URL MODEL
# ============================================================

def contains_ip(domain):

    pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    return bool(
        re.match(
            pattern,
            domain
        )
    )


def extract_url_features(url):

    url = str(url).lower()

    parsed = urlparse(
        url
        if url.startswith(("http://", "https://"))
        else "http://" + url
    )

    domain = parsed.netloc

    feature_text = " ".join([
        url,
        domain,
        str(len(url)),
        str(url.count(".")),
        str(url.count("-")),
        str(url.count("@")),
        str(url.count("/")),
        str(url.count("?")),
        str(url.count("=")),
        str(url.count("%")),
        "https" if parsed.scheme == "https" else "http",
        "ip" if contains_ip(domain) else "domain"
    ])

    return feature_text


def train_url_model():

    print("\n[1/3] Training URL model...")

    dataset_path = os.path.join(
        DATA_DIR,
        "url_dataset.csv"
    )

    output_path = os.path.join(
        MODEL_DIR,
        "url_model.pkl"
    )

    df = pd.read_csv(
        dataset_path
    )

    df = df.dropna(
        subset=[
            "url",
            "label"
        ]
    )

    X = [
        extract_url_features(url)
        for url in df["url"]
    ]

    y = df["label"].astype(str)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=5000
    )

    X_vectorized = vectorizer.fit_transform(
        X
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_vectorized,
        y
    )

    model_data = {
        "model": model,
        "vectorizer": vectorizer
    }

    joblib.dump(
        model_data,
        output_path
    )

    print(
        f"✅ URL model saved: {output_path}"
    )


# ============================================================
# EMAIL MODEL
# ============================================================

def train_email_model():

    print("\n[2/3] Training Email model...")

    dataset_path = os.path.join(
        DATA_DIR,
        "email_dataset.csv"
    )

    output_path = os.path.join(
        MODEL_DIR,
        "email_model.pkl"
    )

    df = pd.read_csv(
        dataset_path
    )

    df = df.dropna(
        subset=[
            "sender",
            "subject",
            "body",
            "label"
        ]
    )

    X = (
        df["sender"].astype(str)
        + " "
        + df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    )

    y = df["label"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000
    )

    X_vectorized = vectorizer.fit_transform(
        X
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_vectorized,
        y
    )

    model_data = {
        "model": model,
        "vectorizer": vectorizer
    }

    joblib.dump(
        model_data,
        output_path
    )

    print(
        f"✅ Email model saved: {output_path}"
    )


# ============================================================
# NETWORK MODEL
# ============================================================

def train_network_model():

    print("\n[3/3] Training Network model...")

    dataset_path = os.path.join(
        DATA_DIR,
        "network_dataset.csv"
    )

    output_path = os.path.join(
        MODEL_DIR,
        "network_model.pkl"
    )

    df = pd.read_csv(
        dataset_path
    )

    df = df.dropna(
        subset=[
            "source_port",
            "destination_port",
            "packet_size",
            "protocol",
            "label"
        ]
    )

    features = [
        "source_port",
        "destination_port",
        "packet_size",
        "protocol"
    ]

    X = df[features]

    y = df["label"].astype(str)

    numeric_features = [
        "source_port",
        "destination_port",
        "packet_size"
    ]

    categorical_features = [
        "protocol"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features
            ),
            (
                "protocol",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    pipeline.fit(
        X,
        y
    )

    joblib.dump(
        pipeline,
        output_path
    )

    print(
        f"✅ Network model saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🛡️ AI-CyberShield Model Training")
    print("=" * 60)

    train_url_model()

    train_email_model()

    train_network_model()

    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        "models/url_model.pkl"
    )

    print(
        "models/email_model.pkl"
    )

    print(
        "models/network_model.pkl"
    )
