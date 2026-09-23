import os
import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class URLDetector:

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._train_model()

    def _get_dataset_path(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "url_dataset.csv"
        )

    def _extract_features(self, urls):
        features = []

        for url in urls:

            url = str(url).lower()

            parsed = urlparse(
                url if url.startswith(("http://", "https://"))
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
                "ip" if self._contains_ip(domain) else "domain"
            ])

            features.append(feature_text)

        return features

    @staticmethod
    def _contains_ip(domain):
        pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
        return bool(re.match(pattern, domain))

    def _train_model(self):

        path = self._get_dataset_path()

        if not os.path.exists(path):
            return

        df = pd.read_csv(path)

        if "url" not in df.columns or "label" not in df.columns:
            return

        df = df.dropna(
            subset=["url", "label"]
        )

        if len(df["label"].unique()) < 2:
            return

        X = self._extract_features(
            df["url"].tolist()
        )

        y = df["label"].astype(str)

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=5000
        )

        X_vectorized = self.vectorizer.fit_transform(X)

        self.model = LogisticRegression(
            max_iter=1000
        )

        self.model.fit(
            X_vectorized,
            y
        )

    def predict(self, url):

        if not url:
            return {
                "prediction": "Unknown",
                "risk": 0
            }

        if self.model is None:
            return self._rule_based_detection(url)

        features = self._extract_features([url])

        X = self.vectorizer.transform(features)

        prediction = self.model.predict(X)[0]

        try:
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(max(probabilities))
        except Exception:
            confidence = 0.5

        if str(prediction).lower() in [
            "phishing",
            "malicious",
            "unsafe"
        ]:
            risk = round(confidence * 100, 2)
        else:
            risk = round((1 - confidence) * 100, 2)

        return {
            "prediction": str(prediction),
            "risk": risk,
            "confidence": round(confidence * 100, 2)
        }

    def _rule_based_detection(self, url):

        suspicious_words = [
            "login",
            "verify",
            "verification",
            "password",
            "account",
            "secure",
            "update",
            "payment",
            "bank",
            "free",
            "winner",
            "prize",
            "urgent"
        ]

        url_lower = url.lower()

        matches = [
            word for word in suspicious_words
            if word in url_lower
        ]

        risk = min(
            95,
            len(matches) * 15
        )

        if risk >= 30:
            prediction = "suspicious"
        else:
            prediction = "benign"

        return {
            "prediction": prediction,
            "risk": risk,
            "matched_keywords": matches
        }
