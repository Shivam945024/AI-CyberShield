import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class EmailDetector:

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._train_model()

    def _get_dataset_path(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "email_dataset.csv"
        )

    def _combine_text(
        self,
        sender,
        subject,
        body
    ):
        return (
            str(sender) + " " +
            str(subject) + " " +
            str(body)
        )

    def _train_model(self):

        path = self._get_dataset_path()

        if not os.path.exists(path):
            return

        df = pd.read_csv(path)

        required = [
            "sender",
            "subject",
            "body",
            "label"
        ]

        if not all(
            column in df.columns
            for column in required
        ):
            return

        df = df.dropna(
            subset=required
        )

        if len(df["label"].unique()) < 2:
            return

        X = [
            self._combine_text(
                row["sender"],
                row["subject"],
                row["body"]
            )
            for _, row in df.iterrows()
        ]

        y = df["label"].astype(str)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000
        )

        X_vectorized = self.vectorizer.fit_transform(X)

        self.model = LogisticRegression(
            max_iter=1000
        )

        self.model.fit(
            X_vectorized,
            y
        )

    def predict(
        self,
        sender="",
        subject="",
        body=""
    ):

        text = self._combine_text(
            sender,
            subject,
            body
        )

        if self.model is None:
            return self._rule_based_detection(text)

        X = self.vectorizer.transform(
            [text]
        )

        prediction = self.model.predict(X)[0]

        try:
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(max(probabilities))
        except Exception:
            confidence = 0.5

        if str(prediction).lower() in [
            "phishing",
            "malicious",
            "spam",
            "suspicious"
        ]:
            risk = confidence * 100
        else:
            risk = (1 - confidence) * 100

        return {
            "prediction": str(prediction),
            "risk": round(risk, 2),
            "confidence": round(
                confidence * 100,
                2
            )
        }

    def _rule_based_detection(
        self,
        text
    ):

        suspicious_words = [
            "urgent",
            "verify",
            "password",
            "otp",
            "bank",
            "payment",
            "winner",
            "prize",
            "free",
            "suspended",
            "click",
            "immediately"
        ]

        text = text.lower()

        matches = [
            word
            for word in suspicious_words
            if word in text
        ]

        risk = min(
            95,
            len(matches) * 10
        )

        if risk >= 30:
            prediction = "phishing"
        else:
            prediction = "benign"

        return {
            "prediction": prediction,
            "risk": risk,
            "matched_keywords": matches
        }
