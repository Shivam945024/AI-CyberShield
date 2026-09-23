import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class NetworkDetector:

    def __init__(self):

        self.model = None

        self._train_model()

    def _get_dataset_path(self):

        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "network_dataset.csv"
        )

    def _train_model(self):

        path = self._get_dataset_path()

        if not os.path.exists(path):
            return

        df = pd.read_csv(path)

        required = [
            "source_port",
            "destination_port",
            "packet_size",
            "protocol",
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

        X = df[
            [
                "source_port",
                "destination_port",
                "packet_size",
                "protocol"
            ]
        ]

        y = df["label"].astype(str)

        categorical_features = [
            "protocol"
        ]

        numeric_features = [
            "source_port",
            "destination_port",
            "packet_size"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "protocol",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_features
                ),
                (
                    "numeric",
                    "passthrough",
                    numeric_features
                )
            ]
        )

        self.model = Pipeline(
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

        self.model.fit(
            X,
            y
        )

    def predict(
        self,
        source_ip="",
        destination_ip="",
        protocol="TCP",
        source_port=0,
        destination_port=0,
        packet_size=0
    ):

        if self.model is None:

            return self._rule_based_detection(
                source_port,
                destination_port,
                packet_size
            )

        data = pd.DataFrame([
            {
                "source_port": source_port,
                "destination_port": destination_port,
                "packet_size": packet_size,
                "protocol": protocol
            }
        ])

        prediction = self.model.predict(
            data
        )[0]

        try:

            probabilities = (
                self.model.predict_proba(data)[0]
            )

            confidence = float(
                max(probabilities)
            )

        except Exception:

            confidence = 0.5

        if str(prediction).lower() in [
            "attack",
            "malicious",
            "intrusion",
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
            ),
            "source_ip": source_ip,
            "destination_ip": destination_ip
        }

    def _rule_based_detection(
        self,
        source_port,
        destination_port,
        packet_size
    ):

        suspicious_ports = [
            4444,
            31337,
            12345,
            6666,
            6667,
            7777,
            9999
        ]

        risk = 0

        if source_port in suspicious_ports:
            risk += 40

        if destination_port in suspicious_ports:
            risk += 40

        if packet_size > 50000:
            risk += 20

        risk = min(
            risk,
            100
        )

        if risk >= 40:
            prediction = "attack"
        else:
            prediction = "benign"

        return {
            "prediction": prediction,
            "risk": risk
        }
