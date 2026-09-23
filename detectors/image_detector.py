import os
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


class ImageDetector:

    def __init__(self):

        self.model_path = os.path.join(
            "models",
            "face_deepfake"
        )

        self.model = None
        self.processor = None
        self.model_loaded = False

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.load_model()

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    def load_model(self):

        try:

            if not os.path.exists(
                self.model_path
            ):
                return

            self.processor = (
                AutoImageProcessor.from_pretrained(
                    self.model_path
                )
            )

            self.model = (
                AutoModelForImageClassification
                .from_pretrained(
                    self.model_path
                )
            )

            self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True

        except Exception as e:

            print(
                "Image deepfake model error:",
                e
            )

            self.model_loaded = False

    # --------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------

    def read_image(self, uploaded_file):

        uploaded_file.seek(0)

        data = uploaded_file.read()

        array = np.frombuffer(
            data,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        return image

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, uploaded_file):

        image = self.read_image(
            uploaded_file
        )

        if image is None:

            return {
                "prediction": "INVALID IMAGE",
                "confidence": 0,
                "risk": "UNKNOWN",
                "model_status": "ERROR",
                "faces_detected": 0
            }

        # Detect face
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:

            return {
                "prediction": "NO FACE",
                "confidence": 0,
                "risk": "UNKNOWN",
                "model_status": (
                    "READY"
                    if self.model_loaded
                    else "PENDING"
                ),
                "faces_detected": 0
            }

        # Use largest face
        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        face = image[
            y:y+h,
            x:x+w
        ]

        # Model unavailable
        if not self.model_loaded:

            return {
                "prediction": "MODEL PENDING",
                "confidence": 0,
                "risk": "UNKNOWN",
                "model_status": "PENDING",
                "faces_detected": len(faces)
            }

        # BGR → RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            face_rgb
        )

        # Prepare input
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Prediction
        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        prediction_id = int(
            torch.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                prediction_id
            ]
        )

        label = self.model.config.id2label.get(
            prediction_id,
            str(prediction_id)
        )

        label = label.upper()

        # Normalize result
        if (
            "DEEPFAKE" in label
            or "FAKE" in label
        ):

            prediction = "DEEPFAKE"
            risk = "HIGH"

        elif "REAL" in label:

            prediction = "REAL"
            risk = "LOW"

        else:

            prediction = label
            risk = "UNKNOWN"

        return {

            "prediction": prediction,

            "confidence": round(
                confidence * 100,
                2
            ),

            "risk": risk,

            "model_status": "READY",

            "faces_detected": len(faces),

            "model_label": label
        }
