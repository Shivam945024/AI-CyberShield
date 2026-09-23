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

        # AI-CyberShield root directory
        ROOT_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.model_path = os.path.join(
            ROOT_DIR,
            "models",
            "face_deepfake"
        )

        print("\n================================")
        print("IMAGE DEEPFAKE DETECTOR")
        print("================================")

        print("Model path:")
        print(self.model_path)

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

    def load_model(self):

        try:

            # Check folder
            if not os.path.isdir(self.model_path):

                print("❌ Model folder does not exist")
                print(self.model_path)

                return

            print("\nModel folder found.")

            files = os.listdir(self.model_path)

            print("Model files:")
            for file in files:
                print("  ", file)

            # Check required files
            config_file = os.path.join(
                self.model_path,
                "config.json"
            )

            model_file = os.path.join(
                self.model_path,
                "model.safetensors"
            )

            if not os.path.exists(config_file):

                print("❌ config.json missing")

                return

            if not os.path.exists(model_file):

                print("❌ model.safetensors missing")

                return

            print("\nLoading processor...")

            self.processor = (
                AutoImageProcessor
                .from_pretrained(self.model_path)
            )

            print("Processor loaded.")

            print("Loading deepfake model...")

            self.model = (
                AutoModelForImageClassification
                .from_pretrained(self.model_path)
            )

            self.model.to(self.device)

            self.model.eval()

            self.model_loaded = True

            print("\n✅ Deepfake model loaded successfully")
            print("Device:", self.device)

        except Exception as e:

            print("\n❌ MODEL LOADING ERROR")
            print(type(e).__name__)
            print(str(e))

            self.model_loaded = False

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

    def predict(self, uploaded_file):

        image = self.read_image(uploaded_file)

        if image is None:

            return {
                "prediction": "INVALID IMAGE",
                "confidence": 0,
                "risk": "UNKNOWN",
                "model_status": "ERROR",
                "faces_detected": 0
            }

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

        # Largest face
        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        face = image[
            y:y+h,
            x:x+w
        ]

        if not self.model_loaded:

            return {
                "prediction": "MODEL PENDING",
                "confidence": 0,
                "risk": "UNKNOWN",
                "model_status": "PENDING",
                "faces_detected": len(faces)
            }

        # OpenCV BGR → RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(face_rgb)

        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        prediction_id = int(
            torch.argmax(probabilities)
        )

        confidence = float(
            probabilities[prediction_id]
        )

        label = self.model.config.id2label.get(
            prediction_id,
            str(prediction_id)
        ).upper()

        if "DEEPFAKE" in label or "FAKE" in label:

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
