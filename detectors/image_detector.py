import os
import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

from detectors.face_detector import FaceDetector


class ImageDetector:

    def __init__(self):

        # -------------------------------------------------
        # Project root
        # -------------------------------------------------

        self.root_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.model_path = os.path.join(
            self.root_dir,
            "models",
            "face_deepfake"
        )

        self.processor = None
        self.model = None

        self.model_status = "PENDING"
        self.error_message = ""

        self.face_detector = FaceDetector()

        print("\n======================================")
        print("AI-CyberShield Face Deepfake Detector")
        print("======================================")

        print(
            "Model path:",
            self.model_path
        )

        self.load_model()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load_model(self):

        try:

            # ---------------------------------------------
            # Model directory
            # ---------------------------------------------

            if not os.path.isdir(self.model_path):

                self.error_message = (
                    "Model directory does not exist: "
                    + self.model_path
                )

                print(self.error_message)

                self.model_status = "PENDING"

                return

            # ---------------------------------------------
            # Required files
            # ---------------------------------------------

            config_file = os.path.join(
                self.model_path,
                "config.json"
            )

            safetensors_file = os.path.join(
                self.model_path,
                "model.safetensors"
            )

            pytorch_file = os.path.join(
                self.model_path,
                "pytorch_model.bin"
            )

            processor_file = os.path.join(
                self.model_path,
                "preprocessor_config.json"
            )

            print("\nChecking model files...")

            print(
                "config.json:",
                os.path.exists(config_file)
            )

            print(
                "model.safetensors:",
                os.path.exists(safetensors_file)
            )

            print(
                "pytorch_model.bin:",
                os.path.exists(pytorch_file)
            )

            print(
                "preprocessor_config.json:",
                os.path.exists(processor_file)
            )

            # ---------------------------------------------
            # Check model weights
            # ---------------------------------------------

            if not os.path.exists(safetensors_file):

                if not os.path.exists(pytorch_file):

                    self.error_message = (
                        "No model weights found. "
                        "Expected model.safetensors "
                        "or pytorch_model.bin."
                    )

                    print(
                        self.error_message
                    )

                    self.model_status = "PENDING"

                    return

            # ---------------------------------------------
            # Load processor
            # ---------------------------------------------

            print("\nLoading image processor...")

            self.processor = (
                AutoImageProcessor.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
            )

            print(
                "Image processor loaded."
            )

            # ---------------------------------------------
            # Load AI model
            # ---------------------------------------------

            print(
                "\nLoading AI deepfake model..."
            )

            self.model = (
                AutoModelForImageClassification
                .from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
            )

            # ---------------------------------------------
            # Evaluation mode
            # ---------------------------------------------

            self.model.eval()

            # ---------------------------------------------
            # Status
            # ---------------------------------------------

            self.model_status = "READY"

            print(
                "\nSUCCESS: AI Deepfake Model READY"
            )

        except Exception as e:

            self.model = None
            self.processor = None

            self.model_status = "ERROR"

            self.error_message = str(e)

            print("\nMODEL LOADING ERROR:")
            print(e)

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, uploaded_file):

        result = {

            "prediction": "MODEL PENDING",

            "confidence": 0.0,

            "risk": "UNKNOWN",

            "model_status": self.model_status,

            "faces_detected": 0,

            "error": self.error_message
        }

        # -------------------------------------------------
        # Check model
        # -------------------------------------------------

        if self.model is None:

            return result

        try:

            # -------------------------------------------------
            # Read uploaded image
            # -------------------------------------------------

            image_bytes = uploaded_file.getvalue()

            image = Image.open(
                __import__("io").BytesIO(
                    image_bytes
                )
            ).convert("RGB")

            # -------------------------------------------------
            # Face detection
            # -------------------------------------------------

            face_image, face_count = (
                self.face_detector.detect_and_crop(
                    image_bytes
                )
            )

            result["faces_detected"] = face_count

            # -------------------------------------------------
            # No face
            # -------------------------------------------------

            if face_image is None:

                result["prediction"] = "NO FACE"

                result["risk"] = "UNKNOWN"

                return result

            # -------------------------------------------------
            # Face crop
            # -------------------------------------------------

            face_image = face_image.convert(
                "RGB"
            )

            # -------------------------------------------------
            # AI preprocessing
            # -------------------------------------------------

            inputs = self.processor(
                images=face_image,
                return_tensors="pt"
            )

            # -------------------------------------------------
            # AI prediction
            # -------------------------------------------------

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )[0]

            # -------------------------------------------------
            # Highest probability
            # -------------------------------------------------

            predicted_id = int(
                torch.argmax(
                    probabilities
                ).item()
            )

            confidence = float(
                probabilities[
                    predicted_id
                ].item()
            ) * 100

            # -------------------------------------------------
            # Model label
            # -------------------------------------------------

            id2label = (
                self.model.config.id2label
            )

            model_label = id2label.get(
                predicted_id,
                str(predicted_id)
            ).upper()

            # -------------------------------------------------
            # Normalize label
            # -------------------------------------------------

            if (
                "DEEPFAKE" in model_label
                or
                "FAKE" in model_label
            ):

                prediction = "DEEPFAKE"

            elif "REAL" in model_label:

                prediction = "REAL"

            else:

                if predicted_id == 1:

                    prediction = "DEEPFAKE"

                else:

                    prediction = "REAL"

            # -------------------------------------------------
            # Risk calculation
            # -------------------------------------------------

            if prediction == "DEEPFAKE":

                if confidence >= 90:

                    risk = "CRITICAL"

                elif confidence >= 75:

                    risk = "HIGH"

                elif confidence >= 60:

                    risk = "MEDIUM"

                else:

                    risk = "LOW"

            else:

                if confidence >= 90:

                    risk = "LOW"

                elif confidence >= 75:

                    risk = "LOW-MEDIUM"

                else:

                    risk = "MEDIUM"

            # -------------------------------------------------
            # Final result
            # -------------------------------------------------

            result = {

                "prediction": prediction,

                "confidence": confidence,

                "risk": risk,

                "model_status": "READY",

                "faces_detected": face_count,

                "model_label": model_label,

                "error": ""
            }

            return result

        except Exception as e:

            print(
                "\nPrediction Error:"
            )

            print(e)

            return {

                "prediction": "ERROR",

                "confidence": 0.0,

                "risk": "UNKNOWN",

                "model_status": "ERROR",

                "faces_detected": 0,

                "error": str(e)
            }
