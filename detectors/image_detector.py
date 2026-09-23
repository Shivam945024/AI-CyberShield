import cv2
import numpy as np


class ImageDetector:

    def __init__(self):

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def predict(self, image):

        if image is None:

            return {
                "prediction": "unknown",
                "risk": 0
            }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(faces) > 0:

            return {
                "prediction": "face_detected",
                "risk": 0,
                "faces_detected": len(faces),
                "message": (
                    "Face detected. "
                    "Deepfake classification model "
                    "can be applied here."
                )
            }

        return {
            "prediction": "no_face_detected",
            "risk": 0,
            "faces_detected": 0
        }
