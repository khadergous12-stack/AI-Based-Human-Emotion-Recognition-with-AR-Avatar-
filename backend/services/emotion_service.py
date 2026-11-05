
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path
from backend.models.model_architecture import build_emotion_cnn


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = BASE_DIR / "backend" / "models" / "emotion_custom_cnn_weights.npz"

# Emotion labels must follow the original training class order
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


class EmotionRecognitionService:

    def __init__(self):
        self.model = build_emotion_cnn()
        self._load_trained_weights()

        # OpenCV face detector
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(cascade_path)

        if self.face_detector.empty():
            raise RuntimeError("Could not load OpenCV face detector.")

        print("Emotion Recognition Service initialized!")

    def _load_trained_weights(self):
        """Load trained CNN weights from the NumPy file."""

        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Trained weights not found: {WEIGHTS_PATH}"
            )

        with np.load(WEIGHTS_PATH, allow_pickle=False) as data:

            for index, layer in enumerate(self.model.layers):
                expected_weights = layer.get_weights()

                if not expected_weights:
                    continue

                transferred_weights = [
                    data[f"layer_{index}_weight_{weight_index}"]
                    for weight_index in range(len(expected_weights))
                ]

                for expected, transferred in zip(
                    expected_weights, transferred_weights
                ):
                    if expected.shape != transferred.shape:
                        raise ValueError(
                            f"Weight shape mismatch in {layer.name}: "
                            f"{expected.shape} != {transferred.shape}"
                        )

                layer.set_weights(transferred_weights)

        print("Trained CNN weights loaded successfully!")

    def predict_emotion(self, image):
        """
        Detect the largest face and predict its emotion.

        Args:
            image: OpenCV image in BGR format.

        Returns:
            Dictionary containing emotion, confidence,
            probabilities and face coordinates.
        """

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image supplied.")

        if image.size == 0:
            raise ValueError("Image is empty.")

        if image.ndim == 2:
            gray_image = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )
        else:
            raise ValueError("Expected a grayscale or BGR image.")

        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            raise ValueError("No face detected in the image.")

        # Select the largest detected face
        x, y, width, height = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        # Crop the face
        face = gray_image[y:y + height, x:x + width]

        # Resize to the CNN input dimensions
        face = cv2.resize(
            face,
            (48, 48),
            interpolation=cv2.INTER_AREA
        )

        # Normalize pixel values to 0-1
        face = face.astype("float32") / 255.0

        # Prepare model input: batch, height, width, channels
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        predictions = self.model.predict(face, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_emotion = EMOTIONS[predicted_index]
        confidence = float(predictions[predicted_index])

        probabilities = {
            emotion: float(probability)
            for emotion, probability in zip(
                EMOTIONS, predictions
            )
        }

        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities,
            "face_coordinates": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            }
        }


# Create one reusable service instance
emotion_service = EmotionRecognitionService()
