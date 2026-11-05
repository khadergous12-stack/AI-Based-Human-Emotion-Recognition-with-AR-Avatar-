
import cv2

from backend.services.emotion_service import emotion_service

# Change this to the path of a real face photograph
IMAGE_PATH = "backend/test_face.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )

result = emotion_service.predict_emotion(image)

print("\n========== EMOTION PREDICTION ==========")
print("Predicted emotion:", result["emotion"])
print("Confidence:", result["confidence"], "%")

print("\nAll emotion probabilities:")

for emotion, probability in result["probabilities"].items():
    print(f"{emotion.capitalize():10s}: {probability * 100:.2f}%")

print("\nFace coordinates:", result["face_coordinates"])
print("========================================")
