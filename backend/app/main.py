
from datetime import datetime

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.services.emotion_service import emotion_service

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Emotion Recognition API",
    description=(
        "Backend API for AI-Based Human Emotion "
        "Recognition with AR Avatar"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "AI-Based Human Emotion Recognition with AR Avatar",
        "message": "Welcome to the AI Emotion Recognition System",
        "status": "Running",
        "version": "1.0.0"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Emotion Recognition Backend",
        "timestamp": datetime.now().isoformat()
    }


# --------------------------------------------------
# SUPPORTED EMOTIONS
# --------------------------------------------------

@app.get("/api/emotions")
def get_emotions():
    return {
        "supported_emotions": [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise"
        ],
        "total_emotions": 7
    }


# --------------------------------------------------
# EMOTION PREDICTION
# --------------------------------------------------

@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...)):

    # Validate uploaded file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # Read uploaded image
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Decode image bytes using OpenCV
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image."
        )

    # Predict emotion
    try:
        result = emotion_service.predict_emotion(image)

        return {
            "success": True,
            "message": "Emotion prediction completed successfully.",
            "filename": file.filename,
            "prediction": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction."
        )
