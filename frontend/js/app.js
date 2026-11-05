
"use strict";

// ============================================================
// EMOTIONAI - FRONTEND APPLICATION
// CNN predictions remain connected to the existing FastAPI API.
// MediaPipe is used only for the optional visual face overlay.
// ============================================================

const API_URL = "http://127.0.0.1:8000";

// ============================================================
// DOM ELEMENTS
// ============================================================

const $ = id => document.getElementById(id);

const uploadTab = $("uploadTab");
const cameraTab = $("cameraTab");
const uploadSection = $("uploadSection");
const cameraSection = $("cameraSection");

const imageInput = $("imageInput");
const uploadArea = $("uploadArea");
const imagePreview = $("imagePreview");
const previewContainer = $("previewContainer");
const predictButton = $("predictButton");
const removeImage = $("removeImage");

const webcam = $("webcam");
const captureCanvas = $("captureCanvas");
const faceOverlay = $("faceOverlay");
const cameraPlaceholder = $("cameraPlaceholder");
const overlayContext = faceOverlay.getContext("2d");

const startCamera = $("startCamera");
const stopCamera = $("stopCamera");
const cameraStatus = $("cameraStatus");

const emptyResult = $("emptyResult");
const predictionResult = $("predictionResult");

const emotionName = $("emotionName");
const emotionEmoji = $("emotionEmoji");
const confidenceText = $("confidenceText");
const confidenceValue = $("confidenceValue");
const confidenceBar = $("confidenceBar");
const probabilityBars = $("probabilityBars");
const faceCoordinates = $("faceCoordinates");
const faceDetectedTag = $("faceDetectedTag");

const errorMessage = $("errorMessage");
const loadingMessage = $("loadingMessage");

const systemStatus = $("systemStatus");
const sessionState = $("sessionState");
const sessionHint = $("sessionHint");
const framesAnalyzed = $("framesAnalyzed");
const sessionDuration = $("sessionDuration");
const trackingState = $("trackingState");
const trackingHint = $("trackingHint");
const overlayStatus = $("overlayStatus");
const cameraStatusLine = document.querySelector(".camera-status-line");


// ============================================================
// APPLICATION STATE
// ============================================================

let selectedFile = null;
let previewObjectUrl = null;

let cameraStream = null;
let cameraTimer = null;
let sessionTimer = null;
let sessionStartedAt = null;

let cameraRequestInProgress = false;
let isAnalyzing = false;
let sessionFrameCount = 0;

let smoothedProbabilities = null;
let lastFaceCoordinates = null;

const SMOOTHING_ALPHA = 0.35;
const CAMERA_INTERVAL = 1000;

const emotionEmojis = {
    angry: "😠",
    disgust: "🤢",
    fear: "😨",
    happy: "😄",
    neutral: "😐",
    sad: "😢",
    surprise: "😲"
};

const emotionOrder = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
];


// ============================================================
// MEDIAPIPE FACE LANDMARK SCANNER
// ============================================================

// This is a visual-only overlay.
// It does NOT change the image sent to the CNN backend.

const MEDIAPIPE_VERSION = "0.10.14";

const MEDIAPIPE_WASM =
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

const FACE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let faceLandmarker = null;
let scannerReady = false;
let scannerLoading = false;
let scannerFailed = false;

let scannerAnimationId = null;
let lastMeshFrameTime = 0;

const MESH_FRAME_INTERVAL = 1000 / 12;


// Preload the scanner after the page becomes available.
// Camera startup does not wait for this function.

// ============================================================
// DRAWING HELPERS FOR FACIAL LANDMARKS
// ============================================================

function drawLandmarkConnections(
    landmarks,
    connections,
    width,
    height,
    color,
    lineWidth,
    alpha = 1
) {

    if (!connections || !connections.length) return;

    overlayContext.save();

    overlayContext.globalAlpha = alpha;
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = lineWidth;
    overlayContext.lineCap = "round";
    overlayContext.lineJoin = "round";

    overlayContext.shadowColor = color;
    overlayContext.shadowBlur = 5;

    overlayContext.beginPath();

    for (const connection of connections) {

        const start = landmarks[connection.start];
        const end = landmarks[connection.end];

        if (!start || !end) continue;

        // Mirror X because the webcam preview is mirrored.
        const startX = (1 - start.x) * width;
        const startY = start.y * height;

        const endX = (1 - end.x) * width;
        const endY = end.y * height;

        overlayContext.moveTo(startX, startY);
        overlayContext.lineTo(endX, endY);
    }

    overlayContext.stroke();
    overlayContext.restore();
}


function drawLandmarkPoints(
    landmarks,
    width,
    height
) {

    // Only selected points are highlighted.
    // Avoid drawing every landmark to keep the face natural.

    const selectedPoints = [
        33, 133, 159, 145,
        362, 263, 386, 374,
        1, 4, 5, 6,
        61, 291, 13, 14,
        70, 105, 107,
        336, 334, 300,
        10, 152, 234, 454
    ];

    overlayContext.save();

    for (const index of selectedPoints) {

        const point = landmarks[index];

        if (!point) continue;

        const x = (1 - point.x) * width;
        const y = point.y * height;

        overlayContext.beginPath();

        overlayContext.arc(
            x,
            y,
            Math.max(1.4, width * 0.0025),
            0,
            Math.PI * 2
        );

        overlayContext.fillStyle = "rgba(104, 255, 225, 0.9)";
        overlayContext.shadowColor = "#55e8d1";
        overlayContext.shadowBlur = 7;

        overlayContext.fill();
    }

    overlayContext.restore();
}


// ============================================================
// DRAW SUBTLE FULL-FACE SCAN
// ============================================================

function drawFacialOverlay(landmarks) {

    if (
        !landmarks ||
        !webcam.videoWidth ||
        !webcam.videoHeight
    ) {
        return;
    }

    const width = webcam.videoWidth;
    const height = webcam.videoHeight;

    if (
        faceOverlay.width !== width ||
        faceOverlay.height !== height
    ) {
        faceOverlay.width = width;
        faceOverlay.height = height;
    }

    overlayContext.clearRect(0, 0, width, height);

    const faceLandmarkerClass = window.__emotionFaceLandmarker;

    if (!faceLandmarkerClass) {
        return;
    }

    const {
        FACE_LANDMARKS_FACE_OVAL,
        FACE_LANDMARKS_LEFT_EYE,
        FACE_LANDMARKS_RIGHT_EYE,
        FACE_LANDMARKS_LEFT_EYEBROW,
        FACE_LANDMARKS_RIGHT_EYEBROW,
        FACE_LANDMARKS_LIPS,
        FACE_LANDMARKS_NOSE
    } = faceLandmarkerClass;

    const thinLine = Math.max(0.8, width * 0.0018);
    const featureLine = Math.max(1, width * 0.0022);

    // Full face boundary: subtle cyan outline.
    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_FACE_OVAL,
        width,
        height,
        "#65f0d7",
        featureLine,
        0.75
    );

    // Eye contours.
    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_LEFT_EYE,
        width,
        height,
        "#a8fff0",
        featureLine,
        0.82
    );

    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_RIGHT_EYE,
        width,
        height,
        "#a8fff0",
        featureLine,
        0.82
    );

    // Eyebrows.
    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_LEFT_EYEBROW,
        width,
        height,
        "#8cefe3",
        thinLine,
        0.55
    );

    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_RIGHT_EYEBROW,
        width,
        height,
        "#8cefe3",
        thinLine,
        0.55
    );

    // Nose contour.
    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_NOSE,
        width,
        height,
        "#75e8d9",
        thinLine,
        0.55
    );

    // Lip contour.
    drawLandmarkConnections(
        landmarks,
        FACE_LANDMARKS_LIPS,
        width,
        height,
        "#a0f7e9",
        featureLine,
        0.72
    );

    drawLandmarkPoints(landmarks, width, height);
}


// ============================================================
// FACE SCANNER ANIMATION LOOP
// ============================================================

function startFaceScannerLoop() {

    if (
        !scannerReady ||
        !cameraStream ||
        scannerAnimationId
    ) {
        return;
    }

    lastMeshFrameTime = 0;

    function processScannerFrame(timestamp) {

        if (!cameraStream || !scannerReady) {
            scannerAnimationId = null;
            return;
        }

        scannerAnimationId = requestAnimationFrame(
            processScannerFrame
        );

        if (
            timestamp - lastMeshFrameTime <
            MESH_FRAME_INTERVAL
        ) {
            return;
        }

        lastMeshFrameTime = timestamp;

        if (
            webcam.readyState <
            HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
            return;
        }

        try {

            const result = faceLandmarker.detectForVideo(
                webcam,
                timestamp
            );

            if (
                result.faceLandmarks &&
                result.faceLandmarks.length > 0
            ) {

                // Draw only the first detected face.
                drawFacialOverlay(
                    result.faceLandmarks[0]
                );

                trackingState.textContent = "Tracking";
                trackingHint.textContent = "Facial landmarks active";
                overlayStatus.textContent = "FACE TRACKING ACTIVE";

            } else {

                clearFaceOverlay();

                trackingState.textContent = "Searching";
                trackingHint.textContent = "Position your face in view";
                overlayStatus.textContent = "SEARCHING FOR FACE";
            }

        } catch (error) {

            console.warn(
                "Face scanner frame skipped:",
                error
            );
        }
    }

    scannerAnimationId = requestAnimationFrame(
        processScannerFrame
    );
}


function stopFaceScannerLoop() {

    if (scannerAnimationId) {

        cancelAnimationFrame(scannerAnimationId);

        scannerAnimationId = null;
    }

    clearFaceOverlay();
}


function clearFaceOverlay() {

    if (!overlayContext) return;

    overlayContext.clearRect(
        0,
        0,
        faceOverlay.width,
        faceOverlay.height
    );
}


// ============================================================
// INITIALIZE MEDIAPIPE CONNECTION REFERENCES
// ============================================================

// Store the landmark connection constants after import.
// This keeps the drawing functions independent of the loader.

async function initializeFaceScanner() {

    if (scannerReady || scannerLoading || scannerFailed) {
        return;
    }

    scannerLoading = true;

    trackingState.textContent = "Loading";
    trackingHint.textContent = "Preparing visual scanner";

    try {

        const visionModule = await import(
            `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/+esm`
        );

        const { FilesetResolver, FaceLandmarker } = visionModule;

        const vision = await FilesetResolver.forVisionTasks(
            MEDIAPIPE_WASM
        );

        const options = {
            baseOptions: {
                modelAssetPath: FACE_MODEL_URL,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        };

        try {

            faceLandmarker = await FaceLandmarker.createFromOptions(
                vision,
                options
            );

        } catch (gpuError) {

            console.warn("GPU initialization failed. Trying CPU.", gpuError);

            faceLandmarker = await FaceLandmarker.createFromOptions(
                vision,
                {
                    ...options,
                    baseOptions: {
                        modelAssetPath: FACE_MODEL_URL,
                        delegate: "CPU"
                    }
                }
            );
        }

        // Save the official connection definitions for the renderer.
        window.__emotionFaceLandmarker = FaceLandmarker;

        scannerReady = true;
        scannerFailed = false;

        trackingState.textContent = "Ready";
        trackingHint.textContent = "Visual scanner initialized";
        systemStatus.textContent = "Scanner ready";

        if (cameraStream) {
            startFaceScannerLoop();
        }

        console.info("MediaPipe face scanner ready.");

    } catch (error) {

        console.error("MediaPipe initialization failed:", error);

        scannerFailed = true;

        trackingState.textContent = "Unavailable";
        trackingHint.textContent = "OpenCV face detection remains active";

        systemStatus.textContent = "CNN service ready";

    } finally {

        scannerLoading = false;
    }
}


// ============================================================
// TAB SWITCHING
// ============================================================

uploadTab.addEventListener("click", () => {

    stopWebcam();

    uploadTab.classList.add("active");
    cameraTab.classList.remove("active");

    uploadTab.setAttribute("aria-selected", "true");
    cameraTab.setAttribute("aria-selected", "false");

    uploadSection.classList.remove("hidden");
    cameraSection.classList.add("hidden");

    hideError();
});


cameraTab.addEventListener("click", () => {

    uploadTab.classList.remove("active");
    cameraTab.classList.add("active");

    uploadTab.setAttribute("aria-selected", "false");
    cameraTab.setAttribute("aria-selected", "true");

    uploadSection.classList.add("hidden");
    cameraSection.classList.remove("hidden");

    hideError();
});


// ============================================================
// IMAGE UPLOAD
// ============================================================

uploadArea.addEventListener("click", () => {
    imageInput.click();
});


imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {

        showError("Please select a valid image file.");
        imageInput.value = "";
        return;
    }

    if (file.size > 10 * 1024 * 1024) {

        showError("Image size must be below 10 MB.");
        imageInput.value = "";
        return;
    }

    clearPreviewUrl();

    selectedFile = file;
    previewObjectUrl = URL.createObjectURL(file);

    imagePreview.src = previewObjectUrl;

    previewContainer.classList.remove("hidden");
    predictButton.disabled = false;

    resetPrediction();
    hideError();
});


function clearPreviewUrl() {

    if (previewObjectUrl) {

        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }
}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeImage.addEventListener("click", clearSelectedImage);


function clearSelectedImage() {

    clearPreviewUrl();

    selectedFile = null;

    imageInput.value = "";
    imagePreview.src = "";

    previewContainer.classList.add("hidden");
    predictButton.disabled = true;

    resetPrediction();
    hideError();
}


// ============================================================
// IMAGE PREDICTION
// ============================================================

predictButton.addEventListener("click", async () => {

    if (!selectedFile || isAnalyzing) return;

    await predictFile(selectedFile);
});


async function predictFile(file) {

    const formData = new FormData();

    formData.append("file", file);

    setLoading(true);
    hideError();

    try {

        const response = await fetch(
            `${API_URL}/api/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail || "Emotion prediction failed."
            );
        }

        if (!data.prediction) {
            throw new Error("The server returned no prediction data.");
        }

        displayPrediction(data.prediction);

    } catch (error) {

        showError(error.message);

    } finally {

        setLoading(false);
    }
}


// ============================================================
// PREDICTION SMOOTHING
// ============================================================

function smoothPrediction(prediction) {

    const probabilities = prediction.probabilities;

    if (!probabilities) {
        return prediction;
    }

    if (smoothedProbabilities === null) {

        smoothedProbabilities = { ...probabilities };

    } else {

        Object.keys(probabilities).forEach(emotion => {

            smoothedProbabilities[emotion] =
                SMOOTHING_ALPHA * probabilities[emotion] +
                (1 - SMOOTHING_ALPHA) *
                smoothedProbabilities[emotion];

        });
    }

    const [predictedEmotion, predictedProbability] =
        Object.entries(smoothedProbabilities).reduce(
            (best, current) =>
                current[1] > best[1] ? current : best
        );

    return {
        ...prediction,
        emotion: predictedEmotion,
        confidence: Number(
            (predictedProbability * 100).toFixed(2)
        ),
        probabilities: {
            ...smoothedProbabilities
        }
    };
}


// ============================================================
// START WEBCAM
// ============================================================

startCamera.addEventListener("click", async () => {

    if (cameraStream) return;

    hideError();

    if (!navigator.mediaDevices?.getUserMedia) {

        showError(
            "Camera access is unavailable. Open the website through localhost or HTTPS."
        );

        return;
    }

    smoothedProbabilities = null;
    lastFaceCoordinates = null;
    sessionFrameCount = 0;

    framesAnalyzed.textContent = "0";
    sessionState.textContent = "Starting";
    sessionHint.textContent = "Connecting to camera";
    trackingState.textContent = scannerReady ? "Ready" : "Loading";

    clearFaceOverlay();

    try {

        // Start the camera immediately.
        // MediaPipe initialization happens separately.

        cameraStream = await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },

            audio: false
        });

        webcam.srcObject = cameraStream;

        await webcam.play();

        cameraPlaceholder.classList.add("hidden");

        startCamera.disabled = true;
        stopCamera.disabled = false;

        sessionStartedAt = Date.now();

        sessionState.textContent = "Active";
        sessionHint.textContent = "Live camera analysis";
        cameraStatus.textContent = "Camera active. Analysing facial expressions.";

        cameraStatusLine.classList.add("is-active");
        overlayStatus.textContent = "INITIALIZING FACE TRACKING";

        startSessionClock();

        // Start CNN inference without waiting for MediaPipe.
        cameraTimer = setInterval(
            captureAndPredict,
            CAMERA_INTERVAL
        );

        // Start the optional visual scanner if already initialized.
        if (scannerReady) {
            startFaceScannerLoop();
        } else {
            initializeFaceScanner();
        }

    } catch (error) {

        console.error("Camera startup failed:", error);

        stopWebcam();

        showError(
            "Unable to access camera. Check browser permissions and ensure no other application is using it."
        );

        cameraStatus.textContent = "Camera could not be started.";
    }
});


// ============================================================
// SESSION CLOCK
// ============================================================

function startSessionClock() {

    if (sessionTimer) {
        clearInterval(sessionTimer);
    }

    sessionTimer = setInterval(() => {

        if (!sessionStartedAt) return;

        const elapsedSeconds = Math.floor(
            (Date.now() - sessionStartedAt) / 1000
        );

        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;

        sessionDuration.textContent =
            `${String(minutes).padStart(2, "0")}:` +
            `${String(seconds).padStart(2, "0")}`;

    }, 1000);
}


function stopSessionClock() {

    if (sessionTimer) {

        clearInterval(sessionTimer);
        sessionTimer = null;
    }

    sessionStartedAt = null;
}


// ============================================================
// WEBCAM FRAME CAPTURE AND CNN PREDICTION
// ============================================================

async function captureAndPredict() {

    if (
        !cameraStream ||
        webcam.readyState < 2 ||
        cameraRequestInProgress
    ) {
        return;
    }

    cameraRequestInProgress = true;

    try {

        const context = captureCanvas.getContext("2d");

        captureCanvas.width = webcam.videoWidth;
        captureCanvas.height = webcam.videoHeight;

        // IMPORTANT:
        // Capture the original unmirrored frame.
        // The face overlay is a separate visual layer.
        // No scanner graphics are sent to the CNN.

        context.drawImage(
            webcam,
            0,
            0,
            captureCanvas.width,
            captureCanvas.height
        );

        const blob = await new Promise(resolve => {

            captureCanvas.toBlob(
                resolve,
                "image/jpeg",
                0.85
            );

        });

        if (!blob) {
            throw new Error("Could not capture camera frame.");
        }

        const formData = new FormData();

        formData.append(
            "file",
            blob,
            "webcam_frame.jpg"
        );

        const response = await fetch(
            `${API_URL}/api/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {

            if (response.status === 422) {

                clearFaceOverlay();

                smoothedProbabilities = null;
                lastFaceCoordinates = null;

                cameraStatus.textContent =
                    "No face detected. Adjust your position.";

                faceDetectedTag.textContent = "● NOT DETECTED";
                faceDetectedTag.style.color = "#ff7e8d";

                return;
            }

            throw new Error(
                data.detail || "Camera prediction failed."
            );
        }

        if (!data.prediction) {
            throw new Error("The server returned no prediction data.");
        }

        const prediction = data.prediction;

        // Face coordinates are retained for the information panel.
        lastFaceCoordinates = prediction.face_coordinates || null;

        // Smooth CNN probabilities.
        const smoothedResult = smoothPrediction(prediction);

        // Update result panel.
        displayPrediction(smoothedResult);

        // Update session metrics.
        sessionFrameCount += 1;

        framesAnalyzed.textContent =
            sessionFrameCount.toLocaleString();

        sessionState.textContent = "Active";
        sessionHint.textContent = "Live camera analysis";

        faceDetectedTag.textContent = "● DETECTED";
        faceDetectedTag.style.color = "#5ce1a0";

        cameraStatus.textContent =
            `Live prediction: ${smoothedResult.emotion} ` +
            `(${smoothedResult.confidence}%)`;

    } catch (error) {

        console.error("Prediction request failed:", error);

        cameraStatus.textContent = "Prediction request failed.";

        showError(error.message);

    } finally {

        cameraRequestInProgress = false;
    }
}


// ============================================================
// STOP WEBCAM
// ============================================================

stopCamera.addEventListener("click", stopWebcam);


function stopWebcam() {

    if (cameraTimer) {

        clearInterval(cameraTimer);
        cameraTimer = null;
    }

    stopFaceScannerLoop();
    stopSessionClock();

    if (cameraStream) {

        cameraStream.getTracks().forEach(track => {
            track.stop();
        });

        cameraStream = null;
    }

    webcam.srcObject = null;

    cameraPlaceholder.classList.remove("hidden");

    startCamera.disabled = false;
    stopCamera.disabled = true;

    smoothedProbabilities = null;
    lastFaceCoordinates = null;

    cameraRequestInProgress = false;

    clearFaceOverlay();

    sessionState.textContent = "Standby";
    sessionHint.textContent = "Waiting for input";

    sessionDuration.textContent = "00:00";

    trackingState.textContent = scannerReady
        ? "Ready"
        : scannerFailed
            ? "Unavailable"
            : "Standby";

    trackingHint.textContent = scannerReady
        ? "Visual scanner initialized"
        : "Scanner on standby";

    overlayStatus.textContent = "WAITING FOR CAMERA";

    cameraStatusLine.classList.remove("is-active");

    cameraStatus.textContent = "Camera is currently stopped.";
}


// ============================================================
// DISPLAY PREDICTION
// ============================================================

function displayPrediction(prediction) {

    if (!prediction) {

        showError("Prediction data is missing.");
        return;
    }

    const emotion = String(
        prediction.emotion || "neutral"
    ).toLowerCase();

    emptyResult.classList.add("hidden");
    predictionResult.classList.remove("hidden");

    emotionName.textContent =
        emotion.charAt(0).toUpperCase() +
        emotion.slice(1);

    emotionEmoji.textContent =
        emotionEmojis[emotion] || "🙂";

    const confidence = Number(prediction.confidence || 0);

    confidenceText.textContent =
        `${confidence.toFixed(2)}% confidence`;

    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;

    confidenceBar.style.width =
        `${Math.max(0, Math.min(100, confidence))}%`;

    // Render emotion probabilities in a stable order.
    probabilityBars.innerHTML = "";

    const probabilities = prediction.probabilities || {};

    emotionOrder.forEach(name => {

        const probability = Number(probabilities[name] || 0);
        const percentage = Math.max(0, probability * 100);

        const row = document.createElement("div");
        row.className = "probability-row";

        const header = document.createElement("div");
        header.className = "probability-header";

        const label = document.createElement("span");
        label.textContent =
            name.charAt(0).toUpperCase() + name.slice(1);

        const value = document.createElement("span");
        value.textContent = `${percentage.toFixed(2)}%`;

        header.append(label, value);

        const track = document.createElement("div");
        track.className = "probability-track";

        const fill = document.createElement("div");
        fill.className = "probability-fill";

        fill.style.width =
            `${Math.max(0, Math.min(100, percentage))}%`;

        track.appendChild(fill);
        row.append(header, track);

        probabilityBars.appendChild(row);
    });

    // Face coordinates from the existing backend response.
    if (prediction.face_coordinates) {

        const coordinates = prediction.face_coordinates;

        faceCoordinates.textContent =
            `X: ${coordinates.x} | Y: ${coordinates.y} | ` +
            `Width: ${coordinates.width} | ` +
            `Height: ${coordinates.height}`;

        faceDetectedTag.textContent = "● DETECTED";
        faceDetectedTag.style.color = "#5ce1a0";

    } else {

        faceCoordinates.textContent =
            "Face coordinates unavailable.";

        faceDetectedTag.textContent = "● NOT AVAILABLE";
        faceDetectedTag.style.color = "#ffbd70";
    }
}


// ============================================================
// UI HELPERS
// ============================================================

function setLoading(active) {

    isAnalyzing = active;

    loadingMessage.classList.toggle(
        "hidden",
        !active
    );

    predictButton.disabled =
        active || !selectedFile;
}


function showError(message) {

    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


function hideError() {

    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


function resetPrediction() {

    emptyResult.classList.remove("hidden");
    predictionResult.classList.add("hidden");
}


// ============================================================
// BACKEND HEALTH CHECK
// ============================================================

async function checkBackendStatus() {

    try {

        const response = await fetch(
            `${API_URL}/health`,
            {
                method: "GET"
            }
        );

        if (response.ok) {

            systemStatus.textContent = "Backend connected";

        } else {

            systemStatus.textContent = "Backend response issue";
        }

    } catch (error) {

        systemStatus.textContent = "Backend unavailable";

        console.warn(
            "Backend health check failed:",
            error
        );
    }
}


// ============================================================
// INITIALIZATION
// ============================================================

function initializeDashboard() {

    sessionState.textContent = "Standby";
    framesAnalyzed.textContent = "0";
    sessionDuration.textContent = "00:00";

    trackingState.textContent = "Loading";
    trackingHint.textContent = "Preparing visual scanner";

    checkBackendStatus();

    // Load MediaPipe after the 3D intro has closed.
    // This avoids competing with the intro's WebGL rendering for GPU resources.
    const introScreen = document.getElementById("emotion3DIntro");

    if (introScreen) {
        window.addEventListener(
            "emotion-intro-closed",
            initializeFaceScanner,
            { once: true }
        );
    } else {
        initializeFaceScanner();
    }
}


// ============================================================
// CLEANUP
// ============================================================

window.addEventListener("beforeunload", () => {

    stopWebcam();
    clearPreviewUrl();

    if (faceLandmarker) {
        faceLandmarker.close();
        faceLandmarker = null;
    }
});


// Start the application.
initializeDashboard();


// ============================================================
// PREMIUM INTERACTIONS AND PRODUCTIVITY FEATURES
// Append this block at the bottom of app.js
// ============================================================


// ------------------------------------------------------------
// 1. TOAST NOTIFICATIONS
// ------------------------------------------------------------

let toastTimer = null;

function showToast(message, icon = "✦") {

    let toast = document.getElementById("emotionToast");

    if (!toast) {

        toast = document.createElement("div");
        toast.id = "emotionToast";
        toast.className = "emotion-toast";

        document.body.appendChild(toast);
    }

    toast.innerHTML = "";

    const iconElement = document.createElement("span");
    iconElement.className = "emotion-toast-icon";
    iconElement.textContent = icon;

    const messageElement = document.createElement("span");
    messageElement.textContent = message;

    toast.append(iconElement, messageElement);
    toast.classList.add("visible");

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {
        toast.classList.remove("visible");
    }, 2600);
}


// ------------------------------------------------------------
// 2. CURSOR-BASED BACKGROUND LIGHTING
// ------------------------------------------------------------

let cursorFrame = null;

document.addEventListener("pointermove", event => {

    if (cursorFrame) return;

    cursorFrame = requestAnimationFrame(() => {

        document.documentElement.style.setProperty(
            "--mouse-x",
            `${event.clientX}px`
        );

        document.documentElement.style.setProperty(
            "--mouse-y",
            `${event.clientY}px`
        );

        cursorFrame = null;
    });
});


// ------------------------------------------------------------
// 3. CARD LIGHTING FOLLOWS POINTER
// ------------------------------------------------------------

document.querySelectorAll(
    ".metric-card, .panel, .info-card"
).forEach(card => {

    card.addEventListener("pointermove", event => {

        const rect = card.getBoundingClientRect();

        const x = ((event.clientX - rect.left) / rect.width) * 100;
        const y = ((event.clientY - rect.top) / rect.height) * 100;

        card.style.setProperty("--card-x", `${x}%`);
        card.style.setProperty("--card-y", `${y}%`);
    });
});


// ------------------------------------------------------------
// 4. HEADER SCROLL EFFECT
// ------------------------------------------------------------

const premiumTopbar = document.querySelector(".topbar");

function updateHeaderAppearance() {

    if (!premiumTopbar) return;

    premiumTopbar.classList.toggle(
        "scrolled",
        window.scrollY > 25
    );
}

window.addEventListener(
    "scroll",
    updateHeaderAppearance,
    { passive: true }
);

updateHeaderAppearance();


// ------------------------------------------------------------
// 5. SCROLL REVEAL ANIMATIONS
// ------------------------------------------------------------

const revealElements = document.querySelectorAll(
    ".metric-card, .workspace > .panel, .lower-grid > .panel, .disclaimer"
);

if ("IntersectionObserver" in window) {

    const revealObserver = new IntersectionObserver(
        entries => {

            entries.forEach(entry => {

                if (entry.isIntersecting) {

                    entry.target.classList.add("revealed");
                    revealObserver.unobserve(entry.target);
                }
            });
        },
        {
            threshold: 0.08
        }
    );

    revealElements.forEach((element, index) => {

        element.classList.add("reveal-ready");

        element.style.transitionDelay =
            `${Math.min(index % 4, 3) * 70}ms`;

        revealObserver.observe(element);
    });

} else {

    revealElements.forEach(element => {
        element.classList.add("revealed");
    });
}


// ------------------------------------------------------------
// 6. DRAG AND DROP IMAGE UPLOAD
// ------------------------------------------------------------

if (uploadArea) {

    ["dragenter", "dragover"].forEach(eventName => {

        uploadArea.addEventListener(eventName, event => {

            event.preventDefault();
            event.stopPropagation();

            uploadArea.classList.add("drag-over");
        });
    });

    ["dragleave", "dragend", "drop"].forEach(eventName => {

        uploadArea.addEventListener(eventName, event => {

            event.preventDefault();
            event.stopPropagation();

            uploadArea.classList.remove("drag-over");
        });
    });

    uploadArea.addEventListener("drop", event => {

        const files = event.dataTransfer?.files;

        if (!files || files.length === 0) return;

        const file = files[0];

        if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {

            showToast("Please drop a JPG, PNG or WEBP image.", "!");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {

            showToast("Image must be smaller than 10 MB.", "!");
            return;
        }

        // Assign the dropped file to the existing file input.
        const transfer = new DataTransfer();
        transfer.items.add(file);

        imageInput.files = transfer.files;

        imageInput.dispatchEvent(
            new Event("change", { bubbles: true })
        );

        showToast("Image added. Ready for analysis.", "✓");
    });
}


// ------------------------------------------------------------
// 7. ADD REPORT ACTIONS TO THE RESULTS PANEL
// ------------------------------------------------------------

function createReportActions() {

    if (!predictionResult) return;

    if (document.getElementById("resultActions")) return;

    const actions = document.createElement("div");

    actions.id = "resultActions";
    actions.className = "result-actions";

    const copyButton = document.createElement("button");

    copyButton.type = "button";
    copyButton.className = "result-action-button";
    copyButton.textContent = "Copy analysis";

    copyButton.addEventListener("click", copyAnalysisReport);

    const downloadButton = document.createElement("button");

    downloadButton.type = "button";
    downloadButton.className = "result-action-button";
    downloadButton.textContent = "Download report";

    downloadButton.addEventListener(
        "click",
        downloadAnalysisReport
    );

    actions.append(copyButton, downloadButton);

    const faceInfo = predictionResult.querySelector(".face-info");

    if (faceInfo) {
        faceInfo.insertAdjacentElement("afterend", actions);
    } else {
        predictionResult.appendChild(actions);
    }
}


// ------------------------------------------------------------
// 8. GENERATE TEXT REPORT
// ------------------------------------------------------------

function generateAnalysisReport() {

    const emotion = $("emotionName")?.textContent || "Not available";

    const confidence = $("confidenceText")?.textContent || "Not available";

    const coordinates = $("faceCoordinates")?.textContent || "Not available";

    const reportDate = new Date().toLocaleString();

    const reportLines = [
        "EMOTIONAI - FACIAL EMOTION ANALYSIS REPORT",
        "==========================================",
        "",
        `Generated: ${reportDate}`,
        `Detected emotion: ${emotion}`,
        `Confidence score: ${confidence}`,
        "",
        "EMOTION PROBABILITIES",
        "--------------------"
    ];

    document.querySelectorAll(".probability-row").forEach(row => {

        const label = row.querySelector(
            ".probability-header span:first-child"
        )?.textContent;

        const value = row.querySelector(
            ".probability-header span:last-child"
        )?.textContent;

        if (label && value) {
            reportLines.push(`${label}: ${value}`);
        }
    });

    reportLines.push(
        "",
        "FACE DETECTION INFORMATION",
        "--------------------------",
        coordinates,
        "",
        "MODEL INFORMATION",
        "-----------------",
        "Model: Custom Convolutional Neural Network",
        "Emotion classes: 7",
        "Backend: FastAPI",
        "Framework: TensorFlow / Keras",
        "",
        "NOTE",
        "----",
        "Predictions are estimates from facial expressions and",
        "should not be treated as definitive measurements of",
        "a person's internal emotional state."
    );

    return reportLines.join("\n");
}


// ------------------------------------------------------------
// 9. COPY REPORT
// ------------------------------------------------------------

async function copyAnalysisReport() {

    if (predictionResult.classList.contains("hidden")) {
        showToast("Generate an analysis first.", "!");
        return;
    }

    const report = generateAnalysisReport();

    try {

        await navigator.clipboard.writeText(report);

        showToast("Analysis report copied to clipboard.", "✓");

    } catch (error) {

        // Fallback for browsers where Clipboard API is unavailable.
        const textArea = document.createElement("textarea");

        textArea.value = report;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";

        document.body.appendChild(textArea);

        textArea.select();

        const copied = document.execCommand("copy");

        textArea.remove();

        if (copied) {
            showToast("Analysis report copied.", "✓");
        } else {
            showToast("Could not copy report in this browser.", "!");
        }
    }
}


// ------------------------------------------------------------
// 10. DOWNLOAD REPORT
// ------------------------------------------------------------

function downloadAnalysisReport() {

    if (predictionResult.classList.contains("hidden")) {
        showToast("Generate an analysis first.", "!");
        return;
    }

    const report = generateAnalysisReport();

    const blob = new Blob(
        [report],
        { type: "text/plain;charset=utf-8" }
    );

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;
    link.download =
        `EmotionAI_Report_${Date.now()}.txt`;

    document.body.appendChild(link);

    link.click();
    link.remove();

    URL.revokeObjectURL(url);

    showToast("Analysis report downloaded.", "✓");
}


// ------------------------------------------------------------
// 11. CAMERA VISUAL STATE
// ------------------------------------------------------------

const premiumCameraFrame = document.querySelector(".camera-frame");

if (startCamera && premiumCameraFrame) {

    startCamera.addEventListener("click", () => {

        // Camera startup is asynchronous; wait for the video.
        setTimeout(() => {

            if (webcam.srcObject) {
                premiumCameraFrame.classList.add("camera-active");
            }

        }, 700);
    });
}

if (stopCamera && premiumCameraFrame) {

    stopCamera.addEventListener("click", () => {

        premiumCameraFrame.classList.remove("camera-active");
    });
}


// ------------------------------------------------------------
// 12. UPDATE RESULT ACTIONS WHEN A PREDICTION APPEARS
// ------------------------------------------------------------

if (predictionResult) {

    const resultObserver = new MutationObserver(() => {

        if (!predictionResult.classList.contains("hidden")) {
            createReportActions();
        }
    });

    resultObserver.observe(
        predictionResult,
        {
            attributes: true,
            attributeFilter: ["class"],
            childList: true,
            subtree: true
        }
    );
}


// ------------------------------------------------------------
// 13. KEYBOARD SHORTCUTS
// ------------------------------------------------------------

document.addEventListener("keydown", event => {

    const activeElement = document.activeElement;

    const isTyping =
        activeElement &&
        ["INPUT", "TEXTAREA", "SELECT"].includes(activeElement.tagName);

    if (isTyping || event.ctrlKey || event.altKey || event.metaKey) {
        return;
    }

    // U = Image upload mode
    if (event.key.toLowerCase() === "u") {
        uploadTab?.click();
    }

    // C = Camera mode
    if (event.key.toLowerCase() === "c") {
        cameraTab?.click();
    }

    // Escape = Stop camera
    if (event.key === "Escape" && cameraStream) {
        stopWebcam();
    }
});


// ------------------------------------------------------------
// 14. WELCOME MESSAGE
// ------------------------------------------------------------

window.addEventListener("load", () => {

    console.info(
        "%cEmotionAI Premium Dashboard",
        "color:#54e3d0;font-size:16px;font-weight:bold;"
    );

    console.info(
        "Interactive dashboard enhancements initialized."
    );
});
