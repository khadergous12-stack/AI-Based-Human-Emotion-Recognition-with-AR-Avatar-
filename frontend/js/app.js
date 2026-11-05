
"use strict";

// ============================================================
// EMOTISENSE - FRONTEND APPLICATION
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
const faceOverlay = document.createElement("canvas");
const cameraPlaceholder = $("cameraPlaceholder");
const overlayContext = faceOverlay.getContext("2d");

// Attach the MediaPipe canvas directly above the mirrored webcam.
// Landmark X coordinates are mirrored in drawLandmarkConnections(), so
// the canvas itself must NOT be CSS-flipped.
faceOverlay.id = "faceOverlay";
faceOverlay.className = "face-overlay";
faceOverlay.setAttribute("aria-hidden", "true");
if (webcam && webcam.parentElement && !document.getElementById("faceOverlay")) {
    webcam.insertAdjacentElement("afterend", faceOverlay);
}

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
let cameraAbortController = null;
let cameraSessionId = 0;
let isAnalyzing = false;
let sessionFrameCount = 0;

let smoothedProbabilities = null;
let lastFaceCoordinates = null;
let smoothedLandmarks = null;

const LANDMARK_SMOOTHING_ALPHA = 0.42;
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

// Map MediaPipe's normalized source-video coordinates into the exact
// visible webcam viewport. This mirrors the X coordinate and applies the
// same object-fit:cover crop as the video element.
function landmarkToViewport(point, width, height) {
    if (!point || !webcam.videoWidth || !webcam.videoHeight) return null;

    const scale = Math.max(
        width / webcam.videoWidth,
        height / webcam.videoHeight
    );
    const renderedWidth = webcam.videoWidth * scale;
    const renderedHeight = webcam.videoHeight * scale;
    const cropX = (renderedWidth - width) / 2;
    const cropY = (renderedHeight - height) / 2;

    const sourceX = point.x * renderedWidth - cropX;
    const sourceY = point.y * renderedHeight - cropY;

    // The webcam preview is mirrored with CSS. Mirror the overlay coordinates
    // here too, rather than applying a second CSS transform to the canvas.
    return {
        x: width - sourceX,
        y: sourceY
    };
}

function drawLandmarkConnections(
    landmarks, connections, width, height, color, lineWidth, alpha = 1
) {
    if (!connections || !connections.length) return;

    overlayContext.save();
    overlayContext.globalAlpha = alpha;
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = lineWidth;
    overlayContext.lineCap = "round";
    overlayContext.lineJoin = "round";
    overlayContext.shadowColor = color;
    overlayContext.shadowBlur = 2;
    overlayContext.beginPath();

    for (const connection of connections) {
        const start = landmarkToViewport(landmarks[connection.start], width, height);
        const end = landmarkToViewport(landmarks[connection.end], width, height);
        if (!start || !end) continue;
        overlayContext.moveTo(start.x, start.y);
        overlayContext.lineTo(end.x, end.y);
    }

    overlayContext.stroke();
    overlayContext.restore();
}

function drawLandmarkPoints(landmarks, width, height) {
    // Only key facial feature points are highlighted; no dense full-face mesh.
    const selectedPoints = [
        33, 133, 159, 145, 362, 263, 386, 374,
        1, 4, 5, 6, 61, 291, 13, 14, 10, 152
    ];

    overlayContext.save();
    for (const index of selectedPoints) {
        const point = landmarkToViewport(landmarks[index], width, height);
        if (!point) continue;

        overlayContext.beginPath();
        overlayContext.arc(point.x, point.y, Math.max(1.15, width * 0.0018), 0, Math.PI * 2);
        overlayContext.fillStyle = "rgba(83, 226, 207, 0.78)";
        overlayContext.shadowColor = "#55d9cb";
        overlayContext.shadowBlur = 3;
        overlayContext.fill();
    }
    overlayContext.restore();
}

// Draw a restrained facial landmark guide aligned to the visible webcam crop.
// This is a visual tracking overlay only; the CNN still predicts emotion.
function drawFacialOverlay(landmarks) {
    if (!landmarks || !webcam.videoWidth || !webcam.videoHeight) return;

    const frame = webcam.closest(".camera-frame");
    if (!frame) return;

    const width = Math.max(1, frame.clientWidth);
    const height = Math.max(1, frame.clientHeight);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    if (
        faceOverlay.width !== Math.round(width * dpr) ||
        faceOverlay.height !== Math.round(height * dpr)
    ) {
        faceOverlay.width = Math.round(width * dpr);
        faceOverlay.height = Math.round(height * dpr);
        faceOverlay.style.width = "100%";
        faceOverlay.style.height = "100%";
    }

    overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
    overlayContext.clearRect(0, 0, width, height);

    const faceLandmarkerClass = window.__emotionFaceLandmarker;
    if (!faceLandmarkerClass) return;

    const {
        FACE_LANDMARKS_FACE_OVAL,
        FACE_LANDMARKS_LEFT_EYE,
        FACE_LANDMARKS_RIGHT_EYE,
        FACE_LANDMARKS_LEFT_EYEBROW,
        FACE_LANDMARKS_RIGHT_EYEBROW,
        FACE_LANDMARKS_LIPS,
        FACE_LANDMARKS_NOSE
    } = faceLandmarkerClass;

    const thinLine = Math.max(0.65, width * 0.0011);
    const featureLine = Math.max(0.8, width * 0.00135);

    // Outer contour only, with a soft low-opacity stroke.
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_FACE_OVAL, width, height,
        "#5ce1d0", featureLine, 0.52
    );

    // Eyes and brows are emphasized gently so the tracking guide is readable
    // without covering the person's face with a dense triangular mask.
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_LEFT_EYE, width, height,
        "#7de9dc", featureLine, 0.72
    );
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_RIGHT_EYE, width, height,
        "#7de9dc", featureLine, 0.72
    );
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_LEFT_EYEBROW, width, height,
        "#78dacc", thinLine, 0.52
    );
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_RIGHT_EYEBROW, width, height,
        "#78dacc", thinLine, 0.52
    );
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_NOSE, width, height,
        "#72d9ce", thinLine, 0.38
    );
    drawLandmarkConnections(
        landmarks, FACE_LANDMARKS_LIPS, width, height,
        "#8de9dd", featureLine, 0.66
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

                // Smooth landmark movement to reduce jitter while keeping
                // the mesh attached to the current facial geometry.
                const rawLandmarks = result.faceLandmarks[0];
                if (!smoothedLandmarks || smoothedLandmarks.length !== rawLandmarks.length) {
                    smoothedLandmarks = rawLandmarks.map(point => ({ ...point }));
                } else {
                    smoothedLandmarks = rawLandmarks.map((point, index) => {
                        const previous = smoothedLandmarks[index];
                        return {
                            x: previous.x + (point.x - previous.x) * LANDMARK_SMOOTHING_ALPHA,
                            y: previous.y + (point.y - previous.y) * LANDMARK_SMOOTHING_ALPHA,
                            z: (previous.z || 0) + ((point.z || 0) - (previous.z || 0)) * LANDMARK_SMOOTHING_ALPHA
                        };
                    });
                }
                drawFacialOverlay(smoothedLandmarks);

                trackingState.textContent = "Tracking";
                trackingHint.textContent = "Face mesh aligned to live landmarks";
                overlayStatus.textContent = "FACE MESH ALIGNED";

            } else {

                smoothedLandmarks = null;
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
    resetPrediction();
    hideError();

    uploadTab.classList.add("active");
    cameraTab.classList.remove("active");

    uploadTab.setAttribute("aria-selected", "true");
    cameraTab.setAttribute("aria-selected", "false");

    uploadSection.classList.remove("hidden");
    cameraSection.classList.add("hidden");

    hideError();
});


cameraTab.addEventListener("click", () => {

    // Do not carry a previous uploaded-image result into webcam mode.
    resetPrediction();
    hideError();
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

        smoothedProbabilities = null;
        displayPrediction(normalizePrediction(data.prediction));

    } catch (error) {

        showError(error.message);

    } finally {

        setLoading(false);
    }
}


// ============================================================
// PREDICTION SMOOTHING
// ============================================================

function normalizePrediction(prediction) {
    if (!prediction || typeof prediction !== "object") {
        throw new Error("The server returned an invalid prediction.");
    }

    const raw = prediction.probabilities;
    if (!raw || typeof raw !== "object") return { ...prediction };

    const values = {};
    emotionOrder.forEach(name => {
        const value = Number(raw[name] ?? 0);
        values[name] = Number.isFinite(value) && value > 0 ? value : 0;
    });

    // Accept either fractional probabilities (0–1) or percentages (0–100).
    const total = Object.values(values).reduce((sum, value) => sum + value, 0);
    if (total > 1.5) {
        Object.keys(values).forEach(name => { values[name] /= 100; });
    }

    const normalizedTotal = Object.values(values).reduce((sum, value) => sum + value, 0);
    if (normalizedTotal > 0) {
        Object.keys(values).forEach(name => { values[name] /= normalizedTotal; });
    }

    const top = Object.entries(values).sort((a, b) => b[1] - a[1])[0];
    const fallbackConfidence = Number(prediction.confidence);
    const confidence = Number.isFinite(fallbackConfidence)
        ? (fallbackConfidence <= 1 ? fallbackConfidence * 100 : fallbackConfidence)
        : (top ? top[1] * 100 : 0);

    return {
        ...prediction,
        emotion: top && normalizedTotal > 0 ? top[0] : String(prediction.emotion || "neutral").toLowerCase(),
        confidence: top && normalizedTotal > 0 ? Number((top[1] * 100).toFixed(2)) : confidence,
        probabilities: values
    };
}

function smoothPrediction(prediction) {
    const normalized = normalizePrediction(prediction);
    const probabilities = normalized.probabilities;

    if (!probabilities) return normalized;

    if (smoothedProbabilities === null) {
        smoothedProbabilities = { ...probabilities };
    } else {
        emotionOrder.forEach(emotion => {
            smoothedProbabilities[emotion] =
                SMOOTHING_ALPHA * (probabilities[emotion] || 0) +
                (1 - SMOOTHING_ALPHA) * (smoothedProbabilities[emotion] || 0);
        });
    }

    const total = Object.values(smoothedProbabilities).reduce((sum, value) => sum + value, 0);
    if (total > 0) {
        Object.keys(smoothedProbabilities).forEach(name => {
            smoothedProbabilities[name] /= total;
        });
    }

    const [predictedEmotion, predictedProbability] = Object.entries(smoothedProbabilities)
        .reduce((best, current) => current[1] > best[1] ? current : best);

    return {
        ...normalized,
        emotion: predictedEmotion,
        confidence: Number((predictedProbability * 100).toFixed(2)),
        probabilities: { ...smoothedProbabilities }
    };
}

// ============================================================
// START WEBCAM
// ============================================================

startCamera.addEventListener("click", async () => {

    if (cameraStream) return;

    hideError();
    resetPrediction();

    if (!navigator.mediaDevices?.getUserMedia) {
        showError("Camera access is unavailable. Open this page using localhost or HTTPS and allow camera permission.");
        return;
    }

    smoothedProbabilities = null;
    lastFaceCoordinates = null;
    smoothedLandmarks = null;
    sessionFrameCount = 0;
    cameraSessionId += 1;

    framesAnalyzed.textContent = "0";
    sessionState.textContent = "Starting";
    sessionHint.textContent = "Requesting camera permission";
    trackingState.textContent = scannerReady ? "Ready" : "Loading";
    trackingHint.textContent = "Preparing visual scanner";
    cameraStatus.textContent = "Waiting for camera permission...";
    overlayStatus.textContent = "CONNECTING CAMERA";
    clearFaceOverlay();
    startCamera.disabled = true;

    try {
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: "user" },
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 24, max: 30 }
                },
                audio: false
            });
        } catch (preferredError) {
            // Retry with browser defaults for webcams that reject ideal constraints.
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }

        // Release the stream if the user navigated away while permission was pending.
        if (!document.getElementById("page-live-analysis")?.classList.contains("active")) {
            stream.getTracks().forEach(track => track.stop());
            startCamera.disabled = false;
            return;
        }

        cameraStream = stream;
        webcam.srcObject = stream;
        webcam.muted = true;
        webcam.playsInline = true;

        await new Promise((resolve, reject) => {
            if (webcam.readyState >= 1 && webcam.videoWidth > 0) return resolve();
            const timeout = setTimeout(() => reject(new Error("Camera did not provide a video stream in time.")), 10000);
            webcam.addEventListener("loadedmetadata", () => { clearTimeout(timeout); resolve(); }, { once: true });
            webcam.addEventListener("error", () => { clearTimeout(timeout); reject(new Error("The browser could not read the camera stream.")); }, { once: true });
        });

        await webcam.play();
        if (!webcam.videoWidth || !webcam.videoHeight) {
            throw new Error("Camera started but no video frames are available. Check camera permission and try again.");
        }

        cameraPlaceholder.classList.add("hidden");
        stopCamera.disabled = false;
        sessionStartedAt = Date.now();
        sessionState.textContent = "Active";
        sessionHint.textContent = "Live camera analysis";
        trackingState.textContent = scannerReady ? "Active" : "Starting";
        trackingHint.textContent = scannerReady ? "Facial landmarks enabled" : "CNN analysis can run while scanner initializes";
        cameraStatus.textContent = "Camera active. Looking for a face...";
        cameraStatusLine.classList.add("is-active");
        overlayStatus.textContent = "CAMERA ACTIVE";
        startSessionClock();

        // First inference immediately, then continue at the configured interval.
        captureAndPredict();
        cameraTimer = setInterval(captureAndPredict, CAMERA_INTERVAL);

        if (scannerReady) startFaceScannerLoop();
        else initializeFaceScanner();

    } catch (error) {
        console.error("Camera startup failed:", error);
        stopWebcam();
        showError(error.message || "Unable to access camera. Check browser permissions and close other apps using the camera.");
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

    if (!cameraStream || webcam.readyState < 2 || !webcam.videoWidth || cameraRequestInProgress) return;

    cameraRequestInProgress = true;
    const requestSessionId = cameraSessionId;
    cameraAbortController = new AbortController();

    try {
        const context = captureCanvas.getContext("2d");
        captureCanvas.width = webcam.videoWidth;
        captureCanvas.height = webcam.videoHeight;
        context.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);

        const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, "image/jpeg", 0.88));
        if (!blob) throw new Error("Could not capture a camera frame.");
        if (requestSessionId !== cameraSessionId || !cameraStream) return;

        const formData = new FormData();
        formData.append("file", blob, "webcam_frame.jpg");

        const response = await fetch(`${API_URL}/api/predict`, {
            method: "POST",
            body: formData,
            signal: cameraAbortController.signal
        });
        const responseText = await response.text();
        let data;
        try {
            data = responseText ? JSON.parse(responseText) : {};
        } catch {
            throw new Error("The prediction API returned an unreadable response. Check the FastAPI server logs.");
        }

        // Ignore responses that arrive after Stop or a new camera session.
        if (requestSessionId !== cameraSessionId || !cameraStream) return;

        if (!response.ok) {
            if (response.status === 422) {
                clearFaceOverlay();
                smoothedProbabilities = null;
                lastFaceCoordinates = null;
                resetPrediction();
                cameraStatus.textContent = "No face detected. Move into the frame and improve lighting.";
                faceDetectedTag.textContent = "● NOT DETECTED";
                faceDetectedTag.style.color = "#d95c70";
                trackingState.textContent = "Searching";
                trackingHint.textContent = "No face found in current frame";
                return;
            }
            throw new Error(data.detail || "Camera prediction failed.");
        }

        const prediction = data.prediction || data.result || data;
        if (!prediction || (!prediction.emotion && !prediction.probabilities)) {
            throw new Error("The server response does not contain an emotion prediction.");
        }

        lastFaceCoordinates = prediction.face_coordinates || null;
        const smoothedResult = smoothPrediction(prediction);
        displayPrediction(smoothedResult);

        sessionFrameCount += 1;
        framesAnalyzed.textContent = sessionFrameCount.toLocaleString();
        sessionState.textContent = "Active";
        sessionHint.textContent = "Live camera analysis";
        faceDetectedTag.textContent = "● DETECTED";
        faceDetectedTag.style.color = "#169b78";
        trackingState.textContent = scannerReady ? "Tracking" : "Detected";
        trackingHint.textContent = scannerReady ? "Facial landmarks active" : "CNN face detection active";
        cameraStatus.textContent = `Live prediction: ${smoothedResult.emotion} (${smoothedResult.confidence}%)`;

    } catch (error) {
        if (error.name !== "AbortError" && requestSessionId === cameraSessionId) {
            console.error("Camera prediction failed:", error);
            cameraStatus.textContent = "Prediction request failed. Check the backend connection.";
            showError(error.message);
        }
    } finally {
        if (requestSessionId === cameraSessionId) {
            cameraRequestInProgress = false;
            cameraAbortController = null;
        }
    }
}

// ============================================================
// STOP WEBCAM
// ============================================================

stopCamera.addEventListener("click", stopWebcam);


function stopWebcam() {

    cameraSessionId += 1;
    if (cameraAbortController) {
        cameraAbortController.abort();
        cameraAbortController = null;
    }
    cameraRequestInProgress = false;

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
    smoothedLandmarks = null;
    sessionFrameCount = 0;
    framesAnalyzed.textContent = "0";
    resetPrediction();
    hideError();

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
        "EMOTISENSE - FACIAL EMOTION ANALYSIS REPORT",
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
        `EmotiSense_Report_${Date.now()}.txt`;

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
        "%cEmotiSense Premium Dashboard",
        "color:#54e3d0;font-size:16px;font-weight:bold;"
    );

    console.info(
        "Interactive dashboard enhancements initialized."
    );
});



// ============================================================
// SINGLE-PAGE SECTION NAVIGATION
// Each top navigation item opens its own full page-like section.
// The analysis controls remain mounted so existing camera/API logic works.
// ============================================================
(() => {
  const navButtons = [...document.querySelectorAll(".main-navigation [data-page]")];
  const views = [...document.querySelectorAll(".page-view[data-view]")];
  function activatePage(name, updateHistory = true) {
    const target = views.find(view => view.dataset.view === name);
    if (!target) return;
    if (name !== "live-analysis" && typeof stopWebcam === "function" && cameraStream) {
      stopWebcam();
    }
    views.forEach(view => {
      const active = view === target;
      view.classList.toggle("active", active);
      view.setAttribute("aria-hidden", String(!active));
    });
    navButtons.forEach(button => {
      const active = button.dataset.page === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-current", active ? "page" : "false");
    });
    if (updateHistory) {
      try { history.replaceState(null, "", `#${name}`); } catch (_) {}
    }
    window.scrollTo({ top: 0, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  navButtons.forEach(button => button.addEventListener("click", () => activatePage(button.dataset.page)));
  const initial = location.hash.replace("#", "");
  activatePage(views.some(v => v.dataset.view === initial) ? initial : "overview", false);
  window.addEventListener("hashchange", () => {
    const page = location.hash.replace("#", "");
    if (views.some(v => v.dataset.view === page)) activatePage(page, false);
  });
})();

// ============================================================
// EMOTISENSE INTRO — ABSTRACT NEURAL NETWORK
// Deliberately contains no face portrait, facial mesh or scan illustration.
// The percentage is a short visual welcome sequence, not model-load progress.
// ============================================================
(() => {
    const intro = document.getElementById("emotion3DIntro");
    const canvas = document.getElementById("networkCanvas");
    if (!intro || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) { intro.remove(); return; }

    const percent = document.getElementById("launchPercent");
    const bar = document.getElementById("launchProgress");
    const status = document.getElementById("launchStatus");
    const message = document.getElementById("launchMessage");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const palette = ["#55a4ee", "#20b9b0", "#8b9cf2", "#b7d9ff"];
    let width = 1, height = 1, raf = 0, closed = false, start = 0, points = [];

    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = Math.max(1, rect.width);
        height = Math.max(1, rect.height);
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const cols = Math.max(6, Math.min(11, Math.round(width / 75)));
        const rows = Math.max(4, Math.min(7, Math.round(height / 65)));
        points = [];
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                points.push({
                    x: width * (0.12 + (c / Math.max(1, cols - 1)) * 0.76),
                    y: height * (0.14 + (r / Math.max(1, rows - 1)) * 0.72),
                    phase: Math.random() * Math.PI * 2,
                    color: palette[(r + c) % palette.length]
                });
            }
        }
    }

    function draw(time) {
        if (closed) return;
        ctx.clearRect(0, 0, width, height);

        const centerX = width / 2, centerY = height / 2;
        const pulse = 0.5 + 0.5 * Math.sin(time * 0.0015);
        const glow = ctx.createRadialGradient(centerX, centerY, 8, centerX, centerY, Math.max(width, height) * 0.48);
        glow.addColorStop(0, "rgba(47, 142, 218, 0.16)");
        glow.addColorStop(1, "rgba(47, 142, 218, 0)");
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, width, height);

        const moved = points.map(p => ({
            x: p.x + Math.sin(time * 0.00055 + p.phase) * 7,
            y: p.y + Math.cos(time * 0.00065 + p.phase) * 7,
            color: p.color
        }));

        // Connect nearby nodes to form an abstract neural network.
        moved.forEach((p, i) => {
            for (let j = i + 1; j < moved.length; j++) {
                const q = moved[j];
                const dx = p.x - q.x, dy = p.y - q.y;
                const distance = Math.hypot(dx, dy);
                if (distance < Math.min(width, height) * 0.24) {
                    ctx.strokeStyle = `rgba(105, 177, 231, ${0.16 * (1 - distance / (Math.min(width, height) * 0.24))})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
                }
            }
        });

        moved.forEach((p, i) => {
            const radius = 2.1 + (i % 5 === 0 ? pulse * 2 : 0);
            ctx.beginPath(); ctx.arc(p.x, p.y, radius + 5, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(51, 157, 218, 0.08)"; ctx.fill();
            ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color; ctx.fill();
        });

        // Central processing node.
        const ringRadius = Math.min(width, height) * (0.105 + pulse * 0.008);
        ctx.beginPath(); ctx.arc(centerX, centerY, ringRadius + 13, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(92, 179, 231, 0.22)"; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.beginPath(); ctx.arc(centerX, centerY, ringRadius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(35, 183, 177, 0.8)"; ctx.lineWidth = 2; ctx.stroke();
        ctx.beginPath(); ctx.arc(centerX, centerY, ringRadius * 0.58, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(52, 132, 207, 0.16)"; ctx.fill();
        ctx.fillStyle = "#eaf5ff";
        ctx.font = `600 ${Math.max(13, Math.min(19, width * 0.025))}px 'DM Sans', sans-serif`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("AI", centerX, centerY);

        if (!reduced) raf = requestAnimationFrame(draw);
    }

    const stages = [
        [0, "PREPARING EXPERIENCE", "Preparing your EmotiSense workspace..."],
        [28, "CONNECTING THE NETWORK", "Establishing visual network connections..."],
        [58, "ACTIVATING AI WORKSPACE", "Bringing the interface online..."],
        [84, "FINALIZING EXPERIENCE", "Almost ready..."],
        [100, "WELCOME TO EMOTISENSE", "Your workspace is ready."]
    ];

    function setProgress(value) {
        if (percent) percent.textContent = `${String(value).padStart(2, "0")}%`;
        if (bar) bar.style.width = `${value}%`;
        const stage = [...stages].reverse().find(item => value >= item[0]);
        if (stage) {
            if (status) status.textContent = stage[1];
            if (message) message.textContent = stage[2];
        }
    }

    function closeIntro() {
        if (closed) return;
        closed = true;
        if (raf) cancelAnimationFrame(raf);
        intro.classList.add("is-closing");
        window.setTimeout(() => {
            intro.remove();
            window.dispatchEvent(new Event("emotion-intro-closed"));
        }, reduced ? 0 : 650);
    }

    resize();
    window.addEventListener("resize", resize, { passive: true });
    if (reduced) {
        setProgress(100);
        draw(0);
        closeIntro();
        return;
    }

    raf = requestAnimationFrame(draw);
    start = performance.now();
    function progressFrame(now) {
        if (closed) return;
        const ratio = Math.min(1, (now - start) / 3000);
        setProgress(Math.round((1 - Math.pow(1 - ratio, 2)) * 100));
        if (ratio < 1) requestAnimationFrame(progressFrame);
        else window.setTimeout(closeIntro, 350);
    }
    requestAnimationFrame(progressFrame);
})();

// Pointer-following sheen for cards (decorative only).
document.addEventListener("pointermove", event => {
    const card = event.target.closest(
        ".metric-card,.panel,.info-card,.wide-card,.overview-intro-band,.tech-card,.goal-card,.team-card,.contact-card"
    );
    if (!card) return;
    const rect = card.getBoundingClientRect();
    card.style.setProperty("--shine-x", `${event.clientX - rect.left}px`);
    card.style.setProperty("--shine-y", `${event.clientY - rect.top}px`);
}, { passive: true });
