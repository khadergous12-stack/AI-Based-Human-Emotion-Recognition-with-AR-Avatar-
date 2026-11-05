
"use strict";

// ==========================================
// EMOTION CONTROLLER - NEON RUNNER
// ==========================================

const API_URL = "http://127.0.0.1:8000/api/predict";

const CONFIG = {
    minConfidence: 0.48,
    historySize: 5,
    requiredVotes: 3,
    emotionLockMs: 1600,
    predictionInterval: 750,
    actionCooldown: 850,
    maxLives: 3,
    minTransitionVotes: 3
};

// Difficulty controls the obstacle speed and spawn frequency.
const DIFFICULTY_SETTINGS = {
    slow: {
        baseSpeed: 2.7,
        maxSpeed: 4.2,
        spawnInterval: 145,
        levelIncrease: 0.18
    },
    medium: {
        baseSpeed: 3.8,
        maxSpeed: 6.5,
        spawnInterval: 112,
        levelIncrease: 0.28
    },
    fast: {
        baseSpeed: 5.1,
        maxSpeed: 9.5,
        spawnInterval: 85,
        levelIncrease: 0.4
    }
};

const EMOTION_ACTIONS = {
    happy: "right",
    sad: "left",
    surprise: "jump",
    angry: "shield",
    neutral: "stay",
    fear: "left",
    disgust: "right"
};

const EMOTION_EMOJI = {
    happy: "😊",
    sad: "😢",
    angry: "😠",
    neutral: "😐",
    fear: "😨",
    disgust: "🤢",
    surprise: "😲"
};

const LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
];

// ==========================================
// DOM ELEMENTS
// ==========================================

const video = document.getElementById("cameraVideo");
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

const cameraBtn = document.getElementById("cameraBtn");
const startBtn = document.getElementById("startGameBtn");
const pauseBtn = document.getElementById("pauseBtn");
const restartBtn = document.getElementById("restartBtn");

const overlay = document.getElementById("gameOverlay");
const overlayTitle = document.getElementById("overlayTitle");
const overlayMessage = document.getElementById("overlayMessage");

const emotionValue = document.getElementById("emotionValue");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceText = document.getElementById("confidenceText");

const activeEmotion = document.getElementById("activeEmotion");
const stabilityText = document.getElementById("stabilityText");

const connectionStatus = document.getElementById("connectionStatus");
const scannerIndicator = document.getElementById("scannerIndicator");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

const scoreDisplay = document.getElementById("score");
const livesDisplay = document.getElementById("lives");
const levelDisplay = document.getElementById("level");
const gameMessage = document.getElementById("gameMessage");
const difficultyButtons = document.querySelectorAll(".difficulty-btn");

// ==========================================
// STATE
// ==========================================

const state = {
    cameraStream: null,
    cameraActive: false,
    predictionBusy: false,
    predictionTimer: null,

    running: false,
    paused: false,
    gameOver: false,
    difficulty: "slow",
    startingLevel: 1,

    score: 0,
    lives: CONFIG.maxLives,
    level: 1,

    emotion: "neutral",
    candidateEmotion: null,
    emotionHistory: [],
    stableSince: 0,
    lastAcceptedConfidence: 0,

    lastActionTime: 0,
    lastFrameTime: 0,

    obstacles: [],
    particles: [],
    collectibles: [],
    spawnTimer: 0,
    collectibleTimer: 0,
    animationTime: 0,

    player: {
        x: 130,
        y: 365,
        width: 38,
        height: 48,
        velocityY: 0,
        jumping: false,
        shieldUntil: 0,
        invulnerableUntil: 0
    }
};

const FLOOR_Y = 420;
const GRAVITY = 0.62;
const JUMP_POWER = -13;

// ==========================================
// DIFFICULTY SELECTION
// ==========================================

function setDifficulty(level) {
    if (state.running || state.paused) return;
    if (!Object.prototype.hasOwnProperty.call(DIFFICULTY_SETTINGS, level)) return;

    state.difficulty = level;

    difficultyButtons.forEach(button => {
        button.classList.toggle("active", button.dataset.speed === level);
    });

    const messages = {
        slow: "Slow difficulty selected. Start when ready.",
        medium: "Medium difficulty selected. Start when ready.",
        fast: "Fast difficulty selected. Start when ready."
    };

    gameMessage.textContent = messages[level];
    drawGame();
}

function updateDifficultyAvailability() {
    const locked = state.running || state.paused;

    difficultyButtons.forEach(button => {
        button.disabled = locked;
    });
}

// ==========================================
// CAMERA
// ==========================================

async function enableCamera() {
    if (state.cameraActive) {
        stopCamera();
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
        alert("Camera access requires localhost or HTTPS.");
        return;
    }

    try {
        cameraBtn.disabled = true;
        cameraBtn.textContent = "CONNECTING...";

        state.cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = state.cameraStream;
        await video.play();

        state.cameraActive = true;

        video.classList.add("visible");
        cameraPlaceholder.classList.add("hidden");

        cameraBtn.textContent = "STOP CAMERA";
        cameraBtn.disabled = false;

        connectionStatus.classList.add("connected");
        connectionStatus.innerHTML =
            '<span class="status-dot"></span> Connected';

        scannerIndicator.classList.add("active");

        resetEmotionFilter();

        emotionValue.textContent = "Detecting...";
        stabilityText.textContent = "Collecting predictions...";
        gameMessage.textContent = "Camera connected. Keep your face visible.";

        startPredictionLoop();

    } catch (error) {
        console.error("Camera error:", error);

        alert(
            "Unable to access camera. Check browser permissions and close other apps using the camera."
        );

        cameraBtn.textContent = "ENABLE CAMERA";
        cameraBtn.disabled = false;
    }
}

function stopCamera() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
    }

    state.cameraStream = null;
    state.cameraActive = false;

    if (state.predictionTimer) {
        clearTimeout(state.predictionTimer);
        state.predictionTimer = null;
    }

    video.srcObject = null;
    video.classList.remove("visible");
    cameraPlaceholder.classList.remove("hidden");

    cameraBtn.textContent = "ENABLE CAMERA";
    cameraBtn.disabled = false;

    connectionStatus.classList.remove("connected");
    connectionStatus.innerHTML =
        '<span class="status-dot"></span> Offline';

    scannerIndicator.classList.remove("active");

    emotionValue.textContent = "Waiting...";
    confidenceBar.style.width = "0%";
    confidenceText.textContent = "0% confidence";

    resetEmotionFilter();
    updateActiveEmotionDisplay();

    if (state.running) {
        pauseGame();
    }

    gameMessage.textContent = "Camera disconnected.";
}

function resetEmotionFilter() {
    state.emotionHistory = [];
    state.candidateEmotion = null;
    state.stableSince = 0;
    state.lastAcceptedConfidence = 0;
}

// ==========================================
// API PREDICTION
// ==========================================

async function predictEmotion() {
    if (
        !state.cameraActive ||
        state.predictionBusy ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
        return;
    }

    state.predictionBusy = true;

    try {
        const capture = document.createElement("canvas");
        capture.width = video.videoWidth;
        capture.height = video.videoHeight;

        const captureContext = capture.getContext("2d");
        captureContext.drawImage(
            video,
            0,
            0,
            capture.width,
            capture.height
        );

        const blob = await new Promise(resolve => {
            capture.toBlob(resolve, "image/jpeg", 0.75);
        });

        if (!blob) {
            throw new Error("Unable to capture camera frame.");
        }

        const formData = new FormData();
        formData.append("file", blob, "emotion-frame.jpg");

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API returned HTTP ${response.status}`);
        }

        const result = await response.json();
        processPrediction(result);

    } catch (error) {
        console.error("Prediction error:", error);
        gameMessage.textContent =
            "Prediction connection issue. Check your backend.";
    } finally {
        state.predictionBusy = false;
    }
}

function startPredictionLoop() {
    if (!state.cameraActive) return;

    predictEmotion().finally(() => {
        if (state.cameraActive) {
            state.predictionTimer = setTimeout(
                startPredictionLoop,
                CONFIG.predictionInterval
            );
        }
    });
}

// ==========================================
// API RESPONSE NORMALIZATION
// ==========================================

function findEmotion(result) {
    const prediction = result?.prediction || result?.result || result?.data || result || {};
    let emotion = prediction.emotion ?? prediction.label ??
        prediction.predicted_emotion ?? prediction.predicted_class ?? prediction.class_name;
    let confidence = prediction.confidence ?? prediction.score ?? prediction.probability;
    const probabilities = prediction.probabilities ??
        prediction.emotion_probabilities ?? prediction.scores ??
        result?.probabilities ?? result?.emotion_probabilities;

    if (typeof emotion === "number") emotion = LABELS[emotion];

    if (probabilities && typeof probabilities === "object") {
        const entries = Array.isArray(probabilities)
            ? probabilities.map((value, index) => [LABELS[index], value])
            : Object.entries(probabilities);
        const valid = entries.filter(([label, value]) =>
            LABELS.includes(String(label).toLowerCase()) && Number.isFinite(Number(value))
        ).sort((a, b) => Number(b[1]) - Number(a[1]));
        if (valid.length) {
            if (!emotion) emotion = valid[0][0];
            if (confidence === undefined || confidence === null) confidence = valid[0][1];
        }
    }

    if (typeof emotion === "string") emotion = emotion.toLowerCase().trim();
    if (typeof confidence === "string") confidence = Number.parseFloat(confidence);
    confidence = Number(confidence);
    if (Number.isFinite(confidence) && confidence > 1 && confidence <= 100) confidence /= 100;

    return {
        emotion: String(emotion || "").toLowerCase(),
        confidence: Number.isFinite(confidence) ? confidence : 0
    };
}

// ==========================================
// EMOTION STABILITY FILTER
// ==========================================

function processPrediction(result) {
    const { emotion, confidence } = findEmotion(result);
    if (!LABELS.includes(emotion)) {
        stabilityText.textContent = "No valid emotion detected.";
        return;
    }

    const now = Date.now();
    emotionValue.textContent = `${EMOTION_EMOJI[emotion]} ${emotion}`;
    confidenceBar.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
    confidenceText.textContent = `${(confidence * 100).toFixed(1)}% confidence`;

    if (confidence < CONFIG.minConfidence) {
        stabilityText.textContent = "Low confidence — improve lighting and face the camera.";
        return;
    }

    state.emotionHistory.push({ emotion, confidence, time: now });
    if (state.emotionHistory.length > CONFIG.historySize) state.emotionHistory.shift();

    const counts = {};
    const confidenceTotals = {};
    state.emotionHistory.forEach(item => {
        counts[item.emotion] = (counts[item.emotion] || 0) + 1;
        confidenceTotals[item.emotion] = (confidenceTotals[item.emotion] || 0) + item.confidence;
    });

    const winner = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    const needed = Math.min(CONFIG.requiredVotes, state.emotionHistory.length);
    if (!winner || winner[1] < needed) {
        stabilityText.textContent = `Stabilizing expression (${state.emotionHistory.length}/${CONFIG.historySize} samples).`;
        return;
    }

    const nextEmotion = winner[0];
    const avgConfidence = confidenceTotals[nextEmotion] / winner[1];
    if (avgConfidence < CONFIG.minConfidence) {
        stabilityText.textContent = "Waiting for a clearer expression.";
        return;
    }

    if (nextEmotion !== state.emotion) {
        if (state.stableSince && now - state.stableSince < CONFIG.emotionLockMs) {
            stabilityText.textContent = `Holding ${state.emotion} control briefly to prevent flicker.`;
            return;
        }
        state.emotion = nextEmotion;
        state.stableSince = now;
        createParticles(state.player.x + 18, state.player.y + 15, 8, "#2f8cff");
        gameMessage.textContent = `${EMOTION_EMOJI[nextEmotion]} ${nextEmotion.toUpperCase()} CONTROL`;
    }

    state.lastAcceptedConfidence = avgConfidence;
    updateActiveEmotionDisplay();
}

function updateActiveEmotionDisplay() {
    activeEmotion.textContent =
        `${EMOTION_EMOJI[state.emotion]} ${state.emotion}`;

    if (state.stableSince) {
        stabilityText.textContent =
            `Control active · ${(state.lastAcceptedConfidence * 100).toFixed(0)}% confidence`;
    } else {
        stabilityText.textContent =
            "Waiting for stable predictions";
    }
}

// ==========================================
// GAME START / PAUSE / RESTART
// ==========================================

function startGame() {
    if (!state.cameraActive) {
        overlayTitle.textContent = "Camera Required";
        overlayMessage.textContent =
            "Enable your camera before starting the game.";

        overlay.classList.remove("hidden");
        return;
    }

    resetGame();

    state.running = true;
    state.paused = false;
    state.gameOver = false;

    overlay.classList.add("hidden");

    pauseBtn.disabled = false;
    pauseBtn.textContent = "Ⅱ Pause";

    startBtn.textContent = "START GAME";
    updateDifficultyAvailability();

    gameMessage.textContent =
        "Game started! Use your expressions to control the runner.";

    requestAnimationFrame(gameLoop);
}

function pauseGame() {
    if (!state.running) return;

    state.running = false;
    state.paused = true;

    pauseBtn.textContent = "▶ Resume";

    overlayTitle.textContent = "Game Paused";
    overlayMessage.textContent =
        "Take a break. Resume when you're ready.";

    startBtn.textContent = "RESUME GAME";

    overlay.classList.remove("hidden");
    updateDifficultyAvailability();
    gameMessage.textContent = "Game paused.";
}

function resumeGame() {
    if (!state.cameraActive) {
        overlayTitle.textContent = "Camera Required";
        overlayMessage.textContent =
            "Enable the camera to continue.";

        overlay.classList.remove("hidden");
        return;
    }

    state.running = true;
    state.paused = false;

    overlay.classList.add("hidden");
    pauseBtn.textContent = "Ⅱ Pause";
    updateDifficultyAvailability();

    requestAnimationFrame(gameLoop);
}

function resetGame() {
    state.running = false;
    state.paused = false;
    state.gameOver = false;

    state.score = 0;
    state.lives = CONFIG.maxLives;
    state.level = state.startingLevel || 1;

    state.obstacles = [];
    state.particles = [];
    state.collectibles = [];

    state.spawnTimer = 0;
    state.collectibleTimer = 0;
    state.lastFrameTime = 0;
    state.animationTime = 0;

    state.player.x = 130;
    state.player.y = FLOOR_Y - state.player.height;
    state.player.velocityY = 0;
    state.player.jumping = false;
    state.player.shieldUntil = 0;
    state.player.invulnerableUntil = 0;

    state.lastActionTime = 0;

    resetEmotionFilter();

    updateHUD();
    updateActiveEmotionDisplay();
    updateDifficultyAvailability();
    drawGame();
}

function restartGame() {
    if (state.cameraActive) {
        startGame();
    } else {
        resetGame();

        overlayTitle.textContent = "Camera Required";
        overlayMessage.textContent =
            "Enable the camera before starting.";

        overlay.classList.remove("hidden");
    }
}

function endGame() {
    state.running = false;
    state.gameOver = true;

    pauseBtn.disabled = true;
    startBtn.textContent = "PLAY AGAIN";

    overlayTitle.textContent = "Game Over!";

    overlayMessage.textContent =
        `You scored ${Math.floor(state.score)} points. Try again and beat your record!`;

    overlay.classList.remove("hidden");
    updateDifficultyAvailability();

    gameMessage.textContent = "Game over. Select a difficulty or play again.";
}

// ==========================================
// EMOTION-BASED MOVEMENT
// ==========================================

function performEmotionAction(now) {
    if (
        !state.running ||
        now - state.lastActionTime < CONFIG.actionCooldown
    ) {
        return;
    }

    const player = state.player;
    const action = EMOTION_ACTIONS[state.emotion];

    switch (action) {
        case "right":
            player.x = Math.min(470, player.x + 38);
            state.lastActionTime = now;
            createParticles(player.x, player.y + 35, 3, "#2478ed");
            break;

        case "left":
            player.x = Math.max(65, player.x - 38);
            state.lastActionTime = now;
            createParticles(player.x, player.y + 35, 3, "#8171ff");
            break;

        case "jump":
            if (!player.jumping) {
                player.velocityY = JUMP_POWER;
                player.jumping = true;
                state.lastActionTime = now;

                createParticles(
                    player.x + 18,
                    player.y + 40,
                    10,
                    "#2478ed"
                );
            }
            break;

        case "shield":
            player.shieldUntil = now + 1300;
            state.lastActionTime = now;

            createParticles(
                player.x + 18,
                player.y + 20,
                12,
                "#8171ff"
            );
            break;

        case "stay":
        default:
            break;
    }
}

// ==========================================
// GAME LOOP
// ==========================================

function gameLoop(timestamp) {
    if (!state.running) return;

    const delta = state.lastFrameTime
        ? Math.min((timestamp - state.lastFrameTime) / 16.67, 2)
        : 1;

    state.lastFrameTime = timestamp;

    updateGame(timestamp, delta);
    drawGame();

    if (state.running) {
        requestAnimationFrame(gameLoop);
    }
}

function updateGame(now, delta) {
    const player = state.player;

    performEmotionAction(now);

    // Jump physics.
    player.velocityY += GRAVITY * delta;
    player.y += player.velocityY * delta;

    if (player.y >= FLOOR_Y - player.height) {
        player.y = FLOOR_Y - player.height;
        player.velocityY = 0;
        player.jumping = false;
    }

    // Level progression and selected difficulty.
    state.level = (state.startingLevel || 1) + Math.floor(state.score / 150);

    const settings = DIFFICULTY_SETTINGS[state.difficulty];

    const speed = Math.min(
        settings.maxSpeed,
        settings.baseSpeed + (state.level - 1) * settings.levelIncrease
    );

    // Spawn obstacles according to the selected difficulty.
    state.spawnTimer += delta;

    const spawnInterval = Math.max(
        55,
        settings.spawnInterval - (state.level - 1) * 3
    );

    if (state.spawnTimer > spawnInterval) {
        state.obstacles.push({
            x: canvas.width + 30,
            y: FLOOR_Y - 38,
            width: 30,
            height: 38,
            passed: false
        });

        state.spawnTimer = 0;
    }

    // Spawn collectibles.
    state.collectibleTimer += delta;

    if (state.collectibleTimer > 150) {
        state.collectibles.push({
            x: canvas.width + 20,
            y: FLOOR_Y - 95 - Math.random() * 65,
            radius: 10,
            collected: false
        });

        state.collectibleTimer = 0;
    }

    // Obstacle movement and collision.
    state.obstacles.forEach(obstacle => {
        obstacle.x -= speed * delta;

        if (
            !obstacle.passed &&
            obstacle.x + obstacle.width < player.x
        ) {
            obstacle.passed = true;
            state.score += 10;
        }

        const collision =
            player.x + player.width - 8 > obstacle.x &&
            player.x + 8 < obstacle.x + obstacle.width &&
            player.y + player.height - 5 > obstacle.y &&
            player.y + 5 < obstacle.y + obstacle.height;

        if (
            collision &&
            now > player.invulnerableUntil
        ) {
            if (now < player.shieldUntil) {
                obstacle.x = -100;
                state.score += 20;

                createParticles(
                    obstacle.x,
                    obstacle.y,
                    12,
                    "#8171ff"
                );
            } else {
                state.lives--;

                player.invulnerableUntil = now + 1400;
                obstacle.x = -100;

                createParticles(
                    player.x,
                    player.y,
                    15,
                    "#ed5574"
                );

                if (state.lives <= 0) {
                    endGame();
                }
            }
        }
    });

    state.obstacles = state.obstacles.filter(
        obstacle => obstacle.x + obstacle.width > -10
    );

    // Collectibles.
    state.collectibles.forEach(item => {
        item.x -= speed * delta;

        const dx =
            player.x + player.width / 2 - item.x;

        const dy =
            player.y + player.height / 2 - item.y;

        if (
            !item.collected &&
            Math.sqrt(dx * dx + dy * dy) < 30
        ) {
            item.collected = true;
            state.score += 25;

            createParticles(
                item.x,
                item.y,
                12,
                "#f2b83e"
            );
        }
    });

    state.collectibles = state.collectibles.filter(
        item => !item.collected && item.x > -20
    );

    // Survival score.
    state.score += 0.04 * delta;

    // Particle physics.
    state.particles.forEach(particle => {
        particle.x += particle.vx * delta;
        particle.y += particle.vy * delta;
        particle.life -= delta;
    });

    state.particles = state.particles.filter(
        particle => particle.life > 0
    );

    state.animationTime += delta;

    updateHUD();
}

// ==========================================
// HUD
// ==========================================

function updateHUD() {
    scoreDisplay.textContent =
        String(Math.floor(state.score)).padStart(4, "0");

    livesDisplay.textContent =
        "♥ ".repeat(state.lives) +
        "♡ ".repeat(CONFIG.maxLives - state.lives);

    levelDisplay.textContent =
        String(state.level).padStart(2, "0");
}

// ==========================================
// GRAPHICS
// ==========================================

function drawGame() {
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const gradient = ctx.createLinearGradient(0, 0, 0, height);

    gradient.addColorStop(0, "#f1f8ff");
    gradient.addColorStop(1, "#cce6ff");

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    drawStars(width);
    drawGrid(width);
    drawFloor(width, height);
    drawCollectibles();
    drawObstacles();
    drawPlayer();
    drawParticles();

    ctx.fillStyle = "#31577f";
    ctx.font = "bold 12px Inter, sans-serif";

    ctx.fillText(
        `AI CONTROL: ${state.emotion.toUpperCase()}`,
        22,
        28
    );

    if (performance.now() < state.player.shieldUntil) {
        ctx.fillStyle = "#7555d8";
        ctx.fillText("SHIELD ACTIVE", 22, 49);
    }
}

function drawStars(width) {
    for (let i = 0; i < 45; i++) {
        const x = (i * 173) % width;
        const y = (i * 79) % 310;

        ctx.fillStyle = "rgba(36,120,237,.18)";
        ctx.fillRect(x, y, 2, 2);
    }
}

function drawGrid(width) {
    ctx.strokeStyle = "rgba(36,120,237,.09)";
    ctx.lineWidth = 1;

    for (let x = 0; x < width; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, FLOOR_Y);
        ctx.stroke();
    }
}

function drawFloor(width, height) {
    const floorGradient = ctx.createLinearGradient(
        0,
        FLOOR_Y,
        0,
        height
    );

    floorGradient.addColorStop(0, "#8bcaff");
    floorGradient.addColorStop(1, "#d8edff");

    ctx.fillStyle = floorGradient;
    ctx.fillRect(0, FLOOR_Y, width, height - FLOOR_Y);

    ctx.strokeStyle = "#2478ed";
    ctx.shadowColor = "#2478ed";
    ctx.shadowBlur = 12;

    ctx.beginPath();
    ctx.moveTo(0, FLOOR_Y);
    ctx.lineTo(width, FLOOR_Y);
    ctx.stroke();

    ctx.shadowBlur = 0;

    ctx.strokeStyle = "rgba(36,120,237,.18)";

    for (let x = 0; x < width; x += 75) {
        ctx.beginPath();
        ctx.moveTo(x, FLOOR_Y + 12);
        ctx.lineTo(x - 35, height);
        ctx.stroke();
    }
}

function drawPlayer() {
    const player = state.player;
    const now = performance.now();

    if (
        now < player.invulnerableUntil &&
        Math.floor(now / 100) % 2 === 0
    ) {
        return;
    }

    const shielded = now < player.shieldUntil;

    ctx.save();

    if (shielded) {
        ctx.strokeStyle = "#8171ff";
        ctx.lineWidth = 4;
        ctx.shadowColor = "#8171ff";
        ctx.shadowBlur = 20;

        ctx.beginPath();
        ctx.arc(
            player.x + player.width / 2,
            player.y + player.height / 2,
            37,
            0,
            Math.PI * 2
        );
        ctx.stroke();
    }

    // Character body.
    ctx.shadowColor = "#2478ed";
    ctx.shadowBlur = 20;
    ctx.fillStyle = "#2478ed";

    roundRect(
        ctx,
        player.x,
        player.y + 13,
        player.width,
        player.height - 13,
        10
    );

    ctx.fill();

    // Head.
    ctx.fillStyle = "#ffffff";

    ctx.beginPath();
    ctx.arc(
        player.x + 19,
        player.y + 10,
        13,
        0,
        Math.PI * 2
    );
    ctx.fill();

    // Eyes.
    ctx.fillStyle = "#17365f";
    ctx.fillRect(player.x + 13, player.y + 7, 4, 4);
    ctx.fillRect(player.x + 22, player.y + 7, 4, 4);

    // Legs.
    ctx.fillStyle = "#1557c0";
    ctx.fillRect(player.x + 5, player.y + 42, 10, 8);
    ctx.fillRect(player.x + 23, player.y + 42, 10, 8);

    ctx.restore();
}

function drawObstacles() {
    state.obstacles.forEach(obstacle => {
        ctx.save();

        ctx.fillStyle = "#ed5574";
        ctx.shadowColor = "#ed5574";
        ctx.shadowBlur = 18;

        ctx.beginPath();
        ctx.moveTo(obstacle.x, obstacle.y + obstacle.height);
        ctx.lineTo(
            obstacle.x + obstacle.width / 2,
            obstacle.y
        );
        ctx.lineTo(
            obstacle.x + obstacle.width,
            obstacle.y + obstacle.height
        );

        ctx.closePath();
        ctx.fill();

        ctx.restore();
    });
}

function drawCollectibles() {
    state.collectibles.forEach(item => {
        ctx.save();

        ctx.fillStyle = "#f2b83e";
        ctx.shadowColor = "#f2b83e";
        ctx.shadowBlur = 18;

        ctx.beginPath();
        ctx.arc(
            item.x,
            item.y,
            item.radius,
            0,
            Math.PI * 2
        );

        ctx.fill();
        ctx.restore();
    });
}

function drawParticles() {
    state.particles.forEach(particle => {
        ctx.save();

        ctx.globalAlpha = Math.max(0, particle.life / 25);
        ctx.fillStyle = particle.color;

        ctx.shadowColor = particle.color;
        ctx.shadowBlur = 10;

        ctx.fillRect(
            particle.x,
            particle.y,
            4,
            4
        );

        ctx.restore();
    });
}

function createParticles(x, y, count, color = "#2478ed") {
    for (let i = 0; i < count; i++) {
        state.particles.push({
            x,
            y,
            vx: (Math.random() - 0.5) * 7,
            vy: (Math.random() - 0.5) * 7,
            life: 20 + Math.random() * 12,
            color
        });
    }
}

function roundRect(context, x, y, width, height, radius) {
    context.beginPath();
    context.roundRect(x, y, width, height, radius);
}


// ==========================================
// MAIN MENU INTEGRATION
// ==========================================
window.emotionControllerGame = {
    getStatus() {
        return {
            running: state.running,
            paused: state.paused,
            gameOver: state.gameOver,
            score: state.score,
            level: state.level
        };
    },
    setStartingLevel(level) {
        const parsed = Math.max(1, Math.min(5, Number(level) || 1));
        if (state.running) return;
        state.startingLevel = parsed;
        state.level = parsed;
        updateHUD();
    },
    showMenuPause() {
        if (state.running) pauseGame();
    },
    stopCamera() {
        if (state.cameraActive) stopCamera();
    }
};

// ==========================================
// KEYBOARD FALLBACK
// ==========================================

document.addEventListener("keydown", event => {
    if (event.repeat || !state.running) return;

    switch (event.key) {
        case "ArrowRight":
            state.emotion = "happy";
            break;

        case "ArrowLeft":
            state.emotion = "sad";
            break;

        case "ArrowUp":
        case " ":
            event.preventDefault();
            state.emotion = "surprise";
            break;

        case "s":
        case "S":
            state.emotion = "angry";
            break;

        default:
            return;
    }

    performEmotionAction(performance.now());
});

// ==========================================
// BUTTON EVENTS
// ==========================================

cameraBtn.addEventListener("click", enableCamera);

startBtn.addEventListener("click", () => {
    if (state.paused && !state.gameOver) {
        resumeGame();
    } else {
        startGame();
    }
});

pauseBtn.addEventListener("click", () => {
    if (state.running) {
        pauseGame();
    } else if (state.paused) {
        resumeGame();
    }
});

restartBtn.addEventListener("click", restartGame);

difficultyButtons.forEach(button => {
    button.addEventListener("click", () => {
        setDifficulty(button.dataset.speed);
    });
});

// ==========================================
// CLEANUP
// ==========================================

window.addEventListener("beforeunload", () => {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
    }

    if (state.predictionTimer) {
        clearTimeout(state.predictionTimer);
    }
});

// ==========================================
// INITIALIZATION
// ==========================================

setDifficulty("slow");
resetGame();
drawGame();
updateDifficultyAvailability();

console.log("Emotion Controller initialized successfully.");
