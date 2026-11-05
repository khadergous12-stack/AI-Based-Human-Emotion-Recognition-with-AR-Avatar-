
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/* ================================================
   EMOTION AI - THREE.JS REAL 3D WELCOME EXPERIENCE
================================================ */

(() => {

    if (document.getElementById("emotion3DIntro")) return;

    const MODEL_PATH = "./assets/classical-male-bust.glb";

    // Change this to Math.PI if your model faces backwards.
    const MODEL_ROTATION_Y = 0;

    const intro = document.createElement("section");
    intro.id = "emotion3DIntro";

    intro.innerHTML = `
        <div class="e3d-grid"></div>
        <div class="e3d-glow"></div>

        <button class="e3d-skip" id="e3dSkip">
            SKIP INTRO ↗
        </button>

        <main class="e3d-layout">

            <div class="e3d-stage" id="e3dStage">

                <div class="e3d-orbit"></div>
                <div class="e3d-orbit orbit-b"></div>
                <div class="e3d-orbit orbit-c"></div>

                <div class="e3d-corner tl"></div>
                <div class="e3d-corner tr"></div>
                <div class="e3d-corner bl"></div>
                <div class="e3d-corner br"></div>

                <canvas class="e3d-canvas" id="e3dCanvas"></canvas>

                <div class="e3d-scan"></div>

                <div class="e3d-stage-label">
                    3D FACIAL INTELLIGENCE CORE
                </div>

                <div class="e3d-error" id="e3dError" hidden></div>
            </div>

            <div class="e3d-copy">

                <div class="e3d-eyebrow">
                    NEXT GENERATION ARTIFICIAL INTELLIGENCE
                </div>

                <h1 class="e3d-title">
                    EMOTION
                    <span>AI.</span>
                </h1>

                <p class="e3d-description">
                    Human Emotion Recognition & AR Intelligence.
                    Discover intelligent facial analysis through
                    real-time emotion recognition and augmented reality.
                </p>

                <div class="e3d-status">
                    <span class="e3d-status-dot"></span>
                    <span id="e3dStatus">
                        INITIALIZING 3D INTELLIGENCE CORE
                    </span>
                </div>

                <div class="e3d-progress">
                    <div class="e3d-progress-fill" id="e3dProgressFill"></div>
                </div>

                <div class="e3d-progress-label" id="e3dProgressLabel">
                    LOADING 3D EXPERIENCE...
                </div>

                <button class="e3d-launch" id="e3dLaunch" disabled>
                    ENTER EMOTION AI
                    <span>→</span>
                </button>

            </div>

        </main>

        <div class="e3d-footer">
            <span>HUMAN-CENTRIC ARTIFICIAL INTELLIGENCE</span>
            <span>EMOTION AI INTELLIGENCE SYSTEM</span>
        </div>
    `;

    document.body.appendChild(intro);

    const canvas = document.getElementById("e3dCanvas");
    const stage = document.getElementById("e3dStage");
    const status = document.getElementById("e3dStatus");
    const progressFill = document.getElementById("e3dProgressFill");
    const progressLabel = document.getElementById("e3dProgressLabel");
    const launchButton = document.getElementById("e3dLaunch");
    const errorBox = document.getElementById("e3dError");

    let renderer;
    let scene;
    let camera;
    let modelPivot;
    let animationFrame;
    let resizeObserver;
    let closed = false;
    let modelLoaded = false;
    let clock;

    const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    // -----------------------------------------------
    // THREE.JS SCENE
    // -----------------------------------------------

    try {

        renderer = new THREE.WebGLRenderer({
            canvas,
            alpha: true,
            antialias: true,
            powerPreference: "high-performance"
        });

        renderer.setPixelRatio(
            Math.min(window.devicePixelRatio || 1, 1.75)
        );

        renderer.setSize(stage.clientWidth, stage.clientHeight);

        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.25;

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(
            35,
            stage.clientWidth / stage.clientHeight,
            0.1,
            100
        );

        camera.position.set(0, 0, 4.8);

        clock = new THREE.Clock();

        // Lighting
        scene.add(new THREE.AmbientLight(0x5ab9d8, 1.3));

        const keyLight = new THREE.DirectionalLight(0x8cecff, 2.5);
        keyLight.position.set(2, 3, 4);
        scene.add(keyLight);

        const sideLight = new THREE.PointLight(0x00bfff, 45, 8);
        sideLight.position.set(-2, 0.5, 2);
        scene.add(sideLight);

        const rimLight = new THREE.PointLight(0x007cff, 35, 8);
        rimLight.position.set(1, 1, -2);
        scene.add(rimLight);

        // Rotating holographic rings
        const ringGroup = new THREE.Group();
        scene.add(ringGroup);

        function createRing(radius, opacity, rotationX, rotationY) {

            const geometry = new THREE.TorusGeometry(
                radius, 0.003, 8, 180
            );

            const material = new THREE.MeshBasicMaterial({
                color: 0x00dfff,
                transparent: true,
                opacity,
                depthWrite: false
            });

            const ring = new THREE.Mesh(geometry, material);

            ring.rotation.x = rotationX;
            ring.rotation.y = rotationY;

            ringGroup.add(ring);

            return ring;
        }

        createRing(1.45, 0.30, Math.PI / 2, 0);
        createRing(1.62, 0.16, Math.PI / 2.15, 0.12);
        createRing(1.78, 0.10, Math.PI / 1.8, -0.1);

        // Floating ambient particles
        const particleCount = 180;
        const particlePositions = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {

            particlePositions[i * 3] =
                (Math.random() - 0.5) * 4.8;

            particlePositions[i * 3 + 1] =
                (Math.random() - 0.5) * 4.8;

            particlePositions[i * 3 + 2] =
                (Math.random() - 0.5) * 2.5 - 0.8;
        }

        const particleGeometry = new THREE.BufferGeometry();

        particleGeometry.setAttribute(
            "position",
            new THREE.BufferAttribute(particlePositions, 3)
        );

        const particleMaterial = new THREE.PointsMaterial({
            color: 0x31dfff,
            size: 0.014,
            transparent: true,
            opacity: 0.7,
            sizeAttenuation: true
        });

        const particles = new THREE.Points(
            particleGeometry,
            particleMaterial
        );

        scene.add(particles);

        // -------------------------------------------
        // LOAD REAL HUMAN GLB MODEL
        // -------------------------------------------

        modelPivot = new THREE.Group();
        scene.add(modelPivot);

        const loader = new GLTFLoader();

        status.textContent = "LOADING REAL 3D HUMAN MODEL";

        loader.load(
            MODEL_PATH,

            (gltf) => {

                if (closed) return;

                const model = gltf.scene;

                // Center and scale the model automatically.
                const bounds = new THREE.Box3().setFromObject(model);
                const size = new THREE.Vector3();
                const center = new THREE.Vector3();

                bounds.getSize(size);
                bounds.getCenter(center);

                model.position.sub(center);

                const targetHeight = 2.55;
                const scale = targetHeight / Math.max(size.y, 0.001);

                modelPivot.scale.setScalar(scale);
                modelPivot.rotation.y = MODEL_ROTATION_Y;

                // Apply holographic materials and mesh.
                model.traverse((child) => {

                    if (!child.isMesh) return;

                    child.castShadow = false;
                    child.receiveShadow = false;

                    const hologramMaterial =
                        new THREE.MeshPhysicalMaterial({
                            color: 0x087d9d,
                            emissive: 0x00617b,
                            emissiveIntensity: 1.15,
                            metalness: 0.62,
                            roughness: 0.28,
                            transparent: true,
                            opacity: 0.64,
                            side: THREE.DoubleSide,
                            depthWrite: false,
                            clearcoat: 0.9,
                            clearcoatRoughness: 0.2
                        });

                    child.material = hologramMaterial;

                    // Fine wireframe overlay on actual 3D geometry.
                    const wireframe = new THREE.LineSegments(
                        new THREE.WireframeGeometry(child.geometry),
                        new THREE.LineBasicMaterial({
                            color: 0x39dfff,
                            transparent: true,
                            opacity: 0.32,
                            depthWrite: false
                        })
                    );

                    child.add(wireframe);

                    // Glowing vertices on the model.
                    const points = new THREE.Points(
                        child.geometry,
                        new THREE.PointsMaterial({
                            color: 0x8df4ff,
                            size: 0.009,
                            transparent: true,
                            opacity: 0.58,
                            sizeAttenuation: true
                        })
                    );

                    child.add(points);
                });

                modelPivot.add(model);

                modelLoaded = true;

                progressFill.style.width = "100%";
                progressLabel.textContent = "3D MODEL LOADED — 100%";
                status.textContent = "HOLOGRAPHIC FACE SYSTEM READY";

                launchButton.disabled = false;
                launchButton.classList.add("ready");

            },

            (event) => {

                if (closed) return;

                if (event.total > 0) {

                    const percent = Math.min(
                        99,
                        Math.round((event.loaded / event.total) * 100)
                    );

                    progressFill.style.width = percent + "%";

                    progressLabel.textContent =
                        `LOADING 3D MODEL — ${percent}%`;

                } else {

                    progressLabel.textContent =
                        "DOWNLOADING 3D MODEL...";
                }
            },

            (error) => {

                console.error("Emotion AI 3D model loading failed:", error);

                if (closed) return;

                status.textContent = "3D MODEL COULD NOT BE LOADED";

                progressLabel.textContent =
                    "CHECK YOUR GLB FILE AND NETWORK CONNECTION";

                errorBox.hidden = false;

                errorBox.textContent =
                    "Place classical-male-bust.glb inside frontend/assets. " +
                    "You can still skip the intro.";

                launchButton.disabled = true;
            }
        );

        // -------------------------------------------
        // RESPONSIVE RENDERING
        // -------------------------------------------

        function resizeScene() {

            if (!renderer || closed) return;

            const width = stage.clientWidth;
            const height = stage.clientHeight;

            if (!width || !height) return;

            camera.aspect = width / height;
            camera.updateProjectionMatrix();

            renderer.setSize(width, height, false);
        }

        if ("ResizeObserver" in window) {

            resizeObserver = new ResizeObserver(resizeScene);
            resizeObserver.observe(stage);

        } else {

            window.addEventListener("resize", resizeScene);
        }

        // -------------------------------------------
        // ANIMATION LOOP
        // -------------------------------------------

        function animate() {

            if (closed) return;

            animationFrame = requestAnimationFrame(animate);

            const elapsed = clock.getElapsedTime();

            if (modelLoaded && !reducedMotion) {

                modelPivot.rotation.y =
                    MODEL_ROTATION_Y + Math.sin(elapsed * 0.45) * 0.09;

                modelPivot.rotation.x =
                    Math.sin(elapsed * 0.35) * 0.025;

                modelPivot.position.y =
                    Math.sin(elapsed * 0.8) * 0.025;

            }

            if (!reducedMotion) {
                ringGroup.rotation.z = elapsed * 0.045;
                particles.rotation.y = elapsed * 0.012;
            }

            renderer.render(scene, camera);
        }

        animate();

    } catch (error) {

        console.error("Three.js initialization failed:", error);

        status.textContent = "3D GRAPHICS INITIALIZATION FAILED";

        progressLabel.textContent =
            "YOUR BROWSER MAY NOT SUPPORT WEBGL";

        errorBox.hidden = false;
        errorBox.textContent =
            "Check browser WebGL support and the browser console.";
    }

    // -----------------------------------------------
    // EXIT INTRO
    // -----------------------------------------------

    function closeIntro() {

        if (closed) return;

        closed = true;

        // Reveal the dashboard only when the user exits the intro.
        document.documentElement.classList.remove("emotion-intro-pending");

        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }

        if (resizeObserver) {
            resizeObserver.disconnect();
        } else {
            window.removeEventListener("resize", resizeScene);
        }

        if (renderer) {
            renderer.dispose();
        }

        // Notify the dashboard that the intro has released its WebGL resources.
        window.dispatchEvent(new Event("emotion-intro-closed"));

        intro.classList.add("is-closing");

        window.setTimeout(() => {
            intro.remove();
        }, 1100);
    }

    launchButton.addEventListener("click", () => {
        if (modelLoaded) closeIntro();
    });

    document.getElementById("e3dSkip")
        .addEventListener("click", closeIntro);

    function handleKeys(event) {

        if (event.key === "Escape") {
            closeIntro();
            document.removeEventListener("keydown", handleKeys);
        }

        if (event.key === "Enter" && modelLoaded) {
            closeIntro();
            document.removeEventListener("keydown", handleKeys);
        }
    }

    document.addEventListener("keydown", handleKeys);

})();
