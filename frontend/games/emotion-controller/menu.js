(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const screens = ["welcomeScreen", "mainMenuScreen", "levelsScreen", "gameScreen"];
  let playerName = "PLAYER";
  let selectedLevel = 1;

  function showScreen(id) {
    screens.forEach(screenId => $(screenId).classList.toggle("hidden", screenId !== id));
  }
  function gameStatus() {
    return window.emotionControllerGame?.getStatus?.() || {running:false, paused:false, gameOver:false};
  }
  function refreshMenu() {
    const status = gameStatus();
    $("menuResumeBtn").disabled = !status.paused;
  }
  function enterMenu() {
    const entered = $("playerName").value.trim();
    if (entered) playerName = entered.slice(0, 18);
    if (!playerName.trim()) playerName = "PLAYER";
    $("menuPlayerName").textContent = playerName.toUpperCase();
    $("gamePlayerName").textContent = playerName.toUpperCase();
    showScreen("mainMenuScreen");
    refreshMenu();
  }
  function openGame() {
    $("gamePlayerName").textContent = playerName.toUpperCase();
    showScreen("gameScreen");
  }
  function goToMenu() {
    const status = gameStatus();
    if (status.running) window.emotionControllerGame?.showMenuPause?.();
    // Stop the camera when leaving gameplay for privacy.
    window.emotionControllerGame?.stopCamera?.();
    showScreen("mainMenuScreen");
    refreshMenu();
  }

  $("enterGameBtn").addEventListener("click", enterMenu);
  $("playerName").addEventListener("keydown", event => {
    if (event.key === "Enter") enterMenu();
  });
  $("menuStartBtn").addEventListener("click", () => showScreen("levelsScreen"));
  $("menuLevelsBtn").addEventListener("click", () => showScreen("levelsScreen"));
  $("menuResumeBtn").addEventListener("click", () => {
    const status = gameStatus();
    if (!status.paused) return;
    openGame();
    $("overlayTitle").textContent = "Game paused";
    $("overlayMessage").textContent = "Enable your camera, then press Resume to continue.";
    $("gameOverlay").classList.remove("hidden");
    $("startGameBtn").textContent = "RESUME RUN";
  });
  document.querySelectorAll("[data-back-menu]").forEach(button =>
    button.addEventListener("click", () => showScreen("mainMenuScreen"))
  );
  document.querySelectorAll(".level-card").forEach(card => {
    card.addEventListener("click", () => {
      selectedLevel = Number(card.dataset.level) || 1;
      document.querySelectorAll(".level-card").forEach(item =>
        item.classList.toggle("selected", item === card)
      );
    });
  });
  $("confirmLevelBtn").addEventListener("click", () => {
    window.emotionControllerGame?.setStartingLevel?.(selectedLevel);
    openGame();
    $("overlayTitle").textContent = `Ready for level ${String(selectedLevel).padStart(2,"0")}?`;
    $("overlayMessage").textContent = "Enable your camera on the right, then start your run.";
    $("startGameBtn").textContent = "START RUN";
    $("gameOverlay").classList.remove("hidden");
  });

  const modal = $("settingsModal");
  $("menuSettingsBtn").addEventListener("click", () => modal.classList.remove("hidden"));
  $("closeSettingsBtn").addEventListener("click", () => modal.classList.add("hidden"));
  $("settingsDoneBtn").addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", event => {
    if (event.target === modal) modal.classList.add("hidden");
  });

  $("gameMenuBtn").addEventListener("click", goToMenu);
  $("fullscreenBtn").addEventListener("click", async () => {
    try {
      if (!document.fullscreenElement) await $("gameScreen").requestFullscreen();
      else await document.exitFullscreen();
    } catch (error) {
      console.warn("Fullscreen is unavailable:", error);
    }
  });

  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !modal.classList.contains("hidden")) {
      modal.classList.add("hidden");
    }
    if (event.key === "Escape" && !$("gameScreen").classList.contains("hidden")) {
      const status = gameStatus();
      if (status.running) {
        $("pauseBtn").click();
      }
    }
  });

  // Keep Resume state in sync after pausing or completing a run.
  const overlay = $("gameOverlay");
  const observer = new MutationObserver(refreshMenu);
  observer.observe(overlay, {attributes:true, attributeFilter:["class"]});
  $("playerName").focus();
})();
