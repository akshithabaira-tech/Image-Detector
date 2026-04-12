// Content script — injects result overlay next to the right-clicked image

let currentPopup = null;
let currentImageUrl = null;

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SHOW_LOADING") {
    currentImageUrl = msg.imageUrl;
    showPopup(msg.imageUrl, "loading");
  } else if (msg.type === "SHOW_RESULT") {
    if (msg.imageUrl === currentImageUrl) {
      showPopup(msg.imageUrl, "result", msg.result);
    }
  } else if (msg.type === "SHOW_ERROR") {
    if (msg.imageUrl === currentImageUrl) {
      showPopup(msg.imageUrl, "error", null, msg.error);
    }
  }
});

function showPopup(imageUrl, state, result = null, error = null) {
  // Remove existing popup
  if (currentPopup) {
    currentPopup.remove();
    currentPopup = null;
  }

  // Find the image element
  const imgEl = findImageElement(imageUrl);

  // Create popup
  const popup = document.createElement("div");
  popup.className = "aid-popup";
  popup.id = "aid-popup";

  if (state === "loading") {
    popup.innerHTML = `
      <div class="aid-header">
        <span class="aid-logo">⬡</span>
        <span class="aid-title">AI Detector</span>
        <button class="aid-close" onclick="this.closest('#aid-popup').remove()">✕</button>
      </div>
      <div class="aid-body">
        <div class="aid-spinner"></div>
        <p class="aid-status-text">Analyzing image...</p>
      </div>
    `;
  } else if (state === "result") {
    const pred = result.prediction;
    const score = result.confidence_score;
    const artifacts = result.artifacts || [];

    let iconClass = "aid-icon-uncertain";
    let icon = "◎";
    let labelClass = "aid-label-uncertain";

    if (pred === "AI-Generated") {
      iconClass = "aid-icon-ai";
      icon = "⚠";
      labelClass = "aid-label-ai";
    } else if (pred === "Real") {
      iconClass = "aid-icon-real";
      icon = "✓";
      labelClass = "aid-label-real";
    } else if (pred === "Manipulated") {
      iconClass = "aid-icon-manipulated";
      icon = "✎";
      labelClass = "aid-label-manipulated";
    }

    const barWidth = Math.round(score * 100);
    const barClass =
      pred === "AI-Generated"
        ? "aid-bar-ai"
        : pred === "Real"
        ? "aid-bar-real"
        : pred === "Manipulated"
        ? "aid-bar-manipulated"
        : "aid-bar-uncertain";

    // Build forensic log HTML
    let forensicHtml = "";
    if (artifacts.length > 0) {
      forensicHtml = `<div class="aid-forensic-log">`;
      artifacts.forEach(art => {
        if (art.includes(':')) {
           const [key, val] = art.split(':');
           forensicHtml += `<div class="aid-forensic-item"><span>${key}</span><span class="aid-forensic-value">${val}</span></div>`;
        } else {
           forensicHtml += `<div class="aid-forensic-item"><span>${art}</span></div>`;
        }
      });
      forensicHtml += `</div>`;
    }

    popup.innerHTML = `
      <div class="aid-header">
        <span class="aid-logo">⬡</span>
        <span class="aid-title">AI Detector</span>
        <button class="aid-close" onclick="this.closest('#aid-popup').remove()">✕</button>
      </div>
      <div class="aid-body">
        <div class="aid-result-icon ${iconClass}">${icon}</div>
        <div class="aid-label ${labelClass}">${pred}</div>
        <div class="aid-confidence">${Math.round(score * 100)}% Match</div>
        <div class="aid-bar-track">
          <div class="aid-bar-fill ${barClass}" style="width: ${barWidth}%"></div>
        </div>
        ${forensicHtml}
        ${
          pred === "Uncertain"
            ? `<p class="aid-note">Analysis inconclusive — image may be borderline or low-res.</p>`
            : ""
        }
      </div>
      <div class="aid-footer">
        <span>Powered by Multi-Signal Forensic Engine</span>
      </div>
    `;
  } else if (state === "error") {
    popup.innerHTML = `
      <div class="aid-header">
        <span class="aid-logo">⬡</span>
        <span class="aid-title">AI Detector</span>
        <button class="aid-close" onclick="this.closest('#aid-popup').remove()">✕</button>
      </div>
      <div class="aid-body">
        <div class="aid-result-icon aid-icon-error">✕</div>
        <div class="aid-label aid-label-error">Error</div>
        <p class="aid-note">${error}</p>
        <p class="aid-note" style="margin-top:4px;opacity:0.6;">Make sure your backend is running on localhost:8000</p>
      </div>
    `;
  }

  // Position popup near image or fixed on screen
  document.body.appendChild(popup);
  currentPopup = popup;

  if (imgEl) {
    positionNearImage(popup, imgEl);
  } else {
    popup.style.position = "fixed";
    popup.style.top = "20px";
    popup.style.right = "20px";
  }

  // Auto-remove after 12 seconds for results
  if (state === "result") {
    setTimeout(() => {
      if (popup.parentNode) popup.remove();
    }, 12000);
  }
}

function findImageElement(src) {
  const imgs = document.querySelectorAll("img");
  for (const img of imgs) {
    if (img.src === src || img.currentSrc === src) return img;
  }
  return null;
}

function positionNearImage(popup, imgEl) {
  const rect = imgEl.getBoundingClientRect();
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;

  popup.style.position = "absolute";

  let top = rect.bottom + scrollY + 8;
  let left = rect.left + scrollX;

  // Keep within viewport
  const popupWidth = 280;
  if (left + popupWidth > window.innerWidth) {
    left = window.innerWidth - popupWidth - 16;
  }

  popup.style.top = `${top}px`;
  popup.style.left = `${left}px`;
}
