// popup.js — handles settings, backend health check, and scanning history

const apiUrlInput = document.getElementById("apiUrl");
const saveBtn = document.getElementById("saveBtn");
const savedMsg = document.getElementById("savedMsg");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const historyContainer = document.getElementById("historyContainer");

// Load initial state
document.addEventListener("DOMContentLoaded", () => {
    chrome.storage.sync.get("apiUrl", (data) => {
        const url = data.apiUrl || "http://localhost:8000";
        apiUrlInput.value = url;
        checkHealth(url);
    });

    loadHistory();
});

// Save API URL
saveBtn.addEventListener("click", () => {
    const url = apiUrlInput.value.trim().replace(/\/$/, "");
    if (!url) return;

    chrome.storage.sync.set({ apiUrl: url }, () => {
        savedMsg.style.display = "block";
        setTimeout(() => (savedMsg.style.display = "none"), 2500);
        checkHealth(url);
    });
});

// Check backend health
async function checkHealth(baseUrl) {
    statusDot.className = "status-dot checking";
    statusText.textContent = "Detecting health...";

    try {
        const res = await fetch(`${baseUrl}/api/health`, {
            signal: AbortSignal.timeout(4000),
        });

        if (res.ok) {
            const data = await res.json();
            if (data.model_loaded) {
                statusDot.className = "status-dot online";
                statusText.textContent = "Backend Operational ✓";
            } else {
                 statusDot.className = "status-dot checking";
                 statusText.textContent = "Models Initialization...";
            }
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch (err) {
        statusDot.className = "status-dot offline";
        statusText.textContent = "Backend Offline — Start FastAPI";
    }
}

// Load and render analysis history
function loadHistory() {
    chrome.storage.local.get("history", (data) => {
        const history = data.history || [];
        
        if (history.length === 0) {
            historyContainer.innerHTML = `<div class="empty-state">No recent scans yet. Right-click an image to begin.</div>`;
            return;
        }

        historyContainer.innerHTML = history.map(item => {
            const isAI = item.prediction === "AI-Generated";
            const isReal = item.prediction === "Real";
            const predText = item.prediction || "Unknown";
            const predClass = isAI ? "ai" : (isReal ? "real" : "");
            
            return `
                <div class="history-item">
                    <img src="${item.imageUrl}" class="hist-thumb" onerror="this.src='icons/icon48.png'">
                    <div class="hist-info">
                        <div class="hist-pred ${predClass}">${predText.toUpperCase()}</div>
                        <div class="hist-conf">${item.confidence || "0%"} confidence</div>
                    </div>
                </div>
            `;
        }).join("");
    });
}

// Listen for storage changes to update history in real-time
chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.history) {
        loadHistory();
    }
});
