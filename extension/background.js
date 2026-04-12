// Background service worker for AI Image Detector extension

const API_BASE = "http://localhost:8000";

// Create context menu on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-ai-image",
    title: "🔍 Check with AI Detector",
    contexts: ["image"],
  });

  // Set default API URL
  chrome.storage.sync.get("apiUrl", (data) => {
    if (!data.apiUrl) {
      chrome.storage.sync.set({ apiUrl: API_BASE });
    }
  });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "check-ai-image") return;

  const imageUrl = info.srcUrl;
  if (!imageUrl) return;

  // Tell content script to show loading state
  chrome.tabs.sendMessage(tab.id, {
    type: "SHOW_LOADING",
    imageUrl: imageUrl,
  });

  try {
    // Get API URL from storage
    const { apiUrl } = await chrome.storage.sync.get("apiUrl");
    const base = apiUrl || API_BASE;

    // Fetch the image as blob
    const imageBlob = await fetchImageAsBlob(imageUrl);
    if (!imageBlob) throw new Error("Could not fetch image");

    // Send to backend
    const formData = new FormData();
    const ext = getExtension(imageUrl);
    formData.append("file", imageBlob, `image.${ext}`);

    const response = await fetch(`${base}/api/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    const result = await response.json();

    // Save result to local history for popup display
    const { history = [] } = await chrome.storage.local.get("history");
    const historyItem = {
      timestamp: Date.now(),
      imageUrl: imageUrl,
      prediction: result.prediction,
      confidence: result.confidence,
    };
    const newHistory = [historyItem, ...history].slice(0, 5); // Keep last 5
    chrome.storage.local.set({ history: newHistory });

    // Send result to content script
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_RESULT",
      imageUrl: imageUrl,
      result: result,
    });
  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_ERROR",
      imageUrl: imageUrl,
      error: err.message || "Analysis failed",
    });
  }
});

// Fetch image as blob (handles CORS by going through background)
async function fetchImageAsBlob(url) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    return blob;
  } catch {
    return null;
  }
}

function getExtension(url) {
  try {
    const path = new URL(url).pathname;
    const ext = path.split(".").pop().toLowerCase();
    return ["jpg", "jpeg", "png", "webp", "gif", "bmp"].includes(ext)
      ? ext
      : "jpg";
  } catch {
    return "jpg";
  }
}
