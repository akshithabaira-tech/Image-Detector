const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultSection = document.getElementById("resultSection");
const resultCard = document.getElementById("resultCard");
const resultText = document.getElementById("resultText");
const confidenceText = document.getElementById("confidenceText");
const additionalInfo = document.getElementById("additionalInfo");
const loading = document.getElementById("loading");
const dropZone = document.getElementById("dropZone");
const dropText = document.getElementById("dropText");
const previewName = document.getElementById("previewName");
const heatmapContainer = document.getElementById("heatmapContainer");
const heatmapImage = document.getElementById("heatmapImage");
const resetBtn = document.getElementById("resetBtn"); // Added resetBtn

let selectedFile = null;

// File input handler with validation
imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            alert("File size too large. Please select an image smaller than 10MB.");
            imageInput.value = "";
            return;
        }

        // Validate file type
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            imageInput.value = "";
            return;
        }

        selectedFile = file;
        dropText.textContent = file.name;
        previewName.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = e => {
            preview.src = e.target.result;
            previewContainer.classList.remove("hidden");
            dropZone.style.display = 'none'; // hide drop zone when image is shown
        };
        reader.readAsDataURL(file);
        analyzeBtn.disabled = false;
        resultSection.classList.add("hidden");
    } else {
        selectedFile = null;
        dropText.textContent = "Drag & Drop or Click to Upload";
        previewContainer.classList.add("hidden");
        dropZone.style.display = 'block';
        analyzeBtn.disabled = true;
    }
});

// Analyze button handler
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    loading.classList.remove("hidden");
    resultSection.classList.add("hidden");
    heatmapContainer.classList.add("hidden");
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        console.log("Sending file to /api/analyze...");

        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("Received data:", data);

        displayResult(data);

    } catch (error) {
        console.error("Analysis error:", error);
        displayError(error.message);
    } finally {
        loading.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});

// Display analysis result
// Reset button handler
resetBtn.addEventListener("click", () => {
    selectedFile = null;
    imageInput.value = "";
    dropText.textContent = "Drag & Drop or Click to Upload";
    previewContainer.classList.add("hidden");
    dropZone.style.display = 'block';
    analyzeBtn.disabled = true;
    resultSection.classList.add("hidden");
    heatmapContainer.classList.add("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// Display analysis result
function displayResult(data) {
    if (data.status === 'success') {
        resultSection.classList.remove('hidden');
        resultCard.className = 'card fade-in-up';
        
        const pred = data.prediction;
        const score = data.confidence_score * 100;
        const msg = data.message;
        
        // Update Verdict Status style and classes
        resultCard.classList.remove('state-ai', 'state-real', 'state-uncertain');
        
        let icon = "❓";
        if (pred === 'AI Generated') {
            resultCard.classList.add('state-ai');
            icon = "🔴";
        } else if (pred === 'Real Image') {
            resultCard.classList.add('state-real');
            icon = "🟢";
        } else if (pred === 'Edited / Manipulated') {
            resultCard.classList.add('state-uncertain');
            icon = "🟡";
        } else {
            resultCard.classList.add('state-uncertain');
            icon = "🟡";
        }

        resultText.innerText = `${icon} ${pred}`;
        
        // Update Confidence Meter
        const meterFill = document.getElementById("confidenceMeter");
        const pctText = document.getElementById("confidenceText");
        
        pctText.innerText = `${Math.round(score)}%`;
        meterFill.style.width = '0%';
        setTimeout(() => {
            meterFill.style.width = `${score}%`;
        }, 300);

        // Update Report Text
        additionalInfo.innerText = data.message || "Detailed forensic report generated below.";
        
        // Populate Signal Breakdown
        const scoreBreakdown = document.getElementById("scoreBreakdown");
        const signalsContainer = document.getElementById("signalsContainer");
        signalsContainer.innerHTML = '';
        
        if (data.artifacts && data.artifacts.length > 0) {
            scoreBreakdown.classList.remove('hidden');
            data.artifacts.forEach(art => {
                // Check if it's a score artifact (e.g. "Ensemble: 0.5") or a tag
                if (art.includes(':')) {
                    const [name, val] = art.split(':');
                    const pill = document.createElement('div');
                    pill.className = 'signal-pill';
                    
                    let displayVal = val.trim();
                    if (displayVal.includes('OFFLINE') || displayVal.includes('ERROR')) {
                        pill.classList.add('signal-offline');
                    }
                    
                    pill.innerHTML = `
                        <span class="signal-val">${displayVal}</span>
                        <span class="signal-name">${name.trim()}</span>
                    `;
                    signalsContainer.appendChild(pill);
                }
            });
        }
        
        // Render Heatmap
        if (data.heatmap) {
            heatmapContainer.classList.remove('hidden');
            heatmapImage.src = `data:image/png;base64,${data.heatmap}`;
        } else {
            heatmapContainer.classList.add('hidden');
        }

        // Auto-scroll to results
        setTimeout(() => {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }
}

// Display error message
function displayError(message) {
    resultCard.className = "card result-error";
    resultText.textContent = "Analysis Interrupted";
    resultText.style.color = "var(--danger-color)";
    confidenceText.textContent = "";
    additionalInfo.textContent = message;
    additionalInfo.style.display = "block";
    additionalInfo.style.color = "var(--danger-color)";

    resultSection.classList.remove("hidden");
}

// Add some visual feedback for file drag and drop
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        imageInput.dispatchEvent(new Event("change"));
    }
});
