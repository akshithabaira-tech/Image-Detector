# 🧠 AI Image Detector

A professional, full-stack web application designed to detect whether an image is Real (created by an optical camera) or AI-Generated. The system is built with a **FastAPI** backend that utilizes a state-of-the-art **Hugging Face** image classification model ensemble, alongside a sleek, responsive "Futuristic Glassmorphic" frontend.

![Frontend UI Preview](./glassmorphic_design_check_1774097715917.png) *(Note: Please place a screenshot of the UI in the root directory if you wish to display it on GitHub)*

---

## ✨ Features

- **Advanced AI Detection Ensemble**: Utilizes local models (`umm-maybe/AI-image-detector`, etc.) to provide high-speed, local image analysis.
- **Gemini 1.5 Vision Arbiter (Hybrid Mode)**: Integrates Google's Gemini-Vision API for state-of-the-art visual reasoning. It detects complex artifacts like "unnatural shadows" or "AI-smooth textures" that smaller models miss, providing a detailed text explanation for its decision.
- **Suspect-Region Heatmap**: Generates a visual breakdown of suspicious regions, highlighting where the AI detected tell-tale generator artifacts in red.
- **Futuristic Glassmorphic UI**: Features a beautifully designed dark-mode interface with neon cyan/indigo accents, glass-like floating panels, smooth CSS animations, and a large interactive drag-and-drop zone.

---

## 🛠️ Technologies Used

### Backend
- **Python 3.10+**
- **FastAPI & Uvicorn**: High-performance asynchronous web framework and ASGI server.
- **Hugging Face Transformers**: For loading and parsing the cutting-edge pre-trained classification models.
- **PyTorch & Pillow**: For robust tensor manipulation and image processing.

### Frontend
- **HTML5 & CSS3**: Vanilla implementation utilizing CSS Grid/Flexbox and native glassmorphism filters (`backdrop-filter`).
- **Vanilla JavaScript**: Fetch API for asynchronous multipart-form uploads and seamless DOM manipulation without relying on heavy frontend frameworks.
- **Google Fonts**: `Outfit` for tech-style headers and `Inter` for highly readable body text.

---

## 📂 Folder Structure

```text
AIImageDetector/
│
├── backend/
│   ├── app.py               # FastAPI application routing and initialization
│   ├── detector.py          # ML Inference logic and ensemble routing
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # Temporary directory used for caching uploads
│
├── frontend/
│   ├── index.html           # Main web page layout
│   ├── style.css            # Glassmorphic UI styling
│   └── script.js            # Drag-and-drop file handling and API calls
│
└── README.md                # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Ensure you have **Python 3.10** or higher installed.
- Ensure you have `Git` installed (optional, for cloning).

### 1. Environment Setup
To isolate project dependencies, it is highly recommended to create a Python Virtual Environment (`venv`):

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Navigate to the `backend` directory and install the required ML packages:

```bash
pip install -r backend/requirements.txt
```

### 3. Run the Server
Launch the FastAPI server using Uvicorn. The `app.py` script serves the frontend statically alongside the API routes.

```bash
# Ensure you are at the project root C:\Users\baira\rtrp\AIImageDetector
python -m uvicorn backend.app:app --reload
```

The application will now be running at: **`http://127.0.0.1:8000/`**

---

## 📡 API Endpoints

The backend exposes the following RESTful endpoints:

- **`GET /api/health`**
  - **Description**: Returns the server status and confirms if the heavy ML model weights have successfully loaded into memory.
  - **Response**: `{"status": "healthy", "model_loaded": true}`

- **`POST /api/analyze`**
  - **Description**: Accepts a `multipart/form-data` request containing an image file. Processes the image through the detector and returns the classification.
  - **Body Format**: Form Data with key `file` (Max size 10MB).
  - **Response Example**:
    ```json
    {
      "prediction": "AI Generated",
      "confidence": "98%",
      "confidence_score": 0.9876,
      "status": "success"
    }
    ```

---

## 📝 Usage

1. Open your web browser and navigate to `http://127.0.0.1:8000/`.
2. Drag and drop any image (JPG, PNG, WEBP) into the glowing drop zone, or click the zone to open your file explorer.
3. Click the **🚀 Analyze Image** button.
4. A futuristic loading spinner will appear while the backend processes the image via PyTorch.
5. The result card will slide up, indicating whether the image is **Real** (Green) or **AI Generated** (Red), along with the model's confidence percentage.

Enjoy detecting deepfakes! 🔍✨
