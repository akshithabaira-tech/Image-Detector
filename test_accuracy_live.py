import os
from backend.detector import ImageDetector

def main():
    detector = ImageDetector()
    img_path = "test_ai.png"
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    print(f"\n--- Testing AI Image: {img_path} ---")
    try:
        verdict, conf, reason, artifacts = detector.predict(img_path)
        print(f"Verdict: {verdict}")
        print(f"Confidence: {conf*100}%")
        print(f"Reason: {reason}")
        print(f"Artifacts: {artifacts}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
