import traceback
import os
from backend.detector import ImageDetector

def main():
    detector = ImageDetector()
    bad_dir = "backend/bad_images"
    if not os.path.exists(bad_dir):
        print("No bad images found.")
        return
        
    for f in os.listdir(bad_dir):
        if f.endswith((".jpg", ".jpeg", ".png", ".webp")):
            img_path = os.path.join(bad_dir, f)
            print(f"Testing {img_path}...")
            try:
                res = detector.predict(img_path)
                print("Result:", res)
            except Exception as e:
                print("Error on", img_path)
                traceback.print_exc()
                break

if __name__ == "__main__":
    main()
