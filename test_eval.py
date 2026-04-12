import traceback
import os
import logging
from backend.detector import ImageDetector

# Configure logging to see the output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    detector = ImageDetector()
    img_path = "extension/icons/icon128.png"
    print(f"\n--- Testing {img_path} ---")
    try:
        res = detector.predict(img_path)
        print("Final Result:", res)
    except Exception as e:
        print("Error on", img_path)
        traceback.print_exc()

if __name__ == "__main__":
    main()
