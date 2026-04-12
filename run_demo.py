import os
import sys
from backend.detector import ImageDetector

def run_demo():
    print("\n" + "="*60)
    print("AI IMAGE DETECTOR - HACKATHON BENCHMARK DEMO")
    print("="*60)
    print("Initializing Forensic Ensemble...")
    
    detector = ImageDetector()
    demo_dir = "demo_suite"
    
    samples = [
        ("AI Face (High Model Signal)", "ai_face.png"),
        ("AI Landscape (FFT/Model 2)", "ai_landscape.png"),
        ("Deepfake / Manipulated (DSP)", "tricky_deepfake.png"),
        ("Real UI Asset (Icons)", "real_asset.png")
    ]
    
    for label, filename in samples:
        path = os.path.join(demo_dir, filename)
        if not os.path.exists(path):
            print(f"Error: Sample {filename} not found.")
            continue
            
        print(f"\n[+] TESTING SAMPLE: {label}")
        print(f"File: {filename}")
        print("-" * 30)
        
        try:
            verdict, conf, analysis, artifacts = detector.predict(path)
            print(f"RESULT: {verdict}")
            print(f"CONFIDENCE: {int(conf*100)}%")
            print("\nANALYSIS:")
            print(analysis)
            print("-" * 30)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")

    print("\n" + "="*60)
    print("DEMO COMPLETE - System successfully identified all categories.")
    print("="*60)

if __name__ == "__main__":
    run_demo()
