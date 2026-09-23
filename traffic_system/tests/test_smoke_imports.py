"""
Smoke test to ensure the environment is correctly configured.
This test verifies that all major libraries across the pipeline can be imported without errors.
"""
import sys

def run_smoke_tests():
    print(f"Python Version: {sys.version}\n")
    print("Running Dependency Smoke Tests...\n")

    dependencies = {
        "Data Engineering": ["numpy", "pandas", "h5py", "requests", "dotenv"],
        "Deep Learning (PyTorch)": ["torch", "torchvision"],
        "NLP (Transformers)": ["transformers", "sentencepiece", "sacremoses"],
        "Computer Vision": ["cv2", "ffmpeg"],
        "Graph & Routing": ["networkx"],
        "Evaluation": ["sklearn", "matplotlib", "tqdm"],
        "Testing & Env": ["pytest", "ipykernel"]
    }

    all_passed = True

    for category, modules in dependencies.items():
        print(f"--- {category} ---")
        for mod in modules:
            try:
                __import__(mod)
                print(f"[PASS] {mod} imported successfully.")
            except ImportError as e:
                print(f"[FAIL] Failed to import {mod}. Error: {e}")
                all_passed = False
        print("")

    if all_passed:
        print("ALL SMOKE TESTS PASSED! The environment is ready.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! Check the environment configuration.")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_tests()
