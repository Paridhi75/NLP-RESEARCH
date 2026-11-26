"""
hindi_extension/predict.py
Delegates execution to the main predictor.
Enables running:
python hindi_extension/predict.py --original "..." --candidate "..."
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from predict import main

if __name__ == "__main__":
    main()
