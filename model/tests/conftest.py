import os
import sys

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
