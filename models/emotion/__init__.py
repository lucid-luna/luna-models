# =============================================================
#  LunaEmotion – Emotion Classification Model Pipeline (core scripts)
#  This single file shows **all** key modules that should live under
#  models/emotion/ according to the directory layout we agreed on.
# =============================================================

# ====================================================================
#  File: models/emotion/__init__.py
# ====================================================================

from importlib import import_module

preprocess = import_module("models.emotion.preprocess").main
train      = import_module("models.emotion.train").main
evaluate   = import_module("models.emotion.evaluate").main
inference  = import_module("models.emotion.inference").main

__all__ = [
    "preprocess",
    "train",
    "evaluate",
    "inference",
]