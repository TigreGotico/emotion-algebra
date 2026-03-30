"""01_basic_emotions.py — Accessing and inspecting the 24 named emotions."""
from copy import copy
from emotion_data.emotions import EMOTIONS, POSITIVE_EMOTIONS, NEGATIVE_EMOTIONS, DIMENSION_TO_EMOTION_MAP

# --- All 24 emotions --------------------------------------------------------
print("=== All emotions ===")
for name, emotion in EMOTIONS.items():
    print(f"  {emotion.name:15s}  flow={emotion.emotional_flow:+d}  "
          f"valence={emotion.valence:+d}  arousal={emotion.arousal}  "
          f"axis={emotion.dimension.axis}")

# --- Positive / negative (Pleasantness axis only) ---------------------------
print("\n=== POSITIVE_EMOTIONS (Pleasantness > 0) ===")
for e in POSITIVE_EMOTIONS:
    print(f"  {e.name}  valence={e.valence:+d}")

print("\n=== NEGATIVE_EMOTIONS (Pleasantness < 0) ===")
for e in NEGATIVE_EMOTIONS:
    print(f"  {e.name}  valence={e.valence:+d}")

# --- Emotions by axis -------------------------------------------------------
print("\n=== Sensitivity axis ===")
for e in DIMENSION_TO_EMOTION_MAP["sensitivity"]:
    print(f"  {e.name:12s}  flow={e.emotional_flow:+d}")
