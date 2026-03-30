"""09_comparisons_sorting.py — Comparing and sorting emotions by intensity."""
from copy import copy
from emotion_data.emotions import EMOTIONS, DIMENSION_TO_EMOTION_MAP

# --- Direct comparison on same axis -----------------------------------------
print("=== Comparison on Sensitivity axis ===")
annoyance   = copy(EMOTIONS["annoyance"])
anger       = copy(EMOTIONS["anger"])
rage        = copy(EMOTIONS["rage"])
apprehension = copy(EMOTIONS["apprehension"])
fear        = copy(EMOTIONS["fear"])
terror      = copy(EMOTIONS["terror"])

print(f"  rage > anger       : {rage > anger}")           # True
print(f"  anger > annoyance  : {anger > annoyance}")      # True
print(f"  fear < apprehension: {fear < apprehension}")    # True (-2 < -1)
print(f"  anger == anger     : {anger == anger}")         # True
print(f"  anger != fear      : {anger != fear}")          # True

# --- Sorting emotions by flow -----------------------------------------------
print("\n=== Sensitivity axis sorted by flow ===")
sensitivity = [copy(e) for e in DIMENSION_TO_EMOTION_MAP["sensitivity"]]
sensitivity.sort()
for e in sensitivity:
    print(f"  {e.name:12s}  flow={e.emotional_flow:+d}")

# --- Sorting Pleasantness axis ----------------------------------------------
print("\n=== Pleasantness axis sorted by flow ===")
pleasantness = [copy(e) for e in DIMENSION_TO_EMOTION_MAP["pleasantness"]]
pleasantness.sort()
for e in pleasantness:
    print(f"  {e.name:12s}  flow={e.emotional_flow:+d}  valence={e.valence:+d}")
