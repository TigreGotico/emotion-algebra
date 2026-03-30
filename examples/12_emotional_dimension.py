"""12_emotional_dimension.py — Inspecting EmotionalDimension axes."""
from emotion_data.plutchik import DIMENSIONS

print("=== All four Hourglass axes ===")
for axis_name, dim in DIMENSIONS.items():
    print(f"\n  Axis: {dim.axis}")
    print(f"    valence : {dim.valence:+d}  ({dim.kind})")
    print(f"    +3 intense : {dim.intense_emotion.name}")
    print(f"    +2 mild    : {dim.mild_emotion.name}")
    print(f"    +1 basic   : {dim.basic_emotion.name}")
    print(f"    -1 basic   : {dim.basic_opposite.name}")
    print(f"    -2 mild    : {dim.mild_opposite.name}")
    print(f"    -3 intense : {dim.intense_opposite.name}")

# --- Dimension containment check --------------------------------------------
from copy import copy
from emotion_data.emotions import EMOTIONS

print("\n=== Containment: 'anger in sensitivity' ===")
anger = copy(EMOTIONS["anger"])
sensitivity = DIMENSIONS["sensitivity"]
print(f"  anger in sensitivity  : {anger in sensitivity}")
print(f"  anger in pleasantness : {anger in DIMENSIONS['pleasantness']}")

# --- Dimension addition (CompositeDimension) --------------------------------
print("\n=== Dimension addition ===")
from emotion_data.composite_emotions import CompositeDimension
combined = DIMENSIONS["sensitivity"] + DIMENSIONS["pleasantness"]
print(f"  sensitivity + pleasantness → {type(combined).__name__}")
print(f"  axes: {combined.axes}")
