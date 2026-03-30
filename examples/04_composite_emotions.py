"""04_composite_emotions.py — Cross-axis addition producing CompositeEmotion."""
from copy import copy
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.composite_emotions import CompositeEmotion, COMPOSITE_EMOTIONS_NAMES

# --- Cross-axis addition always returns CompositeEmotion --------------------
# Named composites use TERTIARY emotions (flow ±3) per COMPOSITE_EMOTIONS_NAMES
ecstasy   = copy(EMOTIONS["ecstasy"])    # Pleasantness +3
admiration = copy(EMOTIONS["admiration"])  # Aptitude +3

composite = ecstasy + admiration
print("=== ecstasy + admiration ===")
print(f"  type    : {type(composite).__name__}")
print(f"  name    : {composite.name}")       # "love"
print(f"  valence : {composite.valence}")    # 3 (ecstasy's Pleasantness)
print(f"  arousal : {composite.arousal}")    # 3 (max |flow|)
print(f"  type    : {composite.type}")       # excited positive
print(f"  flow    : {composite.emotional_flow}")

# --- All 16 named composites ------------------------------------------------
print("\n=== All 16 named CompositeEmotions ===")
for name, (e1, e2) in COMPOSITE_EMOTIONS_NAMES.items():
    a = copy(EMOTIONS[e1])
    b = copy(EMOTIONS[e2])
    c = a + b
    print(f"  {e1:12s} + {e2:12s}  →  {c.name:15s}  type={c.type}")

# --- Composite emotion vector -----------------------------------------------
print("\n=== Emotion vector of aggressiveness (rage + vigilance) ===")
rage      = copy(EMOTIONS["rage"])
vigilance = copy(EMOTIONS["vigilance"])
agg = rage + vigilance
sensitivity, attention, pleasantness, aptitude = agg.emotion_vector
print(f"  sensitivity  : {sensitivity.name} (flow {sensitivity.emotional_flow})")
print(f"  attention    : {attention.name}   (flow {attention.emotional_flow})")
print(f"  pleasantness : {pleasantness.name} (flow {pleasantness.emotional_flow})")
print(f"  aptitude     : {aptitude.name}    (flow {aptitude.emotional_flow})")
print(f"  as_array     : {agg.as_array}")
