"""07_emotion_vectors.py — 4D Hourglass vectors and matrix representations."""
import numpy as np
from copy import copy
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.composite_emotions import CompositeEmotion

# --- Single emotion vector --------------------------------------------------
print("=== 4D vector for joy (Pleasantness +2) ===")
joy = copy(EMOTIONS["joy"])
print(f"  emotion_vector : {[str(e) for e in joy.emotion_vector]}")
print(f"  as_array       : {joy.as_array}")
print(f"  as_matrix      :\n{joy.as_matrix}")

# --- Composite emotion vector -----------------------------------------------
print("\n=== 4D vector for love (joy + trust) ===")
trust = copy(EMOTIONS["trust"])
love = joy + trust  # CompositeEmotion
print(f"  emotion_vector : {[str(e) for e in love.emotion_vector]}")
print(f"  as_array       : {love.as_array}")

# --- All four axes in a composite -------------------------------------------
print("\n=== Building a 4-axis composite ===")
anger     = copy(EMOTIONS["anger"])
vigilance = copy(EMOTIONS["vigilance"])
# Build step by step
c = joy + trust      # pleasantness + aptitude
c2 = c + anger       # + sensitivity
c3 = c2 + vigilance  # + attention
print(f"  components   : {[e.name for e in c3.components]}")
print(f"  as_array     : {c3.as_array}")
print(f"  emotional_flow: {c3.emotional_flow}")  # sum of all flows

# --- composite from named emotions -------------------------------------------
# NOTE: CompositeEmotion.array_to_emotion currently collapses to a bare int
# for any input (Neutrality() + <int> is not an identity operation — see
# ROADMAP.md TEA08.10); build composites from named Emotions instead until
# that lands.
print("\n=== composite built from named emotions ===")
rage = copy(EMOTIONS["rage"])
grief = copy(EMOTIONS["grief"])
reconstructed = rage + grief
print(f"  components   : {[e.name for e in reconstructed.components]}")
print(f"  as_array     : {reconstructed.as_array}")
