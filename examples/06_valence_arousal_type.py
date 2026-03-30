"""06_valence_arousal_type.py — Valence, arousal, and Circumplex type for all emotions."""
from copy import copy
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.plutchik import Neutrality

print("=== Valence and arousal for all 24 emotions ===")
print(f"  {'Name':15s}  {'Axis':12s}  {'flow':>5}  {'valence':>7}  {'arousal':>7}  type")
print("  " + "-" * 80)
for name, e in EMOTIONS.items():
    e = copy(e)
    print(f"  {e.name:15s}  {e.dimension.axis:12s}  "
          f"{e.emotional_flow:+5d}  {e.valence:+7d}  {e.arousal:7d}  {e.type}")

print(f"\n  {'Neutrality':15s}  {'—':12s}  "
      f"{0:+5d}  {0:+7d}  {0:7d}  {Neutrality().type}")

# --- Key insight: anger vs joy ----------------------------------------------
print("\n=== Key insight: anger has valence=0, joy has valence=+2 ===")
anger = copy(EMOTIONS["anger"])
joy   = copy(EMOTIONS["joy"])
print(f"  anger : valence={anger.valence}  arousal={anger.arousal}  type={anger.type!r}")
print(f"  joy   : valence={joy.valence}  arousal={joy.arousal}  type={joy.type!r}")
print()
print("  Anger is NOT negative — it is 'activated neutral'.")
print("  Arousal (reactivity) is orthogonal to hedonic tone (Posner et al. 2005).")
