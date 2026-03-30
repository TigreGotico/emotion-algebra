"""03_negation_opposites.py — Opposite emotions via unary negation."""
from copy import copy
from emotion_data.emotions import EMOTIONS

pairs = [
    ("anger",       "fear"),
    ("joy",         "sadness"),
    ("trust",       "disgust"),
    ("anticipation","surprise"),
    ("rage",        "terror"),
    ("ecstasy",     "grief"),
    ("admiration",  "loathing"),
    ("vigilance",   "amazement"),
]

print("=== Negation: -emotion → opposite pole ===")
print(f"  {'Expression':25s}  {'Result':15s}  {'Expected':15s}  Match")
print("  " + "-" * 70)
for pos, neg in pairs:
    emotion = copy(EMOTIONS[pos])
    result = -emotion
    match = "✓" if result.name == neg else "✗"
    print(f"  -{pos:24s}  {result.name:15s}  {neg:15s}  {match}")

# --- Negation at every intensity level --------------------------------------
print("\n=== Negation at all intensity levels (Sensitivity axis) ===")
for name in ["annoyance", "anger", "rage"]:
    e = copy(EMOTIONS[name])
    opp = -e
    print(f"  -{name:12s} (flow {e.emotional_flow:+d})  →  "
          f"{opp.name:12s} (flow {opp.emotional_flow:+d})")

# --- Double negation is identity --------------------------------------------
print("\n=== Double negation ===")
joy = copy(EMOTIONS["joy"])
print(f"  -(-joy) == joy?  {(-(-joy)).name == joy.name}")
