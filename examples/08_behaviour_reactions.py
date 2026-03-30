"""08_behaviour_reactions.py — Behaviour and BehavioralReaction mappings."""
from copy import copy
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.behaviour import BEHAVIOURS, REACTIONS, REACTION_TO_EMOTION_MAP

# --- What behaviours does anger trigger? ------------------------------------
print("=== Behaviours ===")
for name, behaviour in list(BEHAVIOURS.items())[:5]:
    print(f"  {name}: {behaviour}")

# --- What reactions does an emotion trigger? --------------------------------
print("\n=== Reactions triggered by anger ===")
anger = copy(EMOTIONS["anger"])
for r in anger.triggered_reactions:
    print(f"  {r}")

# --- REACTION_TO_EMOTION_MAP ------------------------------------------------
print("\n=== Reaction → Emotion mapping (first 8) ===")
for reaction_name, emotion in list(REACTION_TO_EMOTION_MAP.items())[:8]:
    print(f"  {reaction_name:20s} → {emotion.name}")

# --- BehavioralReaction.from_data -------------------------------------------
print("\n=== BehavioralReaction.from_data ===")
from emotion_algebra.behaviour import BehavioralReaction
# With valid data
if BEHAVIOURS:
    first_key = next(iter(BEHAVIOURS))
    data = {"behaviour": first_key}
    br = BehavioralReaction.from_data(data)
    print(f"  from_data({{'behaviour': {first_key!r}}}) → {br}")

# With empty data (graceful fallback)
br_empty = BehavioralReaction.from_data({})
print(f"  from_data({{}}) → {br_empty}")
