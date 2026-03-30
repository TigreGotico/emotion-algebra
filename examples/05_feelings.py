"""05_feelings.py — Named Feelings (Plutchik's dyads) and their properties."""
from copy import copy
from emotion_data.feelings import FEELINGS, FEELING_NAMES, OPPOSITE_FEELINGS_NAMES, get_feeling_from_emotions

# --- Accessing named feelings -----------------------------------------------
print("=== Selected named feelings ===")
for name in ["love", "optimism", "hope", "despair", "anxiety", "remorse"]:
    f = copy(FEELINGS[name])
    comps = [e.name for e in f.emotions]
    print(f"  {name:15s}  components={comps}  "
          f"valence={f.valence:+d}  arousal={f.arousal}  type={f.type}")

# --- All feelings with their types ------------------------------------------
print("\n=== All feelings and Circumplex types ===")
by_type: dict = {}
for name in FEELING_NAMES:
    f = copy(FEELINGS[name])
    by_type.setdefault(f.type, []).append(name)

for t in sorted(by_type):
    print(f"  {t}: {', '.join(by_type[t])}")

# --- Opposite feelings ------------------------------------------------------
print("\n=== Opposite feelings ===")
for f, opp in list(OPPOSITE_FEELINGS_NAMES.items())[:8]:
    print(f"  {f:20s}  ↔  {opp}")

# --- Lookup by component emotions -------------------------------------------
print("\n=== get_feeling_from_emotions ===")
pairs = [("joy", "trust"), ("fear", "sadness"), ("anticipation", "joy"), ("sadness", "anger")]
for e1, e2 in pairs:
    name = get_feeling_from_emotions(e1, e2)
    print(f"  {e1} + {e2} → {name}")
