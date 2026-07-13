"""EmotionAnalyzer — the high-level facade.

The case the affect core exists for: two messages that look identical to a
sentiment model, and need opposite responses.
"""
from emotion_algebra import EmotionAnalyzer

a = EmotionAnalyzer()

MESSAGES = [
    "This is the third time your app has lost my work. Fix it.",
    "I don't know if I'm doing this right and I'm scared I've broken something.",
]

print("Two support messages. Both negative. Both activated.\n")

for text, state in zip(MESSAGES, a.analyze_all(MESSAGES)):
    print(f"  {text}")
    print(f"    valence : {state.valence:+.2f}   (how bad does it feel)")
    print(f"    potency : {state.potency:+.2f}   (can they do something about it?)")
    print(f"    reads as: {a.dominant(state)}")
    print()

print("Near-identical valence. Opposite potency.")
print("One will escalate. The other will quietly leave.\n")

# The other models are still here.
print("Plutchik's wheel, faithfully implemented:")
anger = a.emotion("anger")
print(f"  anger + 1  -> {(anger + 1).name}")
print(f"  -anger     -> {(-anger).name}   (Plutchik's 'opposite' — graded METAPHOR)")
print(f"  resolve('love') -> {a.resolve('love').name}")
