"""10_emotion_analyzer.py — EmotionAnalyzer: text → emotion (requires [lexicon] extra)."""
try:
    import pandas  # noqa: F401
except ImportError:
    print("This example requires: pip install 'emotion_data[lexicon]'")
    raise SystemExit(0)

from emotion_algebra import EmotionAnalyzer

analyzer = EmotionAnalyzer()

# --- Classify words from the lexicon ----------------------------------------
words = ["happy", "angry", "terrified", "curious", "disgusted", "serene"]

print("=== Word → Emotion (lexicon lookup) ===")
for word in words:
    emotion = analyzer.emotion(word)
    if emotion:
        print(f"  {word:12s} → {emotion.name:12s}  "
              f"valence={emotion.valence:+d}  arousal={emotion.arousal}  "
              f"type={emotion.type}")
    else:
        print(f"  {word:12s} → not found in lexicon")

# --- get() returns raw label ------------------------------------------------
print("\n=== analyzer.get() returns raw label ===")
for word in words[:4]:
    label = analyzer.get(word)
    print(f"  {word:12s} → {label!r}")
