from emotion_algebra.tag import best_emotion
from emotion_algebra.emotions import EMOTIONS


TEST_SENTENCES = ['I love mom\'s cooking',
                      'I love how you never reply back..',
                      'I love cruising with my homies',
                      'I love messing with yo mind!!',
                      'I love you and now you\'re just gone..',
                      'This is shit',
                      'This is the shit']

for sentence in TEST_SENTENCES:
    emotion = best_emotion(sentence)
    e = EMOTIONS[emotion]
    print(sentence)
    print(e.name)
    print("valence", e.valence)
    print("intensity", e.intensity)
    print("flow", e.emotional_flow)
    print("type", e.type)
    print("opposite emotion", e.opposite_emotion.name)
    print("triggered reactions", [r.__dict__ for r in e.triggered_reactions])
    print("\n")

