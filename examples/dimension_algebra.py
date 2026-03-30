from emotion_algebra.emotions import EMOTIONS, DIMENSIONS
from emotion_algebra.plutchik import EmotionalDimension, Neutrality
from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS, CompositeEmotion, CompositeDimension
from emotion_algebra.feelings import FEELINGS

# dimensions have a kind and valence
d = DIMENSIONS["sensitivity"]
assert d.kind == "negative"
assert d.valence == -1

d = DIMENSIONS["attention"]
assert d.kind == "positive"
assert d.valence == 1

# pleasantness and aptitude valence will depend on the emotion flow
d = DIMENSIONS["pleasantness"]
assert d.kind == "neutral"
assert d.valence == 0

# you can check if dimensions contain an emotion
e = EMOTIONS["rage"]

d = DIMENSIONS["sensitivity"]

assert e in d

# neutrality is in every dimension
n = Neutrality()
for dim in DIMENSIONS:
    assert n in DIMENSIONS[dim]

# dimension emotions can be retrieved
assert d.basic_emotion == EMOTIONS["annoyance"]
assert d.basic_opposite == "apprehension"


# composite dimensions can be created
d2 = DIMENSIONS["aptitude"]
c = d + d2
assert c.name == "sensitivity/aptitude"
assert isinstance(c, CompositeDimension)

# composite dimensions have composite emotions
assert isinstance(c.intense_opposite, CompositeEmotion)

# TODO if you create non standard composite dimensions (as defined by plutchik) emotions will be missing
d = DIMENSIONS["attention"]
d2 = DIMENSIONS["aptitude"]
c = d + d2
assert c.intense_opposite is None

# composite dimensions also allow subtracting dimensions
d3 = c - d
assert d3 == d2
assert isinstance(d3, EmotionalDimension)

# TODO composite dimensions never have base_emotion
d = DIMENSIONS["sensitivity"]
d2 = DIMENSIONS["aptitude"]
c = d + d2
assert c.basic_emotion is None
assert c.basic_opposite is None

# you can check if emotions, feelings and dimensions are part of composite dimensions

e = EMOTIONS["rage"]
assert e in c

# composite emotions are part even if only one of the component emotion's dimension is present
e = COMPOSITE_EMOTIONS["aggressiveness"]
assert d in c


# in case of feelings both dimensions must match
f = FEELINGS["pride"]
assert f not in c

d = DIMENSIONS["sensitivity"]
d2 = DIMENSIONS["pleasantness"]
c = d + d2
assert f in c
