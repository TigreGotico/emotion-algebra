from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.feelings import Feeling, FEELINGS
from emotion_algebra.plutchik import Neutrality, Emotion
import numpy as np

n = Neutrality()

# if summing emotions from different dimensions you get a Feeling
e = EMOTIONS["annoyance"]
e2 = EMOTIONS["trust"]
f = e + e2

assert type(f) == Feeling
assert str(f) == "mix of annoyance and trust"


# feelings might have a explicit name or not
e = EMOTIONS["anticipation"]
e2 = EMOTIONS["joy"]
f = e + e2

assert f.name == "optimism"
assert f.secondary_name == "mix of anticipation and joy"
assert str(f) == f.secondary_name

# strings with emotion can also be used
f = e + "joy"
assert f.name == "optimism"

# or feeling names
f = e + "love"
assert f.name == "mix of joy and trust and anticipation"

f = f - "anticipation"
assert str(f) == "mix of joy and trust"
assert f.name == "love"


# feelings are created with the opposite emotion if subtracting emotions not in the feeling
e = EMOTIONS["acceptance"]
e2 = EMOTIONS["rage"]
f = e2 - e
assert f.secondary_name == "mix of rage and boredom"

# you can get emotions from feelings
f = f - "rage"
assert f.name == "boredom"
assert isinstance(f, Emotion)

# feelings can have any number of emotions
e3 = EMOTIONS["joy"]
f = e + e2 + e3
assert f.secondary_name == "mix of acceptance and rage and joy"

# neutrality is ignored
f = e + e2 + e3 + n
assert f.secondary_name == "mix of acceptance and rage and joy"

# feelings have a length
assert len(f) == 3


# empty feelings are equal to neutrality
f = Feeling()
assert len(f) == 0
assert f.name == "neutrality"
assert len(n) == len(f)
assert f == n
assert f.emotional_flow == 0

# feelings also have emotional flow
# but they have a boolean value == len(f) > 0
assert bool(f) == False

# you can add emotions to feelings
e = EMOTIONS["joy"]
e2 = EMOTIONS["surprise"]
f = f + e + e2
assert f.name == "delight"
assert bool(f) == True
assert str(f) == "mix of joy and surprise"
assert isinstance(f.emotions[0], Emotion)

# feelings can be upgraded
f = f + 1
assert str(f) == "mix of ecstasy and distraction"
f = f + 1
assert str(f) == "mix of mega ecstasy and neutrality"


# some dimension may turn into neutral and disappear
f = f + 1
assert str(f) == "mega ecstasy"
assert isinstance(f, Emotion)
assert not isinstance(f, Feeling)

# feelings can be summed
f = FEELINGS["despair"]
f2 = FEELINGS["envy"]

f_sum = f + f2
assert f_sum.name == "mix of fear and sadness and sadness and anger"
assert len(f_sum) == 4


# feelings can be represented by a 4D vector of emotions
f = FEELINGS["curiosity"]
assert f.emotion_vector == [Neutrality(), EMOTIONS["surprise"], Neutrality(), EMOTIONS["trust"]]
assert f.base_feeling == f

f = f + "surprise"
assert f.emotion_vector == [Neutrality(), EMOTIONS["amazement"], Neutrality(), EMOTIONS["trust"]]
assert f.emotions == [EMOTIONS["trust"], EMOTIONS["surprise"], EMOTIONS["surprise"]]


# you can work with the base feeling constructed from this vector
assert f.name == "mix of trust and surprise and surprise"
assert f.base_feeling.name == "mix of mega amazement and trust"
assert f.base_feeling == f


# feelings also have emotional flow
# flow is the dot product of the emotional_flow in every dimension
# the sum of the flow values is used to compute valence
assert f.emotional_flow == -3.605551275463989


# feelings intensity can be compared to each other
f = FEELINGS["sentimentality"]
f2 = FEELINGS["hatred"]

# intensity
assert f > f2
assert f2 < f

# and also compared to emotions
e = EMOTIONS["rage"]
assert e > f  # rage (very positive) > sentimentality (trust + sadness = neutral)
assert e > f2 # rage (very positive) > hatred (very negative)