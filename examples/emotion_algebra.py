from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.feelings import Feeling
from emotion_algebra.plutchik import Neutrality
from emotion_algebra.composite_emotions import CompositeEmotion, CompositeDimension

e = EMOTIONS["joy"]

# when used as numeric types emotion flow representing intensity is considered
assert e.emotional_flow == int(e)

# when used as string emotion name is considered
assert e.name == str(e)

# len of emotions is always between 0 and 4 (1 per dimension)
assert len(e) == 1

# when used as boolean valence is used (positive vs negative emotion)
assert bool(e) == True

e = EMOTIONS["terror"]
assert bool(e) == False


# Neutrality is the neutral and unitary element of emotions, it changes nothing
n = Neutrality()
assert e + n == e
assert e - n == e
assert e * n == e
assert e / n == e

assert len(n) == 0
assert int(n) == 0

try:
    bool(n)
except:
    # bool for neutrality is undefined
    reason = "neutrality can not be cast to bool"

# neutrality is both true and false
assert n == True
assert n == False

# neutrality is always "in" an emotion
assert n in e

# the absolute value of any emotion is Neutrality
assert abs(e) == n


# you can sum emotions from the same dimension
e = EMOTIONS["annoyance"]
e2 = EMOTIONS["anger"]

assert e + e2 == "rage"

# but if they are from different dimensions you get a Feeling
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


# summing emotions with feelings returns another feeling
f2 = e + f
assert isinstance(f2, Feeling)


# summing opposite emotions returns neutrality
e = EMOTIONS["annoyance"]
e2 = EMOTIONS["apprehension"]

assert e + e2 == n


# emotions can be upgraded with integer operations
assert e + 1 == "anger"
assert e - 1 == "neutrality"

# emotion flow can not be < -3 or > 3
e2 = e + 5
assert e2.emotional_flow <= 3
e2 = e - 34
assert e2.emotional_flow >= -3

# in relation to plutchick emotions can have an offset to indicate strength
# offset defines "mega", "extreme" and "hyper" emotion types
e2 = EMOTIONS["rage"]
e = e2 + 1
assert e.name == "mega rage"
assert e.intensity_offset == 1
assert e.emotional_flow == 3


# offset is a kind of "extra flow" and indicates strength
e = e2 + 4
assert e.name == "hyper rage"
assert e.intensity_offset == 4
assert e.emotional_flow == 3


# offset is never < -6 or > 6
e = e2 + 55
assert e.intensity_offset == 6

# emotions with offset are still considered equal, since the flow does not change
assert e + e + e + e + e + e == e2
assert e + 999 == e2
# casting to int accounts for offset ( use in comparisons)
assert int(e + 34) == 9

# -9 < int < 9
# -3 < flow < 3
# -6 < offset < 6

# you can compare emotions intensity, dimension does not matter for this
# while the emotions are considered equal, the offset is taken into account when comparing
assert e2 < e
assert e > e2

# emotions with offset are still equal ( kinda like if the "is" operator was overrided )
assert e == e2

# but <= and >= work as expected and measure intensity
assert e >= e2
assert e2 <= e

# string operations are also supported if the string is an emotion name
e = EMOTIONS["annoyance"]
assert e + "anger" == "rage"

# else the name is used as a string
assert e2 + " is an emotion" == "rage is an emotion"
assert n + " is the zero of emotions" == "neutrality is the zero of emotions"

# subtracting emotions is also possible
assert e2 - e == "anger"


# emotions can be negated
e = EMOTIONS["acceptance"]
assert e.opposite_emotion == "boredom"
assert - e == "boredom"


# you can also check if dimensions are part of an emotion

from emotion_algebra.plutchik import DIMENSIONS

d = DIMENSIONS["aptitude"]

assert d in e

d = DIMENSIONS["attention"]

assert d not in e

# emotions have a emotion vector representation
import numpy as np

e = EMOTIONS["rage"]
assert isinstance(e.as_array, np.ndarray)
assert e.emotion_vector == [e, Neutrality(), Neutrality(), Neutrality()]

# emotions have kinds and types

# type is a human readable concept calculated from dimension.axis
assert e.type == "negative and lively and forceful"

e2 = EMOTIONS["serenity"]
assert e2.type == "positive and quiet"

# kind is a look up by name, the kind_map is hard coded
# open an issue if you find empty kinds!
assert e2.kind == "future appraisal"
e2 = EMOTIONS["joy"]
assert e2.kind == "event related"

# you can multiply emotions to create a composite emotion
c = e * e2
assert isinstance(c, CompositeEmotion)
assert isinstance(c.dimension, CompositeDimension)


# you can create a neutrality with dimension
n = Neutrality(dimension="aptitude")

# it behaves normally if you sum it to something
weird_emotion = e2 + n
assert weird_emotion.name == "joy"
assert weird_emotion.emotional_flow == 2
assert weird_emotion.dimension == "pleasantness"

# but if you sum something to it instead, you create new weird emotions
weird_emotion = n + e2
assert weird_emotion.dimension == "aptitude"
# name is summed
assert weird_emotion.name == "neutrality joy"
# flow remains 0
assert weird_emotion.emotional_flow == 0
# it is of neutral type
assert weird_emotion.type == "neutral"
# but it is no longer a neutrality
assert not isinstance(weird_emotion, Neutrality)
# kind will be inherited
assert weird_emotion.kind == "event related"


