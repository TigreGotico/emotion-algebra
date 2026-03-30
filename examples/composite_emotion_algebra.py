from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.feelings import Feeling, FEELINGS
from emotion_algebra.plutchik import Neutrality, Emotion
from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS, CompositeEmotion, CompositeDimension
import numpy as np

e = COMPOSITE_EMOTIONS["disapproval"]

# composite emotions are emotions in more than 1 dimension
# some composite emotions have a name
assert e.name == "disapproval"

# if the emotion is not named, the naming is of the sort
# "{dimension name} of {emotion name}, {dimension2 name} of {emotion2 name}"
assert e.secondary_name == "pleasantness of grief, attention of amazement"

# composite emotions have an 4D emotion vector representation
assert e.emotion_vector == [Neutrality(), EMOTIONS["amazement"], EMOTIONS["grief"], Neutrality()]

# composite emotions have an equivalent feeling
assert isinstance(e.equivalent_feeling, Feeling)
assert str(e.equivalent_feeling) == "mix of amazement and grief"
assert e.equivalent_feeling.name == "horror"

# composite emotions can be summed, this operates on the emotion_vector level
e2 = COMPOSITE_EMOTIONS["anxiety"]
assert e + e2 == "sensitivity of terror, pleasantness of grief"
assert isinstance(e2, CompositeEmotion)

e2 = EMOTIONS["rage"]
assert e + e2 == "sensitivity of rage, attention of amazement, pleasantness of grief"
assert isinstance(e + e2, CompositeEmotion)

f = FEELINGS["hatred"]
assert e + f == "sensitivity of rage, attention of amazement, pleasantness of grief, aptitude of loathing"
assert isinstance(e + f, CompositeEmotion)
assert (e + f).equivalent_feeling.name == "mix of rage and amazement and grief and loathing"

assert e - "grief" == "vigilance"
assert isinstance(e - "grief", Emotion)

# the flow is the dot product of the emotional_flow for the emotion vector
assert e.emotional_flow == 4.242640687119285

# composite emotions intensity can also be compared
assert e.emotional_flow > e2.emotional_flow
assert e.emotional_flow < 5
assert e > e2

# composite emotions also have an emotion matrix
# while emotion vectors are lists, matrix is an numpy object
# matrix are of the kind np.matrix(((sensitivity, pleasantness), (attention, aptitude)))
assert isinstance(e.as_matrix, np.matrix)

sensitivity = e.as_matrix.item(0, 0)
assert sensitivity == 0
attention = e.as_matrix.item(0, 1)
assert attention == -3
pleasantness = e.as_matrix.item(1, 0)
assert pleasantness == -3
aptitude = e.as_matrix.item(1, 1)
assert aptitude == 0


# basic emotions can be multiplied to create a composite emotion

e = EMOTIONS["ecstasy"]
e2 = EMOTIONS["admiration"]
c = e * e2
assert isinstance(c, CompositeEmotion)
assert isinstance(c.dimension, CompositeDimension)
assert c.name == "love"


# composite emotions also have kind and type
assert c.type == "positive quiet and caring"
assert c.kind == ""  # usually empty, hard coded dictionary









# TODO TODO TODO TODO
# NOTE you reached construction area
# TODO TODO TODO TODO








# multiplying basic emotions by composite emotions
# composite emotions can be multiplied by using their matrix

e3 = EMOTIONS["rage"]
# love * rage = admiration
print(c * e3)

c2 = COMPOSITE_EMOTIONS["remorse"]
# love * remorse = grief + loathing
print(c * c2)