"""Plutchik's Wheel and Hourglass of Emotions — core Emotion and Dimension algebra.

Classes
-------
Emotion
    A single emotion on the Hourglass scale, supporting full operator algebra.
Neutrality
    The identity element of the emotion algebra (zero-intensity).
EmotionalDimension
    One of the four Hourglass axes: sensitivity, attention, pleasantness, aptitude.
"""
from __future__ import annotations

from copy import copy, deepcopy
from typing import Union

import numpy as np

from emotion_data.reference_maps import EMOTION_CONTRASTS


PRIMARY_EMOTION_NAMES = ["serenity", "pensiveness", "acceptance", "boredom", "apprehension", "annoyance", "distraction",
                         "interest"]
SECONDARY_EMOTIONS_NAMES = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

TERTIARY_EMOTIONS_NAMES = ["ecstasy", "grief", "admiration", "loathing", "terror", "rage", "amazement", "vigilance"]

HOURGLASS_OF_EMOTIONS = {"sensitivity": ["rage", "anger", "annoyance", "apprehension", "fear", "terror"],
                         "attention": ["vigilance", "anticipation", "interest", "distraction", "surprise", "amazement"],
                         "pleasantness": ["ecstasy", "joy", "serenity", "pensiveness", "sadness", "grief"],
                         "aptitude": ["admiration", "trust", "acceptance", "boredom", "disgust", "loathing"]}


# TODO science this instead of eye balling
# expand emotion contrasts, these are manually tagged and a matter of opinion

EMOTION_KIND_NAMES = EMOTION_CONTRASTS.copy()
EMOTION_KIND_NAMES['cathected'].append("contempt")

EMOTION_KIND_NAMES['related to object properties'].append("amazement")
EMOTION_KIND_NAMES['related to object properties'].append("awe")
EMOTION_KIND_NAMES['related to object properties'].append("annoyance")

EMOTION_KIND_NAMES['event related'].append("remorse")
EMOTION_KIND_NAMES['event related'].append("terror")
EMOTION_KIND_NAMES['event related'].append("disapproval")
EMOTION_KIND_NAMES['event related'].append("aggressiveness")
EMOTION_KIND_NAMES['event related'].append("sadness")
EMOTION_KIND_NAMES['event related'].append("acceptance")
EMOTION_KIND_NAMES['event related'].append("ecstasy")
EMOTION_KIND_NAMES['event related'].append("apprehension")
EMOTION_KIND_NAMES['event related'].append("loathing")
EMOTION_KIND_NAMES['event related'].append("gloat")  # TODO where does this fit better?

EMOTION_KIND_NAMES['social'].append("coercion")
EMOTION_KIND_NAMES['social'].append("trust")
EMOTION_KIND_NAMES['social'].append("submission")
EMOTION_KIND_NAMES['social'].append("rivalry")
EMOTION_KIND_NAMES['social'].append("rejection")
EMOTION_KIND_NAMES['social'].append("frivolity")  # TODO where does this fit better?

EMOTION_KIND_NAMES['future appraisal'].append("pensiveness")
EMOTION_KIND_NAMES['future appraisal'].append("optimism")
EMOTION_KIND_NAMES['future appraisal'].append("vigilance")
EMOTION_KIND_NAMES['future appraisal'].append("distraction")
EMOTION_KIND_NAMES['future appraisal'].append("serenity")
EMOTION_KIND_NAMES['future appraisal'].append("anticipation")


class Emotion(object):
    """A single emotion from the Hourglass of Emotions model.

    Supports arithmetic operators that move the emotion along its dimensional axis:
    ``anger + 1 == rage``, ``-anger == fear``, ``joy + trust == Feeling("love")``.

    Parameters
    ----------
    name:
        Canonical lowercase emotion name (e.g. ``"anger"``).
    dimension:
        Optional :class:`EmotionalDimension` or its string axis name.
    """

    def __init__(self, name: str, dimension: "Union[EmotionalDimension, str, None]" = None) -> None:
        self._name: str = name
        if dimension and isinstance(dimension, str):
            dimension = DIMENSIONS[dimension]
        self._dimension: "Union[EmotionalDimension, None]" = dimension
        self.intensity_offset: int = 0
        self._kind: str = ""

    @property
    def dimension(self) -> "Union[EmotionalDimension, None]":
        """The :class:`EmotionalDimension` this emotion belongs to, or ``None``."""
        return self._dimension or DIMENSIONS.get(self.name)

    @property
    def name(self) -> str:
        """Canonical name, prefixed with intensity label when ``intensity_offset > 0``."""
        if self.intensity_offset <= 0:
            return self._name
        if self.emotional_flow < 0:
            return self.intensity + " " + self.dimension.intense_opposite.name
        if self.emotional_flow > 0:
            return self.intensity + " " + self.dimension.intense_emotion.name

    @property
    def triggered_reactions(self):
        from emotion_data.behaviour import REACTION_TO_EMOTION_MAP, REACTIONS
        reactions = []
        for reaction in REACTION_TO_EMOTION_MAP:
            emo = REACTION_TO_EMOTION_MAP[reaction]
            if emo.name == self._name:
                reactions.append(REACTIONS[reaction])
        return reactions

    @property
    def base_emotion(self):
        if self.is_primary:
            return self
        if self.emotional_flow < 0:
            return self._dimension.basic_opposite
        else:
            return self._dimension.basic_emotion

    @property
    def parent_emotion(self):
        if self.is_primary or self.is_composite:
            return None
        if self.is_secondary:
            if self.emotional_flow < 0:
                return self._dimension.basic_opposite
            elif self.emotional_flow > 0:
                return self._dimension.basic_emotion
        elif self.is_tertiary:
            if self.emotional_flow < 0:
                return self._dimension.mild_opposite
            elif self.emotional_flow > 0:
                return self._dimension.mild_emotion
        return None

    @property
    def opposite_emotion(self):
        if self.is_primary and self.emotional_flow > 0:
            return self._dimension.basic_opposite
        elif self.is_primary and self.emotional_flow < 0:
            return self._dimension.basic_emotion
        elif self.is_secondary and self.emotional_flow > 0:
            return self._dimension.mild_opposite
        elif self.is_secondary and self.emotional_flow < 0:
            return self._dimension.mild_emotion
        elif self.is_tertiary and self.emotional_flow > 0:
            return self._dimension.intense_opposite
        elif self.is_tertiary and self.emotional_flow < 0:
            return self._dimension.intense_emotion
        return None

    @property
    def is_primary(self):
        return self._name in PRIMARY_EMOTION_NAMES and not self.is_composite

    @property
    def is_secondary(self):
        return self._name in SECONDARY_EMOTIONS_NAMES and not self.is_composite

    @property
    def is_tertiary(self):
        return self._name in TERTIARY_EMOTIONS_NAMES and not self.is_composite

    @property
    def is_hyper(self):
        return self.intensity_offset > 0

    @property
    def type(self):
        types = []
        valence = 0
        dim = self.dimension

        # type by dimension
        if "sensitivity" in dim.axis and abs(self.emotional_flow) > 1:
            types.append("lively")
        if "attention" in dim.axis and abs(self.emotional_flow) > 1:
            types.append("strong")

        # valence by dimension
        if "sensitivity" in dim.axis:
            valence -= abs(self.emotional_flow)
        if "attention" in dim.axis:
            valence += abs(self.emotional_flow)
        if "pleasantness" in dim.axis and self.emotional_flow:
            valence += self.emotional_flow
        if "pleasantness" in dim.axis and not self.emotional_flow:
            valence -= self.emotional_flow
        if "aptitude" in dim.axis and self.emotional_flow:
            valence += self.emotional_flow
        if "aptitude" in dim.axis and not self.emotional_flow:
            valence -= self.emotional_flow

        # type by column/dyad
        if abs(self.emotional_flow) == 2:
            types.append("not in control")
        elif abs(self.emotional_flow) == 1:
            types.append("quiet")
        elif abs(self.emotional_flow) == 3:
            types.append("forceful")

        # type by valence
        if valence > 0:
            types.insert(0, "positive")
        elif valence == 0:
            types.insert(0, "neutral")
        else:
            types.insert(0, "negative")
        return " and ".join(types)


    @property
    def kind(self):
        # TODO science this instead of eye balling
        for kind in EMOTION_KIND_NAMES:
            if self._name in EMOTION_KIND_NAMES[kind]:
                return kind
        return self._kind

    @property
    def emotional_flow(self) -> int:
        """Signed intensity level: ±1 primary, ±2 secondary, ±3 tertiary, 0 neutral."""
        if self.is_primary and self._dimension.basic_emotion.name == self._name:
            return 1
        elif self.is_primary and self._dimension.basic_opposite.name == self._name:
            return -1
        elif self.is_secondary and self._dimension.mild_emotion.name == self._name:
            return 2
        elif self.is_secondary and self._dimension.mild_opposite.name == self._name:
            return -2
        elif self.is_tertiary and self._dimension.intense_emotion.name == self._name:
            return 3
        elif self.is_tertiary and self._dimension.intense_opposite.name == self._name:
            return -3
        return 0

    @property
    def valence(self) -> bool:
        """``True`` when the emotion has a non-zero flow (i.e. is not neutral)."""
        return bool(self.emotional_flow)

    @property
    def intensity(self) -> str:
        """Human-readable intensity label: ``"neutral"``, ``"basic"``, ``"mild"``, ``"intense"``, or hyper prefix."""
        if abs(self.intensity_offset) == 1:
            return "mega"
        if abs(self.intensity_offset) == 2:
            return "extreme"
        if abs(self.intensity_offset) >= 3:
            return "hyper"

        if abs(self.emotional_flow) == 1:
            return "basic"
        if abs(self.emotional_flow) == 2:
            return "mild"
        if abs(self.emotional_flow) == 3:
            return "intense"

        return "neutral"

    @property
    def is_composite(self) -> bool:
        """Always ``False`` for plain :class:`Emotion`; overridden by :class:`CompositeEmotion`."""
        return False

    def emotion_from_flow(self, flow: Union[int, float]) -> "Emotion":
        flow = int(flow)
        # how to handle invalid flows?
        flow = 9 if flow > 9 else flow if flow > -9 else -9
        offset = abs(flow) - 3
        flow = 3 if flow > 3 else flow if flow > -3 else -3
        if flow == 1:
            emo = copy(self._dimension.basic_emotion)
            emo.intensity_offset = offset
            return emo
        elif flow == 2:
            emo = copy(self._dimension.mild_emotion)
            emo.intensity_offset = offset
            return emo
        elif flow == 3:
            emo = copy(self._dimension.intense_emotion)
            emo.intensity_offset = offset
            return emo
        elif flow == -1:
            emo = copy(self._dimension.basic_opposite)
            emo.intensity_offset = offset
            return emo
        elif flow == -2:
            emo = copy(self._dimension.mild_opposite)
            emo.intensity_offset = offset
            return emo
        elif flow == -3:
            emo = copy(self._dimension.intense_opposite)
            emo.intensity_offset = offset
            return emo
        return copy(Neutrality())

    def __repr__(self):
        return "EmotionObject:" + self.name

    def __str__(self):
        return self.name
    # +, -, *, @, /, //, %, divmod(), <<, >>, &, ^, |
    # TODO radds

    @staticmethod
    def string_to_emotion(string=""):
        from emotion_data.emotions import EMOTIONS
        from emotion_data.feelings import FEELINGS
        if string in EMOTIONS:
            return copy(EMOTIONS[string])
        if string in FEELINGS:
            return FEELINGS[string]
        return string

    # composite equivalent
    @property
    def emotion_vector(self):
        sensitivity = Neutrality()
        attention = Neutrality()
        pleasantness = Neutrality()
        aptitude = Neutrality()

        if self._dimension:
            if self._dimension.axis == "sensitivity":
                sensitivity = copy(self)
            elif self._dimension.axis == "attention":
                attention = copy(self)
            elif self._dimension.axis == "aptitude":
                aptitude = copy(self)
            elif self._dimension.axis == "pleasantness":
                pleasantness = copy(self)

        return [sensitivity, attention, pleasantness, aptitude]

    @property
    def as_array(self):
        return np.array([emo.emotional_flow for emo in self.emotion_vector])

    @property
    def as_matrix(self):
        sensitivity, attention, pleasantness, aptitude = self.emotion_vector
        return np.matrix(((int(sensitivity), int(attention)),
                          (int(pleasantness), int(aptitude))))

    def __len__(self):
        return len([e for e in self.emotion_vector if e.emotional_flow != 0])

    def __add__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
            if isinstance(other, str):
                return self.name + other
        if isinstance(other, Neutrality):
            return deepcopy(self)

        # create feeling
        if isinstance(other, Emotion):
            if other._dimension == self._dimension:
                flow = other.emotional_flow + self.emotional_flow
                return self.emotion_from_flow(flow)
            from emotion_data.feelings import Feeling
            feel = Feeling()
            feel.emotions = [copy(self), other]
            return feel

        from emotion_data.feelings import Feeling
        if isinstance(other, Feeling):
            other = other + self
            return other

        # upgrade emotion
        try:
            other = int(other)
            flow = self.emotional_flow + other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
            if isinstance(other, str):
                return self.name - other
        if isinstance(other, Neutrality):
            return deepcopy(self)
        if isinstance(other, Emotion):
            # add opposite emotion
            other = - other
            return self.__add__(other)
        # upgrade emotion
        try:
            other = int(other)
            flow = self.emotional_flow - other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return deepcopy(self)

        if isinstance(other, Emotion):
            from emotion_data.composite_emotions import CompositeEmotion
            # a composite emotion is created
            c = CompositeEmotion()
            return c + self + other
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return deepcopy(self)
        if isinstance(other, Emotion):
            return NotImplemented

        try:
            other = int(other)
            flow = self.emotional_flow / other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return deepcopy(self)
        if isinstance(other, Emotion):
            return NotImplemented
        try:
            other = int(other)
            flow = self.emotional_flow // other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return deepcopy(self)
        if isinstance(other, Emotion):
            if other._dimension == self._dimension:
                flow = other.emotional_flow - self.emotional_flow
                return self.emotion_from_flow(flow)
            return NotImplemented
        try:
            other = int(other)
            flow = self.emotional_flow - other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return deepcopy(self)
        if isinstance(other, Emotion):
            if other._dimension == self._dimension:
                flow = other.emotional_flow + self.emotional_flow
                return self.emotion_from_flow(flow)
            return NotImplemented
        try:
            other = int(other)
            flow = self.emotional_flow + other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    # TODO
    # (+=, -=, *=, @=, /=, //=, %=, **=, <<=, >>=, &=, ^=, |=).
    #    object.__iadd__(self, other)
    # object.__isub__(self, other)
    # object.__imul__(self, other)
    # object.__imatmul__(self, other)¶
    # object.__itruediv__(self, other)
    # object.__ifloordiv__(self, other)
    # object.__imod__(self, other)
    # object.__ipow__(self, other[, modulo])
    # object.__ilshift__(self, other)
    # object.__irshift__(self, other)
    # object.__iand__(self, other)
    # object.__ixor__(self, other)
    # object.__ior__(self, other)

    def __bool__(self) -> bool:
        """True when this emotion has non-zero intensity (i.e. is not Neutrality).

        Truthiness reflects *presence*, not *valence* — fear and anger are both
        truthy even though they have negative flow.
        """
        return self.emotional_flow != 0

    def __neg__(self):
        # get opposite emotion
        return copy(self.opposite_emotion)

    def __pos__(self):
        if self.emotional_flow:
            return deepcopy(self)
        return copy(self.opposite_emotion)

    def __abs__(self):
        return Neutrality()

    def __int__(self):
        return self.emotional_flow + self.intensity_offset

    def __float__(self):
        return float(self.emotional_flow)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __eq__(self, other):
        if isinstance(other, Emotion):
            if other._dimension == self._dimension:
                return self.emotional_flow == other.emotional_flow
            return False
        return self._name == other

    def __ne__(self, other):
        if isinstance(other, Emotion):
            if other._dimension == self._dimension:
                return self.emotional_flow != other.emotional_flow
            return True
        return self._name != other

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __contains__(self, item):
        if isinstance(item, Neutrality):
            return True
        if isinstance(item, Emotion):
            return item.base_emotion in self.emotion_vector
        if isinstance(item, EmotionalDimension):
            return self._dimension == item
        if isinstance(item, str):
            return self._dimension.axis == item
        return NotImplemented


class Neutrality(Emotion):
    """The identity element of the emotion algebra — zero emotional flow.

    ``emotion + Neutrality() == emotion`` for all emotions.
    """

    def __init__(self, dimension: "Union[EmotionalDimension, str]" = "") -> None:
        Emotion.__init__(self, "neutrality", dimension)

    @property
    def intensity(self):
        return "null"

    @property
    def type(self):
        return "neutral"

    @property
    def kind(self):
        return "neutral"

    @property
    def valence(self):
        return 0

    @property
    def emotional_flow(self):
        return 0

    @property
    def is_primary(self):
        return True

    @property
    def base_emotion(self):
        return self

    @property
    def opposite_emotion(self):
        return self

    @property
    def parent_emotion(self):
        return None

    def __add__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, str):
            return self.name + other
        if isinstance(other, Emotion):
            if self.dimension:
                other._kind = other.kind
                other._dimension = self._dimension
                other._name = self.name + " " + other.name

        return other

    def __sub__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Emotion):
            return - other
        return other

    def __eq__(self, other):
        if isinstance(other, bool):
            return True
        if isinstance(other, Emotion):
            if other.emotional_flow == 0:
                return True
            return False
        if isinstance(other, int) or isinstance(other, float):
            return other == 0
        if isinstance(other, str):
            return other == self.name
        from emotion_data.feelings import Feeling
        if isinstance(other, list) or isinstance(other, tuple) or isinstance(other, Feeling):
            return len(other) == 0
        return NotImplemented

    def __len__(self):
        return 0


class EmotionalDimension(object):
    """One axis of the Hourglass of Emotions model.

    Each dimension has three intensity levels (basic, mild, intense) on each polarity.

    Attributes
    ----------
    axis:
        One of ``"sensitivity"``, ``"attention"``, ``"pleasantness"``, ``"aptitude"``.
    basic_emotion / basic_opposite:
        Primary (flow ±1) emotions.
    mild_emotion / mild_opposite:
        Secondary (flow ±2) emotions.
    intense_emotion / intense_opposite:
        Tertiary (flow ±3) emotions.
    """

    def __init__(self) -> None:
        self.axis: str = ""  # sensitivity, attention, pleasantness, aptitude
        self.mild_emotion: "Union[Emotion, None]" = None
        self.mild_opposite: "Union[Emotion, None]" = None
        self.basic_emotion: "Union[Emotion, None]" = None
        self.basic_opposite: "Union[Emotion, None]" = None
        self.intense_emotion: "Union[Emotion, None]" = None
        self.intense_opposite: "Union[Emotion, None]" = None

    @property
    def name(self):
        return str(self.axis)

    @property
    def valence(self):
        # valence by dimension
        if "sensitivity" in self.axis:
            return - 1
        if "attention" in self.axis:
            return 1
        return 0

    @property
    def kind(self):
        # type by dimension
        if "sensitivity" in self.axis:
            return "negative"
        if "attention" in self.axis:
            return "positive"
        return "neutral"

    def __str__(self):
        return self.name

    def __repr__(self):
        return "DimensionObject:" + self.name

    def __add__(self, other):
        from emotion_data.composite_emotions import CompositeDimension
        if isinstance(other, EmotionalDimension):
            d = CompositeDimension()
            return d + self + other
        return NotImplemented

    def __contains__(self, item):
        if isinstance(item, Neutrality):
            return True
        if isinstance(item, Emotion):
            return item._dimension == self
        return False

    def __eq__(self, other):
        if isinstance(other, EmotionalDimension):
            if other.name == self.name:
                return True
            return False
        return self.name == other


def _get_dimensions():
    # map dimension name to object
    dimension_map = {}
    for d in HOURGLASS_OF_EMOTIONS:
        dimension = EmotionalDimension()
        dimension.axis = d

        # create the emotion objects
        dimension.intense_emotion = Emotion(HOURGLASS_OF_EMOTIONS[d][0].lower())
        dimension.mild_emotion = Emotion(HOURGLASS_OF_EMOTIONS[d][1].lower())
        dimension.basic_emotion = Emotion(HOURGLASS_OF_EMOTIONS[d][2].lower())
        dimension.basic_opposite = Emotion(HOURGLASS_OF_EMOTIONS[d][3].lower())
        dimension.mild_opposite = Emotion(HOURGLASS_OF_EMOTIONS[d][4].lower())
        dimension.intense_opposite = Emotion(HOURGLASS_OF_EMOTIONS[d][5].lower())

        # pass the dimention reference to the emotion
        dimension.basic_emotion._dimension = dimension
        dimension.basic_opposite._dimension = dimension
        dimension.mild_emotion._dimension = dimension
        dimension.mild_opposite._dimension = dimension
        dimension.intense_emotion._dimension = dimension
        dimension.intense_opposite._dimension = dimension

        dimension_map[d] = dimension
    return dimension_map


DIMENSIONS = _get_dimensions()


