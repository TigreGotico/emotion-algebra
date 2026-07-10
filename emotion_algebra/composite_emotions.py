"""Composite emotions — multi-dimensional emotional states spanning two Hourglass axes.

A :class:`CompositeEmotion` is constructed by multiplying two :class:`Emotion` objects
from *different* Hourglass dimensions (``rage * vigilance → aggressiveness``).

Model provenance
----------------
``COMPOSITE_EMOTIONS_NAMES`` lists the named 2-axis combinations from Plutchik's
Wheel.  The numeric algebra (``emotional_flow``, ``emotion_vector``, operators) uses
the Hourglass integer scale from Cambria et al. (2012).

The ``type`` property in this module is a **heuristic approximation** — it
classifies composites by the sign of their net flow and which axes are involved,
but this mapping has no direct basis in either Plutchik or Cambria.  It is retained
for downstream convenience but should not be treated as scientifically grounded.

Note on ``CompositeEmotion`` vs ``Feeling``
-------------------------------------------
Both represent combinations of basic emotions.  ``Feeling`` (feelings.py) models
Plutchik's named dyads (two adjacent wheel sectors) with a flat emotion list.
``CompositeEmotion`` (this file) models two-dimensional Hourglass combinations and
inherits the full ``Emotion`` operator algebra.  They are related but not identical
concepts; a full merge would break the public API.

Classes
-------
CompositeEmotion
    Emotion spanning two :class:`EmotionalDimension` axes.
CompositeDimension
    Paired dimension for composite emotion lookup.
"""
from __future__ import annotations

from copy import copy, deepcopy
from typing import List, Optional, Union

import numpy as np

from emotion_algebra.base import EmotionBase
from emotion_algebra.plutchik import Emotion, EmotionalDimension, DIMENSIONS, Neutrality, _circumplex_type
from emotion_algebra.feelings import Feeling


COMPOSITE_EMOTIONS_NAMES = {
    "aggressiveness": ["rage", "vigilance"],
    "rejection": ["rage", "amazement"],
    "rivalry": ["rage", "admiration"],
    "contempt": ["rage", "loathing"],

    "anxiety": ["terror", "vigilance"],
    "awe": ["terror", "amazement"],
    "submission": ["terror", "admiration"],
    "coercion": ["terror", "loathing"],

    "optimism": ["ecstasy", "vigilance"],
    "frivolity": ["ecstasy", "amazement"],
    "love": ["ecstasy", "admiration"],
    "gloat": ["ecstasy", "loathing"],

    "frustration": ["grief", "vigilance"],
    "disapproval": ["grief", "amazement"],
    "envy": ["grief", "admiration"],
    "remorse": ["grief", "loathing"]
}

OPPOSITE_EMOTIONS_NAMES = {
    "annoyance": "apprehension",
    "interest": "distraction",
    "serenity": "pensiveness",
    "acceptance": "boredom",
    "trust": "disgust",
    "joy": "sadness",
    "anticipation": "surprise",
    "anger": "fear",
    "rage": "terror",
    "vigilance": "amazement",
    "ecstasy": "grief",
    "admiration": "loathing",
    "apprehension": "annoyance",
    "distraction": "interest",
    "pensiveness": "serenity",
    "boredom": "acceptance",
    "disgust": "trust",
    "sadness": "joy",
    "surprise": "anticipation",
    "fear": "anger",
    "terror": "rage",
    "amazement": "vigilance",
    "grief": "ecstasy",
    "loathing": "admiration",
    # composite
    "contempt": "submission",
    "rivalry": "coercion",
    "anxiety": "rejection",
    "awe": "aggressiveness",
    "love": "remorse",
    "envy": "gloat",
    "frivolity": "frustration",
    "disapproval": "optimism",
    "submission": "contempt",
    "coercion": "rivalry",
    "rejection": "anxiety",
    "aggressiveness": "awe",
    "remorse": "love",
    "gloat": "envy",
    "frustration": "frivolity",
    "optimism": "disapproval"
}


class CompositeEmotion(EmotionBase):
    """An emotion spanning two :class:`EmotionalDimension` axes simultaneously.

    Built from named dyads in ``COMPOSITE_EMOTIONS_NAMES`` (e.g.
    ``aggressiveness == rage × vigilance``).

    Parameters
    ----------
    name:
        Optional explicit name; auto-resolved from ``COMPOSITE_EMOTIONS_NAMES`` if empty.
    """

    sensitivity_dimension = DIMENSIONS["sensitivity"]
    attention_dimension = DIMENSIONS["attention"]
    pleasantness_dimension = DIMENSIONS["pleasantness"]
    aptitude_dimension = DIMENSIONS["aptitude"]

    def __init__(self, name: str = "") -> None:
        self._name: str = name
        self._kind: str = ""
        self.components: List[Emotion] = []

    @staticmethod
    def string_to_emotion(string: str = "") -> "Union[Emotion, Feeling, str]":
        """Resolve a string to an Emotion or Feeling object, or return the string unchanged."""
        from emotion_algebra.emotions import EMOTIONS
        from emotion_algebra.feelings import FEELINGS
        if string in EMOTIONS:
            from copy import copy
            return copy(EMOTIONS[string])
        if string in FEELINGS:
            return FEELINGS[string]
        return string

    @staticmethod
    def get_composite_from_emotions(emotion1, emotion2):
        for c in COMPOSITE_EMOTIONS:
            emos = COMPOSITE_EMOTIONS[c]
            if emotion1 in emos and emotion2 in emos:
                return c
        return None

    @property
    def dimension(self):
        d = CompositeDimension()
        for dim in self.dimensions:
            d = d + dim
        return d

    @property
    def dimensions(self):
        return [e.dimension for e in self.components]

    @staticmethod
    def matrix_to_array(m):
        sensitivity = m.item(0, 0)
        attention = m.item(0, 1)
        pleasantness = m.item(1, 0)
        aptitude = m.item(1, 1)
        return np.array([sensitivity, attention, pleasantness, aptitude])

    @staticmethod
    def array_to_emotion(arr):
        sensitivity = Neutrality(dimension="sensitivity") + arr.item(0)
        attention = Neutrality(dimension="attention") + arr.item(1)
        pleasantness = Neutrality(dimension="pleasantness") + arr.item(2)
        aptitude = Neutrality(dimension="aptitude") + arr.item(3)
        return sensitivity + pleasantness + aptitude + attention

    @property
    def emotion_vector(self):
        sensitivity = Neutrality()
        attention = Neutrality()
        pleasantness = Neutrality()
        aptitude = Neutrality()
        for emotion in self.components:
            if emotion.dimension == self.sensitivity_dimension:
                sensitivity = sensitivity + emotion
            elif emotion.dimension == self.attention_dimension:
                attention = attention + emotion
            elif emotion.dimension == self.aptitude_dimension:
                aptitude = aptitude + emotion
            elif emotion.dimension == self.pleasantness_dimension:
                pleasantness = pleasantness + emotion

        return [sensitivity, attention, pleasantness, aptitude]

    @property
    def emotion_matrix(self):
        return self.as_matrix

    @property
    def name(self):
        name = ""
        if len(self.components) == 2:
            name = self.get_composite_from_emotions(self.components[0], self.components[1])
        return self._name or name or self.secondary_name

    @property
    def secondary_name(self):
        if len(self.components):
            name = ""
            for emo in self.components:
                dimension = emo.dimension.axis
                name += dimension + " of " + emo.name + ", "
            return name[:-2]

        return "neutrality"

    @property
    def valence(self) -> int:
        """Pleasantness-axis sum across all components (spec §2.4).

        Only components on the Pleasantness axis contribute to hedonic tone.
        """
        return sum(
            e.emotional_flow
            for e in self.components
            if e._dimension and e._dimension.axis == "pleasantness"
        )

    @property
    def arousal(self) -> int:
        """Peak activation across all components: ``max(|e.emotional_flow|)``."""
        if not self.components:
            return 0
        return max(abs(e.emotional_flow) for e in self.components)

    @property
    def type(self) -> str:
        """Russell (1980) Circumplex classification using composite valence and arousal."""
        return _circumplex_type(self.valence, self.arousal)

    @property
    def kind(self) -> str:
        """Category classification for this composite emotion."""
        return self._kind

    @property
    def base_emotion(self):
        c = CompositeEmotion()
        for e in self.emotion_vector:
            c = c + e
        return c

    @property
    def parent_emotion(self):
        return None

    @property
    def opposite_emotion(self):
        return self.__neg__()

    @property
    def emotional_flow(self) -> int:
        """Signed net intensity: sum of component flows along each axis.

        Positive when the aggregate flow is positive (approach), negative when
        aversive (avoidance), zero when the components cancel out.  Previously
        this was ``np.linalg.norm``, which is always ≥ 0 and made the negative
        branches of :attr:`type` permanently dead code.
        """
        return sum(e.emotional_flow for e in self.components)

    @property
    def is_composite(self):
        return True

    @property
    def equivalent_feeling(self):
        from emotion_algebra.feelings import Feeling
        if len(self.components) == 2:
            from emotion_algebra.feelings import FEELINGS_TO_EMOTION_MAP, FEELINGS
            for feel in FEELINGS_TO_EMOTION_MAP:
                feel = feel.lower()
                if self.components[0].name in [f.name.lower() for f in FEELINGS_TO_EMOTION_MAP[feel]] and self.components[
                    1].name in [f.name.lower() for f in FEELINGS_TO_EMOTION_MAP[feel]]:
                    return copy(FEELINGS[feel])
        f = Feeling()
        f.emotions = self.components
        return f

    def __str__(self):
        return self.name

    def __repr__(self):
        return "CompositeEmotionObject:" + self.name

    def __neg__(self):
        vector = [- e for e in self.emotion_vector]
        c = CompositeEmotion()
        for e in vector:
            c = c + e
        return c

    def __add__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)

        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo

        if isinstance(other, Feeling):
            c = deepcopy(self)
            for e in other.emotions:
                c = c + e
            return c

        if isinstance(other, Emotion):
            # matrix product of emotion vectors
            other_vector = other.emotion_vector

            result_vector = np.array(other_vector) + np.array(self.emotion_vector)
            c = CompositeEmotion()

            for e in result_vector:
                if e.dimension:
                    c.components.append(e)
            return c

        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)

        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo

        if isinstance(other, Feeling):
            c = deepcopy(self)
            for e in other.emotions:
                c = c - e
            return c

        if isinstance(other, Emotion):
            # matrix product of emotion vectors
            other_vector = other.emotion_vector

            result_vector = np.array(self.emotion_vector) - np.array(other_vector)
            c = CompositeEmotion()

            for e in result_vector:
                if e.dimension:
                    c.components.append(e)
            if len(c.components) == 1:
                return c.components[0]
            if not len(c.components):
                return Neutrality()
            return c

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo
        if isinstance(other, Emotion):
            m = np.matmul(self.as_matrix, other.as_matrix)

            return m
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo
        elif isinstance(other, CompositeEmotion):
            return NotImplemented
        elif isinstance(other, Emotion):
            emo = deepcopy(self)
            if other in emo.components:
                emo.components.remove(other)
                return emo
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo
        if isinstance(other, CompositeEmotion):
            return NotImplemented
        elif isinstance(other, Emotion):
            emo = deepcopy(self)
            if other in emo.components:
                emo.components.remove(other)
                return emo
        return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            emo = deepcopy(self)
            return emo
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Emotion):
            return False
        return self.name == other

    def __ne__(self, other):
        if isinstance(other, Emotion):
            return True
        return self.name != other

    def __contains__(self, item):
        if isinstance(item, str):
            item = self.string_to_emotion(item)
        if isinstance(item, Emotion):
            return item in self.components
        if isinstance(item, EmotionalDimension):
            return self.dimension == item
        return NotImplemented

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "composite",
            "name": self._name,
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompositeEmotion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        name = data.get("name", "")
        if name in COMPOSITE_EMOTIONS_NAMES:
            from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS
            return COMPOSITE_EMOTIONS[name]
        comp = cls(name)
        from emotion_algebra.emotions import get_emotion
        comp.components = [
            get_emotion(c["name"]) for c in data.get("components", [])
            if get_emotion(c["name"]) is not None
        ]
        return comp


class CompositeDimension(object):
    """A paired :class:`EmotionalDimension` used for composite emotion resolution."""

    def __init__(self) -> None:
        self.dimensions: List[EmotionalDimension] = []

    @property
    def name(self):
        name = "/"
        for d in self.dimensions:
            name += d.axis + "/"
        return name[1:-1]

    @property
    def intense_emotion(self):
        if self.name == "attention/sensitivity" or self.name == "sensitivity/attention":
            return COMPOSITE_EMOTIONS["aggressiveness"]
        if self.name == "aptitude/sensitivity" or self.name ==  "sensitivity/aptitude":
            return COMPOSITE_EMOTIONS["rivalry"]
        if self.name == "pleasantness/attention" or self.name == "attention/pleasantness":
            return COMPOSITE_EMOTIONS["optimism"]
        if self.name == "pleasantness/aptitude" or self.name == "aptitude/pleasantness":
            return COMPOSITE_EMOTIONS["love"]
        # not yet mapped for this dimension combination
        return None

    @property
    def intense_opposite(self):
        if self.name == "attention/sensitivity" or self.name == "sensitivity/attention":
            return COMPOSITE_EMOTIONS["awe"]
        if self.name == "aptitude/sensitivity" or self.name == "sensitivity/aptitude":
            return COMPOSITE_EMOTIONS["coercion"]
        if self.name == "pleasantness/attention" or self.name == "attention/pleasantness":
            return COMPOSITE_EMOTIONS["disapproval"]
        if self.name == "pleasantness/aptitude" or self.name == "aptitude/pleasantness":
            return COMPOSITE_EMOTIONS["remorse"]
        # not yet mapped for this dimension combination
        return None

    @property
    def mild_emotion(self):
        if self.name == "attention/sensitivity" or self.name == "sensitivity/attention":
            return COMPOSITE_EMOTIONS["anxiety"]
        if self.name == "aptitude/sensitivity" or self.name == "sensitivity/aptitude":
            return COMPOSITE_EMOTIONS["submission"]
        if self.name == "pleasantness/attention" or self.name == "attention/pleasantness":
            return COMPOSITE_EMOTIONS["frustration"]
        if self.name == "pleasantness/aptitude" or self.name == "aptitude/pleasantness":
            return COMPOSITE_EMOTIONS["envy"]
        # not yet mapped for this dimension combination
        return None

    @property
    def mild_opposite(self):
        if self.name == "attention/sensitivity" or self.name == "sensitivity/attention":
            return COMPOSITE_EMOTIONS["rejection"]
        if self.name == "aptitude/sensitivity" or self.name == "sensitivity/aptitude":
            return COMPOSITE_EMOTIONS["contempt"]
        if self.name == "pleasantness/attention" or self.name == "attention/pleasantness":
            return COMPOSITE_EMOTIONS["frivolity"]
        if self.name == "pleasantness/aptitude" or self.name == "aptitude/pleasantness":
            return COMPOSITE_EMOTIONS["gloat"]
        # not yet mapped for this dimension combination
        return None

    @property
    def basic_emotion(self):
        # not yet mapped for this dimension combination
        return None

    @property
    def basic_opposite(self):
        # not yet mapped for this dimension combination
        return None

    def __repr__(self):
        return "CompositeDimensionObject:" + self.name

    def __add__(self, other):
        if isinstance(other, EmotionalDimension):
            d = deepcopy(self)
            d.dimensions.append(other)
            return d
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, EmotionalDimension):
            d = deepcopy(self)
            if other in d.dimensions:
                d.dimensions.remove(other)
            else:
                return None
            if len(d.dimensions) == 1:
                return d.dimensions[0]
            return d
        return NotImplemented

    def __contains__(self, item):
        if isinstance(item, Neutrality):
            return True
        if isinstance(item, Emotion):
            for d in self.dimensions:
                if d in item:
                    return True
            return False
        if isinstance(item, EmotionalDimension):
            if item in self.dimensions:
                return True

        if isinstance(item, Feeling):
            for d in item.dimensions:
                if d not in self.dimensions:
                    return False
            return True

        return False


def _get_composites():
    bucket = {}
    from emotion_algebra.emotions import EMOTIONS
    for emo in COMPOSITE_EMOTIONS_NAMES:
        c = CompositeEmotion(emo)
        for e in COMPOSITE_EMOTIONS_NAMES[emo]:
            e = EMOTIONS[e]
            c.components.append(e)
        bucket[emo] = c
    return bucket


COMPOSITE_EMOTIONS = _get_composites()
