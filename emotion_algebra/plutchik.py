"""Core emotion algebra — Plutchik's Wheel + Cambria's Hourglass of Emotions.

Model provenance
----------------
This module blends two distinct theoretical frameworks:

**Plutchik's Wheel of Emotions** (Robert Plutchik, 1980)
    - 8 primary emotions arranged in opposite pairs on a colour wheel.
    - Dyad composition: adjacent emotions combine into named feelings
      (joy + trust → love, anticipation + joy → optimism).
    - Intensity cone: each primary has a mild (petals) and intense (inner) form.
    - Source: ``FEELING_NAMES`` (feelings.py), ``COMPOSITE_EMOTIONS_NAMES`` (composite_emotions.py).

**Hourglass of Emotions** (Cambria et al., 2012)
    - Maps emotions onto 4 independent signed-integer axes:
      Pleasantness, Attention, Sensitivity, Aptitude (PASA).
    - Each axis runs from −3 (intense negative) to +3 (intense positive).
    - Designed for computational sentiment analysis, not psychological modelling.
    - Source: ``HOURGLASS_OF_EMOTIONS``, ``EmotionalDimension``, ``Emotion.emotional_flow``.

Design decisions (per SPECIFICATION.md)
----------------------------------------
- ``Emotion.valence`` = Pleasantness axis component **only**.  Anger (Sensitivity)
  has valence=0 because arousal and valence are orthogonal (Posner et al. 2005).
- ``Emotion.arousal`` = ``|emotional_flow|`` — activation intensity, axis-independent.
- ``Emotion.type`` uses Russell's (1980) Circumplex quadrants: excited/calm ×
  positive/negative, activated neutral, neutral.
- ``EmotionalDimension.valence``: Sensitivity=0, Attention=0 (reactivity ⊥ hedonics).
- Cross-axis ``+`` returns :class:`~emotion_data.composite_emotions.CompositeEmotion`.
- Behaviour → Emotion mapping skips the cognitive appraisal layer (Lazarus 1991).

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

from emotion_algebra.base import EmotionBase
from emotion_algebra.reference_maps import EMOTION_CONTRASTS


def _circumplex_type(valence: int, arousal: int) -> str:
    """Russell (1980) Circumplex classification from valence and arousal.

    Parameters
    ----------
    valence:
        Hedonic tone (Pleasantness component).  Positive = pleasant, negative = unpleasant.
    arousal:
        Activation intensity (|emotional_flow|).  0 = inert, 1 = low, 2–3 = high.

    Returns
    -------
    str
        One of: ``"neutral"``, ``"activated neutral"``, ``"excited positive"``,
        ``"excited negative"``, ``"calm positive"``, ``"calm negative"``.
    """
    if arousal == 0:
        return "neutral"
    if valence == 0:
        return "activated neutral"
    if valence > 0 and arousal > 1:
        return "excited positive"
    if valence < 0 and arousal > 1:
        return "excited negative"
    if valence > 0:
        return "calm positive"
    return "calm negative"


PRIMARY_EMOTION_NAMES = ["serenity", "pensiveness", "acceptance", "boredom", "apprehension", "annoyance", "distraction",
                         "interest"]
SECONDARY_EMOTIONS_NAMES = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

TERTIARY_EMOTIONS_NAMES = ["ecstasy", "grief", "admiration", "loathing", "terror", "rage", "amazement", "vigilance"]

HOURGLASS_OF_EMOTIONS = {"sensitivity": ["rage", "anger", "annoyance", "apprehension", "fear", "terror"],
                         "attention": ["vigilance", "anticipation", "interest", "distraction", "surprise", "amazement"],
                         "pleasantness": ["ecstasy", "joy", "serenity", "pensiveness", "sadness", "grief"],
                         "aptitude": ["admiration", "trust", "acceptance", "boredom", "disgust", "loathing"]}


# kind assignments are manually curated from Plutchik's prose (approximate)

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
# "frivolity" (ecstasy+amazement): an intense-pleasant reaction to a
# surprising event, alongside ecstasy/joy/elation already in this bucket —
# not inherently social (Ortony/OCC classifies it as event-based, not a
# fortunes-of-others emotion).
EMOTION_KIND_NAMES['event related'].append("frivolity")

EMOTION_KIND_NAMES['social'].append("coercion")
EMOTION_KIND_NAMES['social'].append("trust")
EMOTION_KIND_NAMES['social'].append("submission")
EMOTION_KIND_NAMES['social'].append("rivalry")
EMOTION_KIND_NAMES['social'].append("rejection")
# "gloat" (ecstasy+loathing): joy at another's misfortune — Ortony/OCC
# classifies Schadenfreude as a fortunes-of-others emotion, i.e. social.
EMOTION_KIND_NAMES['social'].append("gloat")

EMOTION_KIND_NAMES['future appraisal'].append("pensiveness")
EMOTION_KIND_NAMES['future appraisal'].append("optimism")
EMOTION_KIND_NAMES['future appraisal'].append("vigilance")
EMOTION_KIND_NAMES['future appraisal'].append("distraction")
EMOTION_KIND_NAMES['future appraisal'].append("serenity")
EMOTION_KIND_NAMES['future appraisal'].append("anticipation")


class Emotion(EmotionBase):
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
        from emotion_algebra.behaviour import REACTION_TO_EMOTION_MAP, REACTIONS
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
    def type(self) -> str:
        """Russell (1980) Circumplex classification.

        Uses ``self.valence`` (Pleasantness component) and ``self.arousal``
        (``|emotional_flow|``) to place this emotion in one of six categories:

        - ``"excited positive"``  — high arousal, pleasant (joy, ecstasy)
        - ``"excited negative"``  — high arousal, unpleasant (grief, sadness)
        - ``"calm positive"``     — low arousal, pleasant (serenity)
        - ``"calm negative"``     — low arousal, unpleasant (pensiveness)
        - ``"activated neutral"`` — nonzero arousal, no hedonic polarity (anger, fear)
        - ``"neutral"``           — zero arousal (Neutrality)
        """
        return _circumplex_type(self.valence, self.arousal)


    @property
    def kind(self):
        # kind assignments are manually curated from Plutchik's prose (approximate)
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
    def valence(self) -> int:
        """Hedonic tone: the Pleasantness axis component only.

        Per Cambria (2012), Pleasantness is the hedonic axis.  Emotions on
        Sensitivity or Attention axes return 0 — reactivity is orthogonal to
        hedonics (Posner et al. 2005, Russell 1980).

        Examples
        --------
        joy.valence   → +2  (Pleasantness +2)
        anger.valence → 0   (Sensitivity axis, no hedonic component)
        sadness.valence → -2
        """
        if self._dimension and self._dimension.axis == "pleasantness":
            return self.emotional_flow
        return 0

    @property
    def arousal(self) -> int:
        """Activation intensity: ``|emotional_flow|``, axis-independent.

        Maps to the arousal dimension of Russell's Circumplex (1980).
        Range: 0 (neutral) to 3 (intense), higher for hyper-emotions.
        """
        return abs(self.emotional_flow)

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
        offset = max(0, abs(flow) - 3)
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

    @staticmethod
    def string_to_emotion(string=""):
        from emotion_algebra.emotions import EMOTIONS
        from emotion_algebra.feelings import FEELINGS
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

    def __len__(self):
        return len([e for e in self.emotion_vector if e.emotional_flow != 0])

    def __add__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
            if isinstance(other, str):
                return self.name + other
        if isinstance(other, Neutrality):
            return deepcopy(self)

        if isinstance(other, Emotion):
            if other._dimension == self._dimension:
                flow = other.emotional_flow + self.emotional_flow
                return self.emotion_from_flow(flow)
            # Cross-axis: return CompositeEmotion (spec §3.3)
            from emotion_algebra.composite_emotions import CompositeEmotion
            c = CompositeEmotion()
            return c + self + other

        from emotion_algebra.feelings import Feeling
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
                return NotImplemented
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
            from emotion_algebra.composite_emotions import CompositeEmotion
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
                flow = self.emotional_flow - other.emotional_flow
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

    def __float__(self) -> float:
        return float(self.emotional_flow)

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

    def __hash__(self):
        # Emotions are immutable value objects; hash consistently with
        # __eq__ (same dimension + same flow == equal), so they are usable
        # as dict keys and in sets.
        dimension_name = self._dimension.axis if self._dimension else None
        return hash((dimension_name, self.emotional_flow))

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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {"type": "emotion", "name": self._name}

    @classmethod
    def from_dict(cls, data: dict) -> "Emotion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        from emotion_algebra.emotions import get_emotion
        return get_emotion(data["name"])


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
        if isinstance(other, (int, float)) and self.dimension:
            # Neutrality is the additive identity: Neutrality(dim) + flow
            # must construct an Emotion at that flow on that dimension, not
            # fall through to the bare int.
            return self.dimension.basic_emotion.emotion_from_flow(other)

        return other

    def __sub__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Emotion):
            return - other
        if isinstance(other, (int, float)) and self.dimension:
            return self.dimension.basic_emotion.emotion_from_flow(-other)
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
        from emotion_algebra.feelings import Feeling
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
    def valence(self) -> int:
        """Hedonic sign of the positive pole of this dimension.

        Pleasantness is explicitly hedonic (+1).  Aptitude (competence) carries
        a social-hedonic valence (+1 = desirable, -1 = aversive) per Cambria (2012).
        Sensitivity (reactivity: anger/fear) and Attention (engagement: vigilance/
        surprise) encode arousal, which is orthogonal to hedonics (Posner et al. 2005).
        """
        if self.axis == "pleasantness":
            return 1
        if self.axis == "aptitude":
            return 1
        return 0

    @property
    def kind(self) -> str:
        """Hedonic classification of this axis."""
        if self.axis in ("pleasantness", "aptitude"):
            return "hedonic"
        return "activation"

    def __str__(self):
        return self.name

    def __repr__(self):
        return "DimensionObject:" + self.name

    def __add__(self, other):
        from emotion_algebra.composite_emotions import CompositeDimension
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


