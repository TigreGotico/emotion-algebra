"""emotion_algebra — public API and EmotionAnalyzer."""
from emotion_algebra.base import EmotionBase
from emotion_algebra.emotions import get_emotion, emotion_to_dimension, random_emotion, get_dimension
from emotion_algebra.feelings import get_feeling, random_feeling
from emotion_algebra.distance import (
    emotion_distance, closest_emotion, emotion_clusters, emotion_similarity,
    MAX_EMOTION_DISTANCE,
)
from emotion_algebra.lovheim import (
    LovheimPoint, CORNERS, CORNER_ANCHORS, CORNER_POINTS, BASELINE, closest_affect,
)
from emotion_algebra.pad import PAD, to_pad, from_pad, pad_distance
from emotion_algebra.taxonomy import (
    resolve, is_ambiguous, describe_collision, collision_table, AMBIGUOUS_NAMES,
)
from emotion_algebra import serialization
from emotion_algebra.serialization import SCHEMA_VERSION, UnsupportedSchemaError
from emotion_algebra.state import EmotionalState, EmotionTimeline
from emotion_algebra.emoji import (
    EMOJI_EMOTION_MAP, from_emoji, score_emojis, from_emojis, DeepMojiAdapter,
    register_emoji, unregister_emoji,
)
from emotion_algebra.appraisal import (
    Appraisal, Agency, appraisal_to_emotion, appraisal_to_float_emotion,
    appraisal_to_lovheim, float_emotion_to_lovheim_deltas,
    float_emotion_to_neuro_deltas,
)
from emotion_algebra.needs import (
    CIADrive, MaxNeefNeed, MurrayNeed,
    need_deficit_to_emotion, need_deficit_to_float_emotion,
    NEED_DEFICIT_EMOTIONS, MAXNEEF_DEFICIT_EMOTIONS, MURRAY_DEFICIT_EMOTIONS,
)
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.deepmoji import DeepMojiONNXAdapter
from emotion_algebra.base import SupportsEmotionVector, hourglass_polarity, AXES, AXIS_MAX

# --- the affect core: the empirically-grounded model everything converts through
from emotion_algebra import evidence, views as _views  # noqa: F401 (wires the graph)
from emotion_algebra.evidence import Grade, grade_of
from emotion_algebra import provenance
from emotion_algebra.provenance import Provenance
from emotion_algebra.affect import AffectState, CORE_AXES, ORIGIN, mixture
from emotion_algebra.prototypes import PROTOTYPES, prototype
from emotion_algebra.readout import label, dominant, entropy
from emotion_algebra.tendency import Mode, action_readiness, dominant_tendency
from emotion_algebra.homeostasis import (
    SET_POINT, BASELINE_TEMPERAMENT, Temperament, at_rest, drive,
    drive_magnitude, perturb, relax,
)
from emotion_algebra.neuro import NeuroState, MODULATORS, LOADINGS
from emotion_algebra.neural import (
    affect_from_text, affect_from_texts, affect_from_features,
)
from emotion_algebra.lang import (
    CHANNELS, LanguageProfile, PROFILES, UnsupportedLanguageError,
    detect_language, typographic_features,
)
# Portuguese and Arabic. EXPERIMENTAL, and needs the `multilingual` extra — see
# the module docstring for what transfers and what does not.
from emotion_algebra import multilingual
from emotion_algebra.projection import (
    Fidelity, convert, fidelity, explain_loss, views as conversion_views,
)
from emotion_algebra.plutchik import Emotion, EmotionalDimension, Neutrality
from emotion_algebra.feelings import Feeling
from emotion_algebra.composite_emotions import CompositeEmotion, CompositeDimension

__all__ = [
    # facade
    "EmotionAnalyzer",
    # --- the affect core (empirically grounded; everything converts through it)
    "AffectState", "CORE_AXES", "ORIGIN", "mixture",
    "PROTOTYPES", "prototype",
    "label", "dominant", "entropy",
    "Mode", "action_readiness", "dominant_tendency",
    "SET_POINT", "BASELINE_TEMPERAMENT", "Temperament", "at_rest", "drive",
    "drive_magnitude",
    "perturb", "relax",
    "NeuroState", "MODULATORS", "LOADINGS",
    "affect_from_text", "affect_from_texts", "affect_from_features",
    # --- the language boundary: text -> AffectState, and nowhere else
    "CHANNELS", "LanguageProfile", "PROFILES", "UnsupportedLanguageError",
    "detect_language", "typographic_features",
    "multilingual",
    "Fidelity", "convert", "fidelity", "explain_loss", "conversion_views",
    "evidence", "Grade", "grade_of",
    "provenance", "Provenance",
    # core types
    "EmotionBase", "SupportsEmotionVector", "Emotion", "EmotionalDimension",
    "Neutrality", "Feeling", "CompositeEmotion", "CompositeDimension",
    "FloatEmotion", "EmotionalState", "EmotionTimeline",
    "AXES", "AXIS_MAX", "hourglass_polarity",
    # registries / lookup
    "get_emotion", "get_dimension", "get_feeling", "emotion_to_dimension",
    "random_emotion", "random_feeling",
    # taxonomy
    "resolve", "is_ambiguous", "describe_collision", "collision_table",
    "AMBIGUOUS_NAMES",
    # metric
    "emotion_distance", "emotion_similarity", "closest_emotion",
    "emotion_clusters", "MAX_EMOTION_DISTANCE",
    # neurochemistry
    "LovheimPoint", "CORNERS", "CORNER_ANCHORS", "CORNER_POINTS", "BASELINE",
    "closest_affect",
    # dimensional interop
    "PAD", "to_pad", "from_pad", "pad_distance",
    # appraisal / needs
    "Appraisal", "Agency", "appraisal_to_emotion", "appraisal_to_float_emotion",
    "appraisal_to_lovheim", "float_emotion_to_lovheim_deltas",
    "float_emotion_to_neuro_deltas",
    "CIADrive", "MaxNeefNeed", "MurrayNeed", "need_deficit_to_emotion",
    "need_deficit_to_float_emotion", "NEED_DEFICIT_EMOTIONS",
    "MAXNEEF_DEFICIT_EMOTIONS", "MURRAY_DEFICIT_EMOTIONS",
    # text / emoji
    "EMOJI_EMOTION_MAP", "from_emoji", "from_emojis", "score_emojis",
    "register_emoji", "unregister_emoji",
    "DeepMojiAdapter", "DeepMojiONNXAdapter",
    # serialization
    "serialization", "SCHEMA_VERSION", "UnsupportedSchemaError",
]


class EmotionAnalyzer(object):
    """High-level facade over the library.

    A convenience wrapper for the common operations. Everything here is also
    available as a plain function — see the module's ``__all__``.
    """

    # ------------------------------------------------------------------
    # The affect core
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(text: str, lang: str = "en"):
        """Return the :class:`~emotion_algebra.affect.AffectState` of *text*.

        Uses the DeepMoji probe, which is the only text path that recovers
        ``potency`` — the axis separating a user who will escalate from one who
        will quietly leave.

        English only; anything else raises
        :class:`~emotion_algebra.lang.UnsupportedLanguageError` rather than
        returning a number it cannot stand behind.
        """
        return affect_from_text(text, lang=lang)

    @staticmethod
    def analyze_all(texts, lang: str = "en"):
        """Batched :meth:`analyze`. Prefer this for more than one string."""
        return affect_from_texts(texts, lang=lang)

    @staticmethod
    def label(state):
        """Return ``P(name | state)`` — a distribution, not a single label."""
        return label(state)

    @staticmethod
    def dominant(state):
        """The single most probable emotion name for *state*."""
        return dominant(state)

    @staticmethod
    def tendency(state):
        """What the organism is getting ready to *do* — see :mod:`~emotion_algebra.tendency`."""
        return dominant_tendency(state)

    @staticmethod
    def prototype(name: str):
        """The core coordinates of a named emotion."""
        return prototype(name)

    @staticmethod
    def appraise(appraisal: "Appraisal"):
        """Map a cognitive :class:`~emotion_algebra.appraisal.Appraisal` to the core."""
        from emotion_algebra.appraisal import appraisal_to_affect

        return appraisal_to_affect(appraisal)

    @staticmethod
    def neuro(state):
        """The neuromodulator readout of an :class:`~emotion_algebra.affect.AffectState`."""
        return NeuroState.from_affect(state)

    # ------------------------------------------------------------------
    # Emoji
    # ------------------------------------------------------------------

    @staticmethod
    def from_emoji(emoji_char: str):
        """Return the emotion for *emoji_char*, or ``None``."""
        return from_emoji(emoji_char)

    @staticmethod
    def score_emojis(text: str):
        """Return an :class:`~emotion_algebra.state.EmotionalState` from emoji in *text*."""
        return score_emojis(text)

    # ------------------------------------------------------------------
    # The other models (see docs/models.md)
    # ------------------------------------------------------------------

    @staticmethod
    def emotion(emotion_name: str):
        """Look up a Plutchik :class:`~emotion_algebra.plutchik.Emotion` by name."""
        return get_emotion(emotion_name)

    @staticmethod
    def feeling(feeling_name: str):
        """Look up a Plutchik dyad (:class:`~emotion_algebra.feelings.Feeling`) by name."""
        return get_feeling(feeling_name)

    @staticmethod
    def dimension(dimension_name: str):
        """Look up an :class:`~emotion_algebra.plutchik.EmotionalDimension` by name."""
        return get_dimension(dimension_name)

    @staticmethod
    def random_emotion():
        """A random Plutchik :class:`~emotion_algebra.plutchik.Emotion`."""
        return random_emotion()

    @staticmethod
    def random_feeling():
        """A random Plutchik dyad."""
        return random_feeling()

    @staticmethod
    def get(name: str):
        """Resolve *name* to an Emotion, Feeling, or EmotionalDimension."""
        name = name.lower().strip()
        return get_emotion(name) or get_feeling(name) or get_dimension(name)

    @staticmethod
    def resolve(name: str):
        """Resolve a name across both dyad lineages — see :mod:`~emotion_algebra.taxonomy`."""
        return resolve(name)

    @staticmethod
    def convert(obj, source: str, target: str):
        """Convert between any two registered models."""
        return convert(obj, source, target)
