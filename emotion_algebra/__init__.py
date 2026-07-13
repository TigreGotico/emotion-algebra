"""emotion_algebra — public API and EmotionAnalyzer."""
from emotion_algebra.base import EmotionBase
from emotion_algebra.emotions import get_emotion, emotion_to_dimension, random_emotion, get_dimension
from emotion_algebra.feelings import get_feeling, random_feeling
from emotion_algebra.lexicons import (
    get_word_emotion, get_sentiment, get_color, get_orientation, get_subjectivity,
)
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
from emotion_algebra.text import from_text, score_text
from emotion_algebra.emoji import (
    EMOJI_EMOTION_MAP, from_emoji, score_emojis, from_emojis, DeepMojiAdapter,
    register_emoji, unregister_emoji,
)
from emotion_algebra.text import score_mixed, from_mixed
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
from emotion_algebra.lexicons import get_afinn_score, get_hourglass, get_float_emotion
from emotion_algebra.deepmoji import DeepMojiONNXAdapter
from emotion_algebra.base import SupportsEmotionVector, hourglass_polarity, AXES, AXIS_MAX

# --- the affect core: the empirically-grounded model everything converts through
from emotion_algebra import evidence, views as _views  # noqa: F401 (wires the graph)
from emotion_algebra.evidence import Grade, grade_of
from emotion_algebra.affect import AffectState, CORE_AXES, ORIGIN, mixture
from emotion_algebra.prototypes import PROTOTYPES, prototype
from emotion_algebra.readout import label, dominant, entropy
from emotion_algebra.tendency import Mode, action_readiness, dominant_tendency
from emotion_algebra.homeostasis import (
    SET_POINT, Temperament, at_rest, drive, drive_magnitude, perturb, relax,
)
from emotion_algebra.neuro import NeuroState, MODULATORS, LOADINGS
from emotion_algebra.neural import affect_from_text, affect_from_texts
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
    "SET_POINT", "Temperament", "at_rest", "drive", "drive_magnitude",
    "perturb", "relax",
    "NeuroState", "MODULATORS", "LOADINGS",
    "affect_from_text", "affect_from_texts",
    "Fidelity", "convert", "fidelity", "explain_loss", "conversion_views",
    "evidence", "Grade", "grade_of",
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
    "from_text", "score_text", "from_mixed", "score_mixed",
    "EMOJI_EMOTION_MAP", "from_emoji", "from_emojis", "score_emojis",
    "register_emoji", "unregister_emoji",
    "DeepMojiAdapter", "DeepMojiONNXAdapter",
    # lexicons
    "get_word_emotion", "get_sentiment", "get_color", "get_orientation",
    "get_subjectivity", "get_afinn_score", "get_hourglass", "get_float_emotion",
    # serialization
    "serialization", "SCHEMA_VERSION", "UnsupportedSchemaError",
]


class EmotionAnalyzer(object):
    """High-level facade over all emotion_algebra sub-modules."""

    @staticmethod
    def get_emotion(word: str):
        """Return the :class:`~emotion_algebra.plutchik.Emotion` for *word*, or ``None``."""
        return get_emotion(word)

    @staticmethod
    def get_sentiment(word: str):
        """Return the sentiment label for *word* from the lexicon, or ``None``."""
        return get_sentiment(word)

    @staticmethod
    def get_color(word: str):
        """Return the colour association for *word* from the lexicon, or ``None``."""
        return get_color(word)

    @staticmethod
    def get_orientation(word: str):
        """Return the orientation label for *word* from the lexicon, or ``None``."""
        return get_orientation(word)

    @staticmethod
    def get_subjectivity(word: str):
        """Return the subjectivity label for *word* from the lexicon, or ``None``."""
        return get_subjectivity(word)

    @staticmethod
    def get_word_emotion(word: str):
        """Return the emotion label (string) for *word* from the lexicon, or ``None``."""
        return get_word_emotion(word)

    @staticmethod
    def random_emotion():
        """Return a random :class:`~emotion_algebra.plutchik.Emotion`."""
        return random_emotion()

    @staticmethod
    def random_feeling():
        """Return a random :class:`~emotion_algebra.feelings.Feeling`."""
        return random_feeling()

    @staticmethod
    def emotion(emotion_name: str):
        """Return the :class:`~emotion_algebra.plutchik.Emotion` for *emotion_name*, or ``None``."""
        return get_emotion(emotion_name)

    @staticmethod
    def feeling(feeling_name: str):
        """Return the :class:`~emotion_algebra.feelings.Feeling` for *feeling_name*, or ``None``."""
        return get_feeling(feeling_name)

    @staticmethod
    def dimension(dimension_name: str):
        """Return the :class:`~emotion_algebra.plutchik.EmotionalDimension` for *dimension_name*, or ``None``."""
        return get_dimension(dimension_name)

    @staticmethod
    def get(word: str):
        """Resolve *word* to an Emotion, Feeling, or EmotionalDimension — whichever matches first."""
        word = word.lower().strip()
        return get_emotion(word) or get_feeling(word) or get_dimension(word)

    @staticmethod
    def analyze_text(text: str):
        """Return the dominant :class:`~emotion_algebra.plutchik.Emotion` inferred from *text*."""
        return from_text(text)

    @staticmethod
    def score_text(text: str):
        """Return a full :class:`~emotion_algebra.state.EmotionalState` for *text*."""
        return score_text(text)

    @staticmethod
    def distance(a: EmotionBase, b: EmotionBase) -> float:
        """Euclidean distance between *a* and *b* in the 4-axis Hourglass space."""
        return emotion_distance(a, b)

    @staticmethod
    def appraise(appraisal: "Appraisal"):
        """Map a cognitive :class:`~emotion_algebra.appraisal.Appraisal` to a primary emotion."""
        return appraisal_to_emotion(appraisal)

    @staticmethod
    def from_emoji(emoji_char: str):
        """Return the :class:`~emotion_algebra.plutchik.Emotion` for *emoji_char*, or ``None``."""
        return from_emoji(emoji_char)

    @staticmethod
    def score_emojis(text: str):
        """Return a full :class:`~emotion_algebra.state.EmotionalState` from emoji characters in *text*."""
        return score_emojis(text)

    @staticmethod
    def analyze_emojis(text: str):
        """Return the dominant emotion inferred from emoji characters in *text*."""
        return from_emojis(text)

    @staticmethod
    def score_mixed(text: str):
        """Return a full :class:`~emotion_algebra.state.EmotionalState` from both words and emoji in *text*."""
        return score_mixed(text)

    @staticmethod
    def analyze_mixed(text: str):
        """Return the dominant emotion from both word lexicon and emoji signals in *text*."""
        return from_mixed(text)
