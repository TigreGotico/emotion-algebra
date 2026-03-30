"""emotion_algebra — public API and EmotionAnalyzer."""
from emotion_algebra.base import EmotionBase
from emotion_algebra.emotions import get_emotion, emotion_to_dimension, random_emotion, get_dimension
from emotion_algebra.feelings import get_feeling, random_feeling
from emotion_algebra.lexicons import (
    get_word_emotion, get_sentiment, get_color, get_orientation, get_subjectivity,
)
from emotion_algebra.distance import emotion_distance, closest_emotion, emotion_clusters
from emotion_algebra.state import EmotionalState, EmotionTimeline
from emotion_algebra.text import from_text, score_text
from emotion_algebra.emoji import (
    EMOJI_EMOTION_MAP, from_emoji, score_emojis, from_emojis, DeepMojiAdapter,
    register_emoji, unregister_emoji,
)
from emotion_algebra.text import score_mixed, from_mixed
from emotion_algebra.appraisal import Appraisal, appraisal_to_emotion
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.lexicons import get_afinn_score, get_hourglass, get_float_emotion
from emotion_algebra.deepmoji import DeepMojiONNXAdapter


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
