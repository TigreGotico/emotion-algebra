"""emotion_data — public API and EmotionAnalyzer.

Optional integrations (deepmoji, tag) are lazy-imported inside the methods
that use them so that ``import emotion_data`` never fails when those optional
dependencies are absent.
"""
from emotion_data.emotions import get_emotion, emotion_to_dimension, random_emotion, get_dimension
from emotion_data.feelings import get_feeling, random_feeling
from emotion_data.lexicons import (
    get_word_emotion, get_sentiment, get_color, get_orientation, get_subjectivity,
)


class EmotionAnalyzer(object):
    """High-level facade over all emotion_data sub-modules."""

    @staticmethod
    def get_emotion(word: str):
        """Return the :class:`~emotion_data.plutchik.Emotion` for *word*, or ``None``."""
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
        """Return a random :class:`~emotion_data.plutchik.Emotion`."""
        return random_emotion()

    @staticmethod
    def random_feeling():
        """Return a random :class:`~emotion_data.feelings.Feeling`."""
        return random_feeling()

    @staticmethod
    def tag_emotions(sentence: str):
        """Tag *sentence* with emotions using deepmoji (requires optional dep)."""
        from emotion_data.deepmoji import get_emotions
        return get_emotions(sentence)

    @staticmethod
    def tag_emojis(sentence: str):
        """Tag *sentence* with emojis using deepmoji (requires optional dep)."""
        from emotion_data.deepmoji import get_emojis
        return get_emojis(sentence)

    @staticmethod
    def emotion(emotion_name: str):
        """Return the :class:`~emotion_data.plutchik.Emotion` for *emotion_name*, or ``None``."""
        return get_emotion(emotion_name)

    @staticmethod
    def feeling(feeling_name: str):
        """Return the :class:`~emotion_data.feelings.Feeling` for *feeling_name*, or ``None``."""
        return get_feeling(feeling_name)

    @staticmethod
    def dimension(dimension_name: str):
        """Return the :class:`~emotion_data.plutchik.EmotionalDimension` for *dimension_name*, or ``None``."""
        return get_dimension(dimension_name)

    @staticmethod
    def get(word: str):
        """Resolve *word* to an Emotion, Feeling, or EmotionalDimension — whichever matches first."""
        word = word.lower().strip()
        return get_emotion(word) or get_feeling(word) or get_dimension(word)
