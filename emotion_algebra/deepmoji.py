"""deepmoji-onnx integration — text → emoji probability → emotion.

:class:`DeepMojiONNXAdapter` wraps ``deepmoji-onnx`` (local package at
``Machine Learning Workspace/deepmoji-onnx``) and converts its emoji-probability
output to :class:`~emotion_algebra.base.EmotionBase` /
:class:`~emotion_algebra.state.EmotionalState` using the existing
:class:`~emotion_algebra.emoji.DeepMojiAdapter` pipeline.

Requires ``deepmoji-onnx`` to be importable (install it in editable mode):

    uv pip install -e "/path/to/deepmoji-onnx"

The model is downloaded from HuggingFace on first use (see
``DeepMojiONNX.from_pretrained``).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.state import EmotionalState


class DeepMojiONNXAdapter:
    """Bridge from ``deepmoji-onnx`` to emotion-algebra.

    Parameters
    ----------
    variant:
        Model precision variant — ``'fp32'`` (default), ``'fp16'``, or ``'int8'``.
    cache_dir:
        Local directory for the downloaded ONNX model.  Defaults to
        ``$DEEPMOJI_CACHE`` or ``~/.cache/deepmoji``.
    top_k:
        Number of top emoji predictions to consider when scoring a sentence.
    """

    def __init__(
        self,
        variant: str = "fp32",
        cache_dir: Optional[str] = None,
        top_k: int = 10,
    ) -> None:
        from deepmoji_onnx import DeepMojiONNX
        self._model = DeepMojiONNX.from_pretrained(variant=variant, cache_dir=cache_dir)
        self._top_k = top_k
        from emotion_algebra.emoji import DeepMojiAdapter
        self._adapter = DeepMojiAdapter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Optional["EmotionBase"]:
        """Return the dominant emotion for *text*.

        Runs DeepMoji emoji prediction, then maps the top-k emoji
        probabilities to a Plutchik emotion via :class:`~emotion_algebra.emoji.DeepMojiAdapter`.

        Parameters
        ----------
        text:
            Input string (a single sentence works best).

        Returns
        -------
        Emotion or None
            ``None`` if none of the top emoji map to a known emotion.
        """
        scores = self._top_scores(text)
        return self._adapter.from_scores(scores)

    def score(self, text: str) -> "EmotionalState":
        """Return a full :class:`~emotion_algebra.state.EmotionalState` for *text*.

        All top-k emoji contribute proportionally to their probability.

        Parameters
        ----------
        text:
            Input string.

        Returns
        -------
        EmotionalState
            Zero-vector if no emoji map to a known emotion.
        """
        scores = self._top_scores(text)
        return self._adapter.score_state(scores)

    def top_emoji_scores(self, text: str) -> dict[str, float]:
        """Return the raw ``{emoji: probability}`` dict from DeepMoji for *text*.

        Useful for inspecting raw model output before emotion mapping.
        """
        return self._top_scores(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _top_scores(self, text: str) -> dict[str, float]:
        """Run model on *text* and return top-k emoji probability dict."""
        results = self._model.top_emojis([text], k=self._top_k)
        return results[0] if results else {}
