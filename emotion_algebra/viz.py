"""Plotting helpers — requires the ``[viz]`` extra (``matplotlib``).

Every function returns the :class:`matplotlib.figure.Figure` it drew, so callers
can save, embed, or further style it.  Nothing is shown or written to disk here;
that is the caller's choice.

    >>> from emotion_algebra.viz import plot_wheel          # doctest: +SKIP
    >>> plot_wheel("joy").savefig("wheel.png")              # doctest: +SKIP
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from emotion_algebra.base import AXIS_MAX

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

    from emotion_algebra.base import EmotionBase
    from emotion_algebra.state import EmotionalState, EmotionTimeline


_MISSING = (
    "emotion_algebra.viz requires matplotlib — install the extra:\n"
    "    uv pip install 'emotion-algebra[viz]'"
)


def _plt():
    """Import matplotlib lazily, with an actionable message when it is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via the [viz] extra
        raise ImportError(_MISSING) from exc
    return plt


def _resolve(emotion: Union[str, "EmotionBase"]) -> "EmotionBase":
    if not isinstance(emotion, str):
        return emotion
    from emotion_algebra.taxonomy import resolve

    found = resolve(emotion)
    if found is None:
        raise ValueError(f"unknown emotion: {emotion!r}")
    return found


#: The wheel's eight petals, in Plutchik's own circular order.
_PETALS = [
    ("joy", "#FFDE3A"),
    ("trust", "#8FD14F"),
    ("fear", "#2E8B57"),
    ("surprise", "#59C4E6"),
    ("sadness", "#4169E1"),
    ("disgust", "#9370DB"),
    ("anger", "#E8453C"),
    ("anticipation", "#F7941D"),
]


def plot_wheel(
    emotion: Optional[Union[str, "EmotionBase"]] = None,
    *,
    title: Optional[str] = None,
) -> "Figure":
    """Draw Plutchik's wheel, optionally highlighting one emotion.

    The three rings are the intensity tiers (mild, basic, intense); the eight
    wedges are the primaries.  When *emotion* is given, the petal it belongs to
    is drawn at full opacity and the rest are dimmed.

    Parameters
    ----------
    emotion:
        An emotion, a feeling, or a name.  ``None`` draws the bare wheel.
    title:
        Overrides the default title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()

    target = _resolve(emotion) if emotion is not None else None
    axis = getattr(getattr(target, "dimension", None), "name", None)
    flow = abs(int(getattr(target, "emotional_flow", 0) or 0)) if target else 0

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    width = 2 * np.pi / len(_PETALS)

    for i, (name, color) in enumerate(_PETALS):
        theta = i * width
        emo = _resolve(name)
        on_axis = axis is not None and emo.dimension is not None and emo.dimension.name == axis
        for tier in (3, 2, 1):  # outermost (intense) first
            lit = on_axis and (flow == 0 or tier <= max(1, min(3, flow)))
            ax.bar(
                theta + width / 2,
                tier,
                width=width * 0.95,
                bottom=0,
                color=color,
                alpha=(0.95 if lit else 0.18) * (0.55 + 0.15 * tier),
                edgecolor="white",
                linewidth=1.0,
                zorder=3 - tier,
            )
        ax.text(
            theta + width / 2,
            3.45,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold" if on_axis else "normal",
            color="#222" if on_axis else "#888",
        )

    ax.set_ylim(0, 3.9)
    ax.set_axis_off()
    if title is None:
        title = f"Plutchik's Wheel — {target.name}" if target else "Plutchik's Wheel"
    ax.set_title(title, pad=18, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_circumplex(*emotions: Union[str, "EmotionBase"], title: str = "") -> "Figure":
    """Plot emotions on Russell's circumplex (polarity × arousal).

    Parameters
    ----------
    *emotions:
        Any number of emotions or names.  With none given, all 24 named
        Plutchik emotions are plotted.
    title:
        Overrides the default title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()

    if emotions:
        subjects = [_resolve(e) for e in emotions]
    else:
        from emotion_algebra.emotions import EMOTIONS

        subjects = list(EMOTIONS.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color="#bbb", linewidth=1, zorder=1)
    ax.axvline(0, color="#bbb", linewidth=1, zorder=1)
    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, color="#ddd", zorder=0))

    for emotion in subjects:
        x = emotion.polarity
        y = emotion.arousal / AXIS_MAX
        ax.scatter(x, y, s=70, zorder=3, color="#E8453C" if x < 0 else "#2E8B57")
        ax.annotate(
            emotion.name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9
        )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("polarity  (unpleasant → pleasant)")
    ax.set_ylabel("arousal  (calm → activated)")
    ax.set_title(title or "Russell's Circumplex", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def plot_timeline(timeline: "EmotionTimeline", title: str = "") -> "Figure":
    """Plot the four Hourglass axes of an :class:`~emotion_algebra.state.EmotionTimeline`.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the timeline is empty — there is nothing to draw.
    """
    plt = _plt()
    from emotion_algebra.base import AXES

    if not len(timeline):
        raise ValueError("cannot plot an empty timeline")

    series = np.array(timeline.snapshots)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(AXES):
        ax.plot(series[:, i], marker="o", label=name, linewidth=2)

    ax.axhline(0, color="#bbb", linewidth=1)
    ax.set_ylim(-AXIS_MAX - 0.3, AXIS_MAX + 0.3)
    ax.set_xlabel("step")
    ax.set_ylabel("axis value")
    ax.set_title(title or (timeline.label or "Emotional timeline"), fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def plot_lovheim(state_or_point, title: str = "") -> "Figure":
    """Draw Lövheim's cube with its eight labelled corner affects.

    Parameters
    ----------
    state_or_point:
        A :class:`~emotion_algebra.lovheim.LovheimPoint`, or anything with a
        ``to_lovheim()`` method (an :class:`~emotion_algebra.state.EmotionalState`,
        a :class:`~emotion_algebra.float_emotion.FloatEmotion`).  The resulting
        point is marked inside the cube.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()
    from emotion_algebra.lovheim import CORNERS, LovheimPoint

    if not isinstance(state_or_point, LovheimPoint):
        state_or_point = state_or_point.to_lovheim()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for corner, name in CORNERS.items():
        ax.scatter(*corner, s=60, color="#444", zorder=2)
        ax.text(corner[0], corner[1], corner[2], f"  {name}", fontsize=8, color="#333")

    # The twelve edges.
    for a in CORNERS:
        for b in CORNERS:
            if sum(abs(x - y) for x, y in zip(a, b)) == 1 and a < b:
                ax.plot(*zip(a, b), color="#ccc", linewidth=1, zorder=1)

    point = state_or_point
    ax.scatter(
        point.serotonin,
        point.dopamine,
        point.noradrenaline,
        s=220,
        color="#E8453C",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
        label=point.closest_affect(),
    )

    ax.set_xlabel("serotonin")
    ax.set_ylabel("dopamine")
    ax.set_zlabel("noradrenaline")
    ax.set_title(title or f"Lövheim cube — {point.closest_affect()}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig
