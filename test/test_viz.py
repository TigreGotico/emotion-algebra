"""Tests for the plotting helpers.

These render headless (Agg) and assert the figures build without error — the
point is that the drawing code stays runnable, not that pixels match.
"""
import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from emotion_algebra.emotions import get_emotion  # noqa: E402
from emotion_algebra.lovheim import LovheimPoint  # noqa: E402
from emotion_algebra.state import EmotionalState, EmotionTimeline  # noqa: E402
from emotion_algebra.viz import (  # noqa: E402
    plot_circumplex,
    plot_lovheim,
    plot_timeline,
    plot_wheel,
)


@pytest.fixture
def timeline():
    tl = EmotionTimeline(label="demo")
    for name in ("joy", "anger", "grief"):
        state = EmotionalState()
        state.apply(get_emotion(name))
        tl.append(state)
    return tl


class TestWheel:
    def test_bare_wheel_renders(self):
        assert isinstance(plot_wheel(), Figure)

    def test_highlighted_wheel_renders(self):
        assert isinstance(plot_wheel("joy"), Figure)

    def test_accepts_an_emotion_object(self):
        assert isinstance(plot_wheel(get_emotion("rage")), Figure)

    def test_title_is_used(self):
        fig = plot_wheel("joy", title="custom")
        assert fig.axes[0].get_title() == "custom"

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError):
            plot_wheel("definitely-not-an-emotion")


class TestCircumplex:
    def test_all_emotions_render(self):
        assert isinstance(plot_circumplex(), Figure)

    def test_selected_emotions_render(self):
        assert isinstance(plot_circumplex("joy", "terror", get_emotion("rage")), Figure)


class TestTimeline:
    def test_renders(self, timeline):
        assert isinstance(plot_timeline(timeline), Figure)

    def test_labels_all_four_axes(self, timeline):
        fig = plot_timeline(timeline)
        assert len(fig.axes[0].get_lines()) >= 4

    def test_empty_timeline_rejected(self):
        with pytest.raises(ValueError):
            plot_timeline(EmotionTimeline())


class TestLovheimCube:
    def test_renders_from_a_point(self):
        assert isinstance(plot_lovheim(LovheimPoint(0.9, 0.9, 0.1)), Figure)

    def test_renders_from_a_state(self):
        state = EmotionalState()
        state.apply(get_emotion("ecstasy"))
        assert isinstance(plot_lovheim(state), Figure)
