"""Round-trip tests for the versioned serialization schema."""
import json

import pytest

from emotion_algebra.emotions import get_emotion
from emotion_algebra.feelings import get_feeling
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.lovheim import LovheimPoint
from emotion_algebra.serialization import (
    SCHEMA_KEY,
    SCHEMA_VERSION,
    UnsupportedSchemaError,
    from_dict,
    from_json,
    to_dict,
    to_json,
)
from emotion_algebra.state import EmotionalState, EmotionTimeline


def _state(*names):
    state = EmotionalState()
    for name in names:
        state.apply(get_emotion(name))
    return state


def _timeline():
    timeline = EmotionTimeline(label="demo")
    timeline.append(_state("joy"))
    timeline.append(_state("grief"))
    return timeline


#: Everything that carries a 4-axis vector — these round-trip by value.
VECTOR_SUBJECTS = [
    get_emotion("joy"),
    get_feeling("love"),
    FloatEmotion(1.0, -2.0, 0.5, 0.0),
    LovheimPoint(0.2, 0.8, 0.4),
]
#: The stateful types — they round-trip, but by snapshot rather than as_array.
ALL_SUBJECTS = VECTOR_SUBJECTS + [_state("joy"), _timeline()]

VEC_IDS = [type(s).__name__ for s in VECTOR_SUBJECTS]
ALL_IDS = [type(s).__name__ for s in ALL_SUBJECTS]


class TestRoundTrip:
    @pytest.mark.parametrize("obj", ALL_SUBJECTS, ids=ALL_IDS)
    def test_dict_round_trip_preserves_type(self, obj):
        assert type(from_dict(to_dict(obj))) is type(obj)

    @pytest.mark.parametrize("obj", VECTOR_SUBJECTS, ids=VEC_IDS)
    def test_json_round_trip_preserves_vector(self, obj):
        back = from_json(to_json(obj))
        assert list(back.as_array) == pytest.approx(list(obj.as_array))

    @pytest.mark.parametrize("obj", ALL_SUBJECTS, ids=ALL_IDS)
    def test_payload_is_json_serializable(self, obj):
        json.loads(to_json(obj))  # must not raise

    @pytest.mark.parametrize("obj", ALL_SUBJECTS, ids=ALL_IDS)
    def test_payload_is_stamped_and_tagged(self, obj):
        payload = to_dict(obj)
        assert payload[SCHEMA_KEY] == SCHEMA_VERSION
        assert payload["type"]

    def test_state_round_trip_preserves_snapshot(self):
        state = _state("joy", "anger")
        back = from_dict(to_dict(state))
        assert list(back.snapshot()) == pytest.approx(list(state.snapshot()))


class TestNeutrality:
    """Neutrality is not in the EMOTIONS registry, so it needs its own loader.

    Without one it round-trips to ``None`` — a silent hole rather than an error.
    """

    def test_round_trips_as_neutrality(self):
        from emotion_algebra.plutchik import Neutrality

        assert isinstance(from_dict(to_dict(Neutrality())), Neutrality)

    def test_round_trip_preserves_its_axis(self):
        from emotion_algebra.plutchik import Neutrality

        original = Neutrality(dimension=get_emotion("joy").dimension)
        back = from_dict(to_dict(original))
        assert back.dimension.name == "pleasantness"

    def test_json_round_trip(self):
        from emotion_algebra.plutchik import Neutrality

        assert isinstance(from_json(to_json(Neutrality())), Neutrality)


class TestSchemaGuard:
    def test_unknown_schema_version_rejected(self):
        payload = to_dict(get_emotion("joy"))
        payload[SCHEMA_KEY] = SCHEMA_VERSION + 1
        with pytest.raises(UnsupportedSchemaError):
            from_dict(payload)

    def test_missing_type_rejected(self):
        with pytest.raises((KeyError, ValueError)):
            from_dict({SCHEMA_KEY: SCHEMA_VERSION})

    def test_unknown_type_rejected(self):
        with pytest.raises((KeyError, ValueError)):
            from_dict({SCHEMA_KEY: SCHEMA_VERSION, "type": "not_an_emotion"})

    def test_unsupported_schema_error_is_a_value_error(self):
        # Callers that only catch ValueError must still be protected.
        assert issubclass(UnsupportedSchemaError, ValueError)

    def test_unserializable_object_rejected(self):
        with pytest.raises(TypeError):
            to_dict(object())

    def test_unknown_emotion_name_rejected(self):
        # A corrupt payload must raise, not deserialize to None.
        with pytest.raises(ValueError):
            from_dict({SCHEMA_KEY: SCHEMA_VERSION, "type": "emotion", "name": "bogus"})


class TestStatefulTypes:
    def test_state_round_trip_preserves_mood(self):
        # Mood is an EMA over the applied history, so it is real state — losing
        # it across a save/load would silently reset the agent's temperament.
        state = _state("joy", "joy", "joy", "joy", "joy")
        back = from_dict(to_dict(state))
        assert list(back.mood) == pytest.approx(list(state.mood))

    def test_timeline_round_trip_preserves_length_and_label(self):
        timeline = _timeline()
        back = from_dict(to_dict(timeline))
        assert len(back) == len(timeline)
        assert back.label == timeline.label

    def test_timeline_round_trip_preserves_drift(self):
        timeline = _timeline()
        back = from_dict(to_dict(timeline))
        assert list(back.drift()) == pytest.approx(list(timeline.drift()))
