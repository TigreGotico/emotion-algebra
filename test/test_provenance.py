"""Every number in the library must say where it came from.

This is the test that makes the provenance registry more than documentation: a
new magic constant turns CI red until someone says what is behind it.
"""
import inspect
import pkgutil
from importlib import import_module

import pytest

import emotion_algebra
from emotion_algebra import provenance
from emotion_algebra.provenance import Provenance

#: Names that are structural rather than tunable: array shapes, scale bounds,
#: axis counts, version numbers. They encode a fact about the format, not a
#: judgement about emotion.
STRUCTURAL = {
    "AXIS_MAX",          # the Hourglass's own [-3, 3] bound
    "N_EMOJI",           # DeepMoji's output width
    "N_TYPO",            # our feature count
    "SCHEMA_VERSION",
    "MAX_EMOTION_DISTANCE",
    "MOOD_ALPHA",        # legacy EmotionalState
    "NEUTRAL_RADIUS",
    "BASELINE",
    "CORNERS",
    "CORNER_ANCHORS",
    "CORNER_POINTS",
    "AXES",
    "CORE_AXES",
    "MODULATORS",
    "PROBE_AXES",
    "N_EMBED",           # the encoder's output width
    "MAX_TOKENS",        # truncation bound: these are utterances, not documents
    "BATCH_SIZE",        # memory bound on the forward pass
    "ENCODER_THREADS",   # CPU courtesy, not a judgement about emotion
    "CHANNELS",
    "EVALUATED",
    "BASELINE_TEMPERAMENT",
    "SET_POINT",         # registered as homeostasis.SET_POINT
    "ORIGIN",
    "NEUTRAL",
    "DEFAULT_PREFERENCE",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AMBIGUOUS_NAMES",
    "TEMPERATURE",       # registered as readout.TEMPERATURE
    "BASELINE_LEVEL",    # views.py re-exports neuro.BASELINE_LEVEL
}


class TestRegistry:
    def test_every_registered_constant_has_a_citation(self):
        for key, c in provenance.all_constants().items():
            assert c.cite.strip(), f"{key} has no citation"

    def test_untrustworthy_constants_explain_themselves(self):
        # A calibrated or assumed number must say what IS known, not just that
        # it was guessed.
        for c in provenance.guessed():
            assert c.note.strip(), f"{c.key} is {c.provenance} but says nothing"

    def test_no_assumed_constants_remain(self):
        # ASSUMED means "a bare number with no reason". Every one is a bug.
        # They must be justified, derived, or deleted.
        assumed = provenance.by_provenance(Provenance.ASSUMED)
        assert not assumed, (
            "unjustified constants: " + ", ".join(c.key for c in assumed)
        )

    def test_the_fitted_ones_name_the_script_that_fitted_them(self):
        for c in provenance.by_provenance(Provenance.FITTED):
            assert "FITTED" in c.cite or "fitted" in c.cite.lower()

    def test_report_renders(self):
        out = provenance.report()
        assert "where the numbers came from" in out
        assert "CALIBRATED" in out

    def test_counts_add_up(self):
        assert sum(provenance.counts().values()) == len(provenance.all_constants())

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            provenance.of("nope.not.a.constant")

    def test_a_constant_cannot_be_registered_twice(self):
        with pytest.raises(ValueError):
            provenance.register_constant(
                "readout.TEMPERATURE", 0.25, Provenance.CALIBRATED, cite="dup"
            )

    def test_a_constant_cannot_be_registered_without_a_citation(self):
        with pytest.raises(ValueError):
            provenance.register_constant(
                "test.blank", 1.0, Provenance.CALIBRATED, cite="   "
            )


class TestEveryMagicNumberIsRegistered:
    """The enforcement.

    Walk the package, find every module-level numeric constant, and require that
    it is either registered in `provenance.py` or listed as STRUCTURAL. A new
    magic number fails this test until someone says what is behind it.
    """

    @staticmethod
    def _module_constants():
        found = {}
        for info in pkgutil.iter_modules(emotion_algebra.__path__):
            if info.name.startswith("_") or info.name in ("version",):
                continue
            mod = import_module(f"emotion_algebra.{info.name}")
            for name, val in vars(mod).items():
                if name.startswith("_") or name.isupper() is False:
                    continue
                if inspect.ismodule(val) or inspect.isclass(val):
                    continue
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    found[f"{info.name}.{name}"] = val
                elif isinstance(val, (dict, tuple, list)):
                    # Only a container OF NUMBERS is a tunable table. A dict of
                    # strings or of Emotion objects is a registry, not a knob.
                    flat = val.values() if isinstance(val, dict) else val
                    nums = [
                        v for v in flat
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    ]
                    if nums:
                        found[f"{info.name}.{name}"] = val
        return found

    def test_no_unregistered_magic_numbers(self):
        registered = set(provenance.all_constants())
        problems = []
        for key, value in self._module_constants().items():
            name = key.split(".", 1)[1]
            if name in STRUCTURAL or key in registered:
                continue
            problems.append(f"{key} = {value!r}")
        assert not problems, (
            "these numbers are not registered in provenance.py and are not "
            "declared STRUCTURAL:\n  " + "\n  ".join(sorted(problems))
        )
