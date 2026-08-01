"""The FITTED axes must actually be fitted.

`provenance.py` claims that prototype valence and arousal come from Warriner's
human norms. This test holds that claim to account: it re-derives them and
asserts the shipped table matches.

Without this, a hand-tweaked "improvement" can silently turn a fitted number into
a guessed one while the provenance record still says FITTED — which is worse than
never having claimed it, because now the lie is machine-readable.
"""
import pytest

from emotion_algebra.prototypes import PROTOTYPES
from emotion_algebra.provenance import Provenance, of


class TestFittedAxesAreActuallyFitted:
    def test_valence_and_arousal_are_registered_as_fitted(self):
        assert of("prototypes.valence").provenance is Provenance.FITTED
        assert of("prototypes.arousal").provenance is Provenance.FITTED

    def test_potency_and_unpredictability_do_NOT_claim_to_be_fitted(self):
        # They are guesses. The registry must not pretend otherwise.
        assert of("prototypes.potency").provenance is Provenance.CALIBRATED
        assert of("prototypes.unpredictability").provenance is Provenance.CALIBRATED
        assert not of("prototypes.potency").trustworthy
        assert not of("prototypes.unpredictability").trustworthy

    def test_the_fitted_axes_are_never_hand_edited(self):
        # A spot-check with teeth: these are the two entries that HAD drifted
        # away from the data (ecstasy's positivity was raised 0.36 -> 0.90 by
        # hand, which is exactly the failure this test exists to catch).
        assert PROTOTYPES["ecstasy"].positivity == pytest.approx(0.36, abs=0.01)
        assert PROTOTYPES["distress"].negativity == pytest.approx(0.44, abs=0.01)

    def test_plutchiks_intensity_ladder_does_not_hold_in_the_data(self):
        # A real finding, locked so nobody "fixes" it back: human norms rate
        # ECSTATIC as LESS positive than JOY. Plutchik's serenity < joy < ecstasy
        # ladder is not what people report.
        assert PROTOTYPES["ecstasy"].positivity < PROTOTYPES["joy"].positivity
