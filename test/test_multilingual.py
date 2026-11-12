"""The multilingual probe.

Runs **offline**. ``test/multilingual_fixtures.json`` holds real encoder outputs
for the sentences below, captured once, so the probe can be exercised without
downloading a 470 MB model on every run.

That split is also the honest one: the probe and the projection into the core are
what this library owns. The encoder is a third-party model, and testing it here
would be testing ``paraphrase-multilingual-MiniLM-L12-v2``, not us.

The load-bearing test is :class:`TestTheCrux`. Everything else in the library
follows from anger and fear separating on **potency** — the angry customer
escalates, the frightened one leaves without a word — and a language where that
fails is a language this library cannot read, however confident its numbers look.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from emotion_algebra.affect import AffectState
from emotion_algebra.lang import UnsupportedLanguageError
from emotion_algebra.multilingual import (
    N_EMBED,
    PROBE_PATH,
    _probe,
    affect_from_features,
    design_matrix,
)
from emotion_algebra.neural import PROBE_AXES

FIXTURES = json.loads(
    (Path(__file__).parent / "multilingual_fixtures.json").read_text()
)
CASES = FIXTURES["cases"]


def case(lang: str, index: int = 0):
    matches = [c for c in CASES if c["lang"] == lang]
    return matches[index]


def state_of(c) -> AffectState:
    embedding = np.array([c["embedding"]], dtype=float)
    return affect_from_features(embedding, [c["text"]], lang=c["lang"])[0]


class TestTheCrux:
    """Anger and fear, in each language. The reason the library exists.

    Both are unpleasant and both are highly aroused, so valence and arousal
    cannot tell them apart. Potency can — and if it cannot, in some language,
    then this library does not read that language and must say so.
    """

    @pytest.mark.parametrize("lang", ["en", "pt", "ar"])
    def test_the_complaint_and_the_goodbye_do_not_collapse_together(self, lang):
        angry = state_of(case(lang, 0))     # "your app lost my work. Fix it."
        afraid = state_of(case(lang, 1))    # "I'm scared I've broken something."

        assert angry.potency > afraid.potency, (
            f"in {lang}, the person who will escalate and the person who will "
            f"quietly leave are indistinguishable. That is the one thing this "
            f"library must never do."
        )

    @pytest.mark.parametrize("lang", ["en", "pt", "ar"])
    def test_both_are_unpleasant_which_is_why_valence_cannot_do_this(self, lang):
        # The whole argument in one assertion: valence agrees, potency does not.
        angry = state_of(case(lang, 0))
        afraid = state_of(case(lang, 1))

        assert angry.valence < 0.25 and afraid.valence < 0.25
        assert abs(angry.potency - afraid.potency) > abs(
            angry.valence - afraid.valence
        ), f"in {lang}, potency must carry more of the distinction than valence"


class TestProbe:
    def test_the_probe_has_the_shape_the_module_claims(self):
        probe = _probe()
        assert probe.shape == (N_EMBED + 8 + 1, len(PROBE_AXES))

    def test_typography_touches_arousal_and_nothing_else(self):
        # Embeddings carry valence; typography carries activation. Letting the
        # typographic rows reach valence is how "this is unacceptable!!!" ends
        # up scored as mildly positive.
        probe = _probe()
        typo = probe[N_EMBED:N_EMBED + 8]
        for i, axis in enumerate(PROBE_AXES):
            if axis == "arousal":
                assert np.abs(typo[:, i]).sum() > 0, "arousal must use them"
            else:
                assert np.abs(typo[:, i]).sum() == 0, f"{axis} must not"

    def test_it_records_that_it_was_fitted_on_english_only(self):
        # The provenance is not a comment: it ships inside the artefact.
        meta = json.loads(PROBE_PATH.read_text())
        assert "ENGLISH ONLY" in meta["note"]
        assert "ZERO-SHOT" in meta["note"]

    def test_states_are_in_range(self):
        for c in CASES:
            s = state_of(c)
            assert 0.0 <= s.positivity <= 1.0
            assert 0.0 <= s.negativity <= 1.0
            assert -1.0 <= s.potency <= 1.0
            assert 0.0 <= s.arousal <= 1.0
            assert 0.0 <= s.unpredictability <= 1.0


class TestTypographyCrossesTheScript:
    """The design matrix must read each language's own orthography."""

    def test_arabic_emphasis_raises_arousal(self):
        # "This is unacceptable!!! Why???" — with the ARABIC question mark. Under
        # an English profile the questioning cue would read 0.0 here.
        loud = case("ar", 2)
        calm = case("ar", 1)
        assert state_of(loud).arousal > state_of(calm).arousal

    def test_the_typographic_block_differs_by_language(self):
        c = case("ar", 2)
        embedding = np.array([c["embedding"]], dtype=float)

        as_arabic = design_matrix(embedding, [c["text"]], lang="ar")
        as_english = design_matrix(embedding, [c["text"]], lang="en")

        typo_ar = as_arabic[0, N_EMBED:N_EMBED + 8]
        typo_en = as_english[0, N_EMBED:N_EMBED + 8]
        assert not np.allclose(typo_ar, typo_en), (
            "reading Arabic with English typography must produce DIFFERENT "
            "features — if it did not, the bug this module fixes would be "
            "invisible"
        )

    def test_an_unregistered_language_raises(self):
        c = case("en", 0)
        with pytest.raises(UnsupportedLanguageError):
            affect_from_features(
                np.array([c["embedding"]]), [c["text"]], lang="de"
            )


class TestShapes:
    def test_wrong_embedding_width_raises(self):
        with pytest.raises(ValueError, match="384"):
            design_matrix(np.zeros((1, 12)), ["hi"], lang="en")

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            design_matrix(np.zeros((2, N_EMBED)), ["only one"], lang="en")


class TestTheTokenizerSettings:
    """The encoder sees each batch padded to its longest text and each text cut
    at ``MAX_TOKENS``, on tokenizers 0.x and 1.x.

    A word-level ``tokenizer.json`` with the encoder's ``<s> ... </s>`` template
    stands in for the real one, and a stub records what the ONNX session gets,
    so the test stays offline. The expected rows follow from the settings by
    hand: ``enable_padding()`` pads with id 0, and the truncation keeps
    ``MAX_TOKENS - 2`` words to leave room for ``<s>`` and ``</s>``. tokenizers
    0.23.3 gives the same rows.
    """

    WORDS = "one two three four five six seven eight nine ten".split()

    @pytest.fixture
    def feed(self, tmp_path, monkeypatch):
        from emotion_algebra import multilingual

        vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        vocab.update({word: 4 + i for i, word in enumerate(self.WORDS)})
        template = [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "</s>", "type_id": 0}}]
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps({
            "version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [], "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing", "single": template,
                "pair": template + [{"Sequence": {"id": "B", "type_id": 0}},
                                    {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                "special_tokens": {
                    "<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]},
                    "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}}},
            "decoder": None,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
        }))

        seen = {}

        class Input:
            def __init__(self, name):
                self.name = name

        class Session:
            def get_inputs(self):
                return [Input("input_ids"), Input("attention_mask")]

            def run(self, _, feed):
                seen.update(feed)
                return [np.ones(feed["input_ids"].shape + (N_EMBED,))]

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda repo, filename: str(path) if filename == "tokenizer.json" else filename,
        )
        monkeypatch.setattr("onnxruntime.InferenceSession", lambda *a, **k: Session())
        multilingual._encoder.cache_clear()
        yield lambda texts: (multilingual._encode_batch(texts), seen)[1]
        multilingual._encoder.cache_clear()

    def test_a_batch_is_padded_and_cut(self, feed):
        from emotion_algebra.multilingual import MAX_TOKENS

        long_text = " ".join(self.WORDS * 15)              # 150 words
        seen = feed(["one two", long_text])

        kept = [4 + i % 10 for i in range(MAX_TOKENS - 2)]
        assert seen["input_ids"].tolist() == [
            [0, 4, 5, 2] + [0] * (MAX_TOKENS - 4),
            [0, *kept, 2],
        ]
        assert seen["attention_mask"].tolist() == [
            [1] * 4 + [0] * (MAX_TOKENS - 4),
            [1] * MAX_TOKENS,
        ]


class _Rc3Tokenizer:
    """tokenizers 1.0.0rc3: settings are properties and the Tokenizer itself
    encodes; there is no compile()."""

    padding = truncation = None

    @classmethod
    def from_file(cls, path):
        return cls()


class _Rc4Tokenizer(_Rc3Tokenizer):
    """tokenizers 1.0.0rc4: only what compile() returns encodes, and it keeps
    the settings the Tokenizer had at that moment."""

    def compile(self):
        return ("compiled", self.padding, self.truncation)


class TestTheLoaderFollowsTheApi:
    """The release candidates of tokenizers 1.0 share the major version and
    differ in API, so a version check picks the wrong path on one of them."""

    @pytest.fixture
    def load(self, monkeypatch):
        import tokenizers
        from emotion_algebra import multilingual

        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            lambda repo, filename: filename)
        monkeypatch.setattr("onnxruntime.InferenceSession", lambda *a, **k: None)
        monkeypatch.setattr(tokenizers, "Padding", lambda: "pad", raising=False)
        monkeypatch.setattr(tokenizers, "Truncation",
                            lambda max_length: ("cut", max_length), raising=False)

        def load(stub, version):
            monkeypatch.setattr(tokenizers, "Tokenizer", stub)
            monkeypatch.setattr(tokenizers, "__version__", version)
            multilingual._encoder.cache_clear()
            return multilingual._encoder()[0]

        yield load
        multilingual._encoder.cache_clear()

    def test_a_tokenizer_without_compile_is_set_up_and_used(self, load):
        from emotion_algebra.multilingual import MAX_TOKENS

        tokenizer = load(_Rc3Tokenizer, "1.0.0-rc.3")
        assert isinstance(tokenizer, _Rc3Tokenizer)
        assert (tokenizer.padding, tokenizer.truncation) == ("pad", ("cut", MAX_TOKENS))

    def test_a_tokenizer_is_set_up_before_it_is_compiled(self, load):
        from emotion_algebra.multilingual import MAX_TOKENS

        assert load(_Rc4Tokenizer, "0.0.0") == ("compiled", "pad", ("cut", MAX_TOKENS))
