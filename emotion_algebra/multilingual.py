"""Text to affect, in more than one language. **Experimental.**

:mod:`emotion_algebra.neural` reads English, and refuses everything else, because
DeepMoji was trained on English tweets and the typographic cues around it are
English orthography. This module is the other half of that sentence.

Experimental means what it says. DeepMoji remains the English path, and English
remains the only language with real gold behind it. What is here is a probe fitted
on English and applied to other languages **zero-shot** — an approach that is
measured rather than assumed, and that **does not fully work yet**:

======  ===========  =====================================
lang    potency d    verdict
======  ===========  =====================================
pt      **+0.46**    transfers — potency is the separating axis
ar      +0.28        **partial** — unpredictability dominates
======  ===========  =====================================

Read that table before you rely on this. The Arabic reading is usable, but the
anger/fear-on-potency distinction that the rest of the library is built on is
**not established for Arabic**, and pretending otherwise would be the exact sin
this package exists to name.

The trick, and why it works
--------------------------
There is no Arabic emotion training data worth fitting on — see below, it is
worse than it sounds — so a probe cannot be fitted *in* Arabic. It does not have
to be.

``paraphrase-multilingual-MiniLM-L12-v2`` (Reimers & Gurevych 2020, *Making
Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*,
EMNLP; Apache-2.0) is trained so that **a sentence and its translation land in
the same place**. Measured on the pair this library was built around:

    "This is the third time your app has lost my work."
    "Já é a terceira vez que a vossa aplicação perdeu o meu trabalho."
    "هذه هي المرة الثالثة التي يفقد فيها تطبيقكم عملي."

    en~pt  0.87      en~ar  0.78      pt~ar  0.95      (cosine)

...against **0.15** for the angry-English/frightened-Portuguese mismatch. The
translations are neighbours; the emotions are not confusable.

So the probe is fitted on **English gold only** — the same GoEmotions and EmoBank
data behind the English probe — and applied to Portuguese and Arabic **zero-shot**.
Nothing about the fit knows those languages exist. Whether the mapping survives
the crossing is not assumed: it is *measured*, on XED, and reported per language
in ``docs/evidence.md`` including where it loses.

Why not fit in-language
-----------------------
Because the data does not exist, and the data that pretends to is worse than
nothing.

**There is no human-rated Arabic valence/arousal/dominance lexicon.** Not "we
could not obtain one" — there is not one. The Arabic entries in the widely-used
NRC lexicon family are **machine translations** of English sentiment scores: they
record what English speakers felt about English words, relabelled in Arabic. To
fit Arabic prototypes on them would be to launder an English judgement into an
Arabic-looking number, which is the precise failure this library exists to call
out. They are not used here.

The only openly available Portuguese affective norms are **Brazilian** (ANEW-Br),
not European.

That is why the 26 prototype coordinates stay **canonical and language-neutral**,
and only their *names* are translated. ``prototypes.cross_lingual_transfer`` is
graded ``CONTESTED`` for exactly this reason: GRID supports the four *dimensions*
replicating across cultures, not the *per-term positions*.

Install with ``pip install emotion-algebra[multilingual]``. The encoder (~470 MB
of ONNX) is downloaded on first use and cached.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import numpy as np

from emotion_algebra.affect import AffectState
from emotion_algebra.lang import CHANNELS, profile, strip_emphasis, typographic_features

#: The sentence encoder. Apache-2.0, 50+ languages, 118M parameters.
ENCODER_REPO = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_FILE = "onnx/model.onnx"

#: Width of its sentence embedding.
N_EMBED = 384

#: Longest input the encoder sees. Beyond this the sentence is truncated — these
#: are utterances, not documents.
MAX_TOKENS = 128

#: Sentences per forward pass. Bounds peak memory: the encoder allocates
#: ``batch x tokens x 384`` floats at once, so an unbatched call on a large
#: corpus is a memory spike waiting to happen.
BATCH_SIZE = 64

#: CPU threads the encoder may use. ONNX Runtime's default is *every core*, which
#: for a 118M-parameter model on batched CPU inference buys little and is openly
#: hostile inside somebody else's process — a library has no business seizing the
#: whole machine. Raise it with ``EMOTION_ALGEBRA_THREADS`` if you have cores to
#: spare and a corpus to get through.
ENCODER_THREADS = max(1, int(os.environ.get("EMOTION_ALGEBRA_THREADS", "4")))

#: Where the fitted probe lives.
PROBE_PATH = Path(__file__).parent / "multilingual_probe.json"

#: Languages this encoder has been *evaluated* on. Being able to tokenize a
#: language is not the same as having measured what the probe does to it, and
#: only the second one earns a place here.
EVALUATED = ("en", "pt", "ar")


@lru_cache(maxsize=1)
def _probe() -> np.ndarray:
    """The ``(384 + 8 + 1, 5)`` probe: embedding + typography + intercept -> 5 axes."""
    data = json.loads(PROBE_PATH.read_text())
    return np.array(data["weights"], dtype=float)


@lru_cache(maxsize=1)
def _encoder():
    """The ONNX encoder and its tokenizer, loaded once."""
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover - depends on the install extra
        raise ImportError(
            "multilingual support needs the extra: "
            "pip install emotion-algebra[multilingual]"
        ) from e

    tokenizer = Tokenizer.from_file(hf_hub_download(ENCODER_REPO, "tokenizer.json"))
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=MAX_TOKENS)

    options = ort.SessionOptions()
    options.intra_op_num_threads = ENCODER_THREADS
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        hf_hub_download(ENCODER_REPO, ENCODER_FILE),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session


def _encode_batch(texts: Sequence[str]) -> np.ndarray:
    tokenizer, session = _encoder()
    encoded = tokenizer.encode_batch([str(t) for t in texts])

    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    feed = {"input_ids": ids, "attention_mask": mask}
    if any(i.name == "token_type_ids" for i in session.get_inputs()):
        feed["token_type_ids"] = np.zeros_like(ids)

    hidden = session.run(None, feed)[0]                    # (n, seq, 384)

    weights = mask[..., None].astype(float)
    pooled = (hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)
    return pooled / np.clip(
        np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None
    )


def encode(texts: Sequence[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Embed *texts* into the shared multilingual space.

    Mean-pooled over the token axis and L2-normalised, which is how this model
    was distilled and therefore the only pooling under which its cross-lingual
    alignment holds.

    Batched, because the encoder allocates ``batch x tokens x 384`` floats per
    forward pass and a whole corpus at once is a memory spike.

    Returns
    -------
    ndarray
        ``(n, 384)``.
    """
    texts = [str(t) for t in texts]
    if not texts:
        raise ValueError("no texts given")

    chunks = [
        _encode_batch(texts[i:i + batch_size])
        for i in range(0, len(texts), batch_size)
    ]
    return np.vstack(chunks)


def design_matrix(embeddings: np.ndarray, texts: Sequence[str], lang: str) -> np.ndarray:
    """Assemble ``[embedding | typography | 1]`` — the probe's input.

    The typographic cues are read with *lang*'s own orthography, which is the
    entire reason :mod:`emotion_algebra.lang` exists: a ``?``-counting rule scores
    zero on every Arabic question ever written, and does so without complaining.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.shape[1] != N_EMBED:
        raise ValueError(
            f"expected (n, {N_EMBED}) embeddings, got {embeddings.shape}"
        )
    if len(texts) != len(embeddings):
        raise ValueError("embeddings and texts must be the same length")

    typo = np.array(
        [typographic_features(t, lang)[0] for t in texts], dtype=float
    )
    return np.hstack([embeddings, typo, np.ones((len(embeddings), 1))])


def affect_from_features(
    embeddings: np.ndarray, texts: Sequence[str], lang: str
) -> List[AffectState]:
    """Map encoder features onto the core.

    This is the part this library owns — the probe and the projection. It needs
    no model, no network and no download, which is why it, rather than
    :func:`affect_from_texts`, is what the tests exercise.
    """
    profile(lang)                       # refuses an unregistered language
    scores = design_matrix(embeddings, texts, lang) @ _probe()

    return [
        AffectState(
            positivity=float(np.clip(row[0], 0.0, 1.0)),
            negativity=float(np.clip(row[1], 0.0, 1.0)),
            potency=float(np.clip(row[2], -1.0, 1.0)),
            arousal=float(np.clip(row[3], 0.0, 1.0)),
            unpredictability=float(np.clip(row[4], 0.0, 1.0)),
        )
        for row in scores
    ]


def affect_from_texts(texts: Sequence[str], lang: str) -> List[AffectState]:
    """Map each string to an :class:`~emotion_algebra.affect.AffectState`.

    Batched — the encoder forward pass dominates the cost.

    The encoder is shown the text with **emphasis stripped**, and the typographic
    cues see it intact. Emphasis changes how *loud* a sentence is, not whether
    what it describes is good or bad, and an encoder left to read ``!!!`` as
    excitement will drag "this is unacceptable!!!" toward neutral valence. In
    Arabic that strip only happens if the regex knows about ``؟`` (U+061F) — which
    is the sort of thing that fails silently, and is why the profile owns it.
    """
    texts = list(texts)
    if not texts:
        raise ValueError("no texts given")
    if not all(isinstance(t, str) for t in texts):
        raise ValueError("every text must be a str")

    profile(lang)

    from emotion_algebra.homeostasis import SET_POINT

    filled = [t if t.strip() else "." for t in texts]
    encoder_input = [strip_emphasis(t, lang) or "." for t in filled]

    states = affect_from_features(encode(encoder_input), filled, lang)
    return [
        SET_POINT if not text.strip() else state
        for text, state in zip(texts, states)
    ]
