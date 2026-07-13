"""Name resolution across the two dyad lineages.

The library carries **two** independent traditions for naming combinations of
primary emotions, and they overlap:

``Feeling`` (:mod:`emotion_algebra.feelings`)
    Plutchik's *dyads*: two adjacent sectors of the Wheel, combined at their
    own intensities.  ``love = joy + trust``.

``CompositeEmotion`` (:mod:`emotion_algebra.composite_emotions`)
    Two-axis Hourglass compounds, built from the **tertiary** (±3) emotions.
    ``love = ecstasy + admiration``.

Both are legitimate, both are used in the literature, and nine names appear in
*both* with different component definitions — see :data:`AMBIGUOUS_NAMES`.
Neither lineage is "right"; they answer different questions.  What is not
acceptable is for a bare name lookup to resolve by import order.

:func:`resolve` is the sanctioned lookup: it always returns the same object for
the same ``(name, prefer)`` pair, and it makes the choice explicit at the call
site.

Examples
--------
>>> resolve("love").name                     # Plutchik dyad by default
'love'
>>> [e.name for e in resolve("love").emotions]
['joy', 'trust']
>>> [e.name for e in resolve("love", prefer="composite").components]
['ecstasy', 'admiration']
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase

#: Preference for :func:`resolve` when a name exists in both lineages.
PreferT = Literal["feeling", "composite"]

#: The default lineage for an unqualified name.
#:
#: Feelings win.  Plutchik's dyads are the older, more widely cited construct,
#: they are what a reader means by "love" without further context, and they are
#: what the library's own documentation and examples have always used.
DEFAULT_PREFERENCE: PreferT = "feeling"


def _collisions() -> frozenset:
    from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS_NAMES
    from emotion_algebra.feelings import FEELING_NAMES
    return frozenset(set(FEELING_NAMES) & set(COMPOSITE_EMOTIONS_NAMES))


#: Names defined by **both** lineages, with different components in each.
#:
#: Every one of these resolves to the Plutchik ``Feeling`` unless you pass
#: ``prefer="composite"``.  The two readings are genuinely different emotions,
#: not different spellings: ``love`` is ``joy + trust`` as a Feeling but
#: ``ecstasy + admiration`` as a CompositeEmotion — the same dyad at a much
#: higher intensity.
AMBIGUOUS_NAMES: frozenset = _collisions()


def is_ambiguous(name: str) -> bool:
    """``True`` when *name* is defined by both lineages — see :data:`AMBIGUOUS_NAMES`.

    Case- and whitespace-insensitive, matching :func:`resolve`.
    """
    return name.lower().strip() in AMBIGUOUS_NAMES


def resolve(
    name: str,
    prefer: PreferT = DEFAULT_PREFERENCE,
) -> Optional["EmotionBase"]:
    """Resolve *name* to exactly one emotional state, deterministically.

    Search order:

    1. The 24 primary/secondary/tertiary :class:`~emotion_algebra.plutchik.Emotion`
       names — these never collide with anything.
    2. The preferred dyad lineage (*prefer*).
    3. The other dyad lineage, as a fallback for names only one of them defines.

    Parameters
    ----------
    name:
        Any emotion, feeling, or composite name.
    prefer:
        Which lineage wins for a name in :data:`AMBIGUOUS_NAMES`.
        ``"feeling"`` (default) → Plutchik dyad; ``"composite"`` → Hourglass
        two-axis compound.

    Returns
    -------
    EmotionBase or None
        ``None`` if no lineage defines *name*.

    Raises
    ------
    ValueError
        If *prefer* is not ``"feeling"`` or ``"composite"``.

    Examples
    --------
    >>> resolve("anger").name
    'anger'
    >>> resolve("optimism", prefer="composite").components[0].name
    'ecstasy'
    >>> resolve("not_an_emotion") is None
    True
    """
    if prefer not in ("feeling", "composite"):
        raise ValueError(
            f"prefer must be 'feeling' or 'composite', got {prefer!r}"
        )

    from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS
    from emotion_algebra.emotions import get_emotion
    from emotion_algebra.feelings import get_feeling

    emotion = get_emotion(name)
    if emotion is not None:
        return emotion

    lineages = (
        (get_feeling(name), COMPOSITE_EMOTIONS.get(name))
        if prefer == "feeling"
        else (COMPOSITE_EMOTIONS.get(name), get_feeling(name))
    )
    for candidate in lineages:
        if candidate is not None:
            return candidate
    return None


def describe_collision(name: str) -> Optional[dict]:
    """Return both readings of an ambiguous *name*, for documentation and debugging.

    Parameters
    ----------
    name:
        A name in :data:`AMBIGUOUS_NAMES`.

    Returns
    -------
    dict or None
        ``{"name", "feeling", "composite"}`` where the two values are the
        component-name lists of each lineage.  ``None`` when *name* is not
        ambiguous.

    Examples
    --------
    >>> describe_collision("love") == {
    ...     "name": "love",
    ...     "feeling": ["joy", "trust"],
    ...     "composite": ["ecstasy", "admiration"],
    ... }
    True
    """
    name = name.lower().strip()
    if not is_ambiguous(name):
        return None
    from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS_NAMES
    from emotion_algebra.feelings import FEELING_NAMES
    return {
        "name": name,
        "feeling": list(FEELING_NAMES[name]),
        "composite": list(COMPOSITE_EMOTIONS_NAMES[name]),
    }


def collision_table() -> list:
    """Every ambiguous name with both readings, sorted — the data behind ``docs/taxonomy.md``.

    Returns
    -------
    list of dict
        One :func:`describe_collision` entry per name in :data:`AMBIGUOUS_NAMES`.
    """
    return [describe_collision(n) for n in sorted(AMBIGUOUS_NAMES)]
