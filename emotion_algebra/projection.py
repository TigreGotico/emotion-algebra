"""Total conversion — every model reaches every other, and declares what it loses.

The theories this library implements are not mutually consistent, and pretending
otherwise is how you get a model that says the average of rage and terror is
*calm*.  So they are not merged.  They are **registered as views on one core**,
and every conversion is routed through that core:

    Hourglass ---.                      .--- PAD
                  \                    /
    Plutchik ------>  AFFECT CORE  <---+--- circumplex
                  /   (GRID axes)      \
    Lövheim ----'                        `--- NeuroState

N models need 2N maps, not N².  Every pair is reachable, always — and every map
carries a :class:`Fidelity` saying how much survives the trip.

    >>> import emotion_algebra.views          # registers the graph
    >>> from emotion_algebra.projection import fidelity
    >>> str(fidelity("hourglass", "core"))
    'heuristic'
    >>> str(fidelity("pad", "core"))
    'lossy'

A conversion that loses information is fine.  A conversion that loses information
*silently* is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from emotion_algebra._compat import StrEnum
from emotion_algebra.affect import AffectState


class Fidelity(StrEnum):
    """How much of a state survives a conversion."""

    #: Bijective on its subspace. Round-trips to machine precision.
    EXACT = "exact"

    #: Information is provably discarded, and :func:`explain_loss` says what.
    LOSSY = "lossy"

    #: Calibrated rather than derived. The numbers are a judgement call, and the
    #: model being converted to may not be empirically supported at all — check
    #: its grade in :mod:`emotion_algebra.evidence`.
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class View:
    """A registered model, and how it maps to and from the core."""

    name: str
    to_core: Callable[[Any], AffectState]
    from_core: Callable[[AffectState], Any]
    fidelity: Fidelity
    loses: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.fidelity})"


_VIEWS: Dict[str, View] = {}

#: The hub itself. Converting the core to the core is the identity.
CORE = "core"


def register_view(
    name: str,
    to_core: Callable[[Any], AffectState],
    from_core: Callable[[AffectState], Any],
    fidelity: Fidelity,
    loses: str = "",
) -> View:
    """Register a model as a view on the core.

    Parameters
    ----------
    name:
        The model's name, e.g. ``"pad"``.
    to_core, from_core:
        The two halves of the projection.
    fidelity:
        How much survives. ``LOSSY`` and ``HEURISTIC`` **must** say what is lost.
    loses:
        Prose naming exactly what this view cannot represent.

    Raises
    ------
    ValueError
        If the name is taken, or a lossy view does not declare its loss.
    """
    if name in _VIEWS or name == CORE:
        raise ValueError(f"view already registered: {name!r}")
    fidelity = Fidelity(fidelity)
    if fidelity is not Fidelity.EXACT and not loses.strip():
        raise ValueError(
            f"view {name!r} is {fidelity} and must declare what it loses — "
            "a conversion may lose information, but never silently"
        )
    view = View(name=name, to_core=to_core, from_core=from_core,
                fidelity=fidelity, loses=loses.strip())
    _VIEWS[name] = view
    return view


def views() -> List[str]:
    """Every registered view, plus the core."""
    return [CORE] + sorted(_VIEWS)


def get_view(name: str) -> View:
    """Return a registered :class:`View`.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _VIEWS:
        raise KeyError(f"unknown view: {name!r}; known: {views()}")
    return _VIEWS[name]


def to_core(obj: Any, source: str) -> AffectState:
    """Lift *obj* from the *source* model into the core."""
    if source == CORE:
        if not isinstance(obj, AffectState):
            raise TypeError(f"expected AffectState for {CORE!r}, got {type(obj).__name__}")
        return obj
    return get_view(source).to_core(obj)


def from_core(state: AffectState, target: str) -> Any:
    """Project *state* from the core into the *target* model."""
    if target == CORE:
        return state
    return get_view(target).from_core(state)


def convert(obj: Any, source: str, target: str) -> Any:
    """Convert between **any** two registered models, routed through the core.

    Always possible. Often lossy — call :func:`fidelity` and :func:`explain_loss`
    to find out how lossy before you trust the result.
    """
    return from_core(to_core(obj, source), target)


def fidelity(source: str, target: str) -> Fidelity:
    """The fidelity of ``source -> target``: the *weaker* of the two legs.

    A chain is only as faithful as its worst step. Hourglass -> PAD is LOSSY even
    though PAD -> core is EXACT, because the Hourglass leg already threw
    information away.
    """
    order = {Fidelity.EXACT: 0, Fidelity.LOSSY: 1, Fidelity.HEURISTIC: 2}
    legs = [
        Fidelity.EXACT if name == CORE else get_view(name).fidelity
        for name in (source, target)
    ]
    return max(legs, key=lambda f: order[f])


def explain_loss(source: str, target: str) -> str:
    """Say, in prose, exactly what a ``source -> target`` conversion discards."""
    parts = []
    for name, direction in ((source, "reading from"), (target, "writing to")):
        if name == CORE:
            continue
        view = get_view(name)
        if view.fidelity is Fidelity.EXACT:
            continue
        parts.append(f"{direction} {name} [{view.fidelity}]: {view.loses}")
    if not parts:
        return f"{source} -> {target} is exact; nothing is lost."
    return "\n".join(parts)


def report() -> str:
    """Render the conversion table."""
    lines = ["emotion-algebra — conversion views", "=" * 34, ""]
    for name in sorted(_VIEWS):
        view = _VIEWS[name]
        lines.append(f"{name}  [{view.fidelity}]")
        if view.loses:
            lines.append(f"    loses: {view.loses}")
    lines.append("")
    lines.append(f"All {len(_VIEWS) + 1} models are mutually reachable via {CORE!r}.")
    return "\n".join(lines)
