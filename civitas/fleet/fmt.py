"""Number and text formatting shared by fleet briefings and reports, and the briefing style they render in."""
from __future__ import annotations

import contextlib
import contextvars
import math

# The briefing style ([fleet] briefing, FleetAgentConfig.briefing): "guided" (the default) names the build, the call
# and the money to ask for; "open" states the same facts (sites, prices, coverage, money) and never an exact call, a
# "This turn" / Next instruction or a to-fund prescription, so the model plans. One switch for every renderer:
# briefing() and ToolBox.call set it for what they render (briefing_style); the renderers ask is_open() or pick a
# text with advice(). Guided is the code path as it was, byte for byte.
STYLES = ("guided", "open")
_STYLE: contextvars.ContextVar[str] = contextvars.ContextVar("fleet_briefing_style", default="guided")


@contextlib.contextmanager
def briefing_style(style: str | None):
    """Render in `style` ("guided" / "open"; None keeps the current one) inside the block."""
    if style is None:
        yield
        return
    if style not in STYLES:
        raise ValueError(f"briefing style must be one of {', '.join(STYLES)}, got {style!r}")
    token = _STYLE.set(style)
    try:
        yield
    finally:
        _STYLE.reset(token)


def is_open() -> bool:
    """The open briefing style is in force (briefing_style)."""
    return _STYLE.get() == "open"


def advice(guided: str, fact: str = "") -> str:
    """`guided` in the guided style; in the open one `fact` (the same facts without the instruction; "" drops it)."""
    return fact if is_open() else guided

LINE_CHARS = 120
TOOL_MESSAGE_CHARS = 4000      # the agent cuts each tool result message to this
REPORT_PREVIEW_CHARS = 200     # end_turn reports quoted in briefings and the fleet report
MESSAGE_PREVIEW_CHARS = 200    # council messages quoted in briefings


def cut(text, n: int = LINE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def num(v) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 10000:
        return f"{v / 1000:.1f}".removesuffix(".0") + "k"
    return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:,.2f}"


def money(v) -> str:
    return ("-" if v < 0 else "") + "$" + num(round(abs(v)))


def cash(v) -> str:
    """Money an agent may spend all of (the treasury), rounded down so that asking for the figure shown never fails:
    $1,937.60 is $1,937 and $12,380 is $12.3k (money() rounds: the mayor read $1,938 and its "allocate 1938" failed).
    Below zero it is money()."""
    if v < 0:
        return money(v)
    d = math.floor(v + 1e-9)
    return "$" + (f"{math.floor(d / 100) / 10:.1f}".removesuffix(".0") + "k" if d >= 10000 else f"{d:,}")


def pct(v) -> str:
    return "n/a" if v is None else f"{v:.0f}%" if abs(v) >= 10 or v == 0 else f"{v:.1f}%"


def months(n: int) -> str:
    return f"{n} month" + ("" if n == 1 else "s")


def signed(v, fmt=num) -> str:
    return ("+" if v >= 0 else "-") + fmt(abs(v))


def a_or_an(name: str) -> str:
    return ("an " if name[:1] in ("a", "e", "i", "o") else "a ") + name     # not u: a university
