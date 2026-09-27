"""Number and text formatting shared by fleet briefings and reports."""
from __future__ import annotations

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


def pct(v) -> str:
    return "n/a" if v is None else f"{v:.0f}%" if abs(v) >= 10 or v == 0 else f"{v:.1f}%"


def months(n: int) -> str:
    return f"{n} month" + ("" if n == 1 else "s")


def signed(v, fmt=num) -> str:
    return ("+" if v >= 0 else "-") + fmt(abs(v))
