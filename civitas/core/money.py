"""The treasury reserve a city keeps back, shared by the fleet's money tools (fleet/tools.py re-exports these) and the
region's deal check (core/region.py: a deal is affordable only within the money above it). agents/scripted.py keeps its
own copy on the observation's numbers."""
from __future__ import annotations

RESERVE_MIN = 3000           # the treasury reserve (mirrors the fleet configs' mayor playbook): this, or 3 months of
                             # expenses (6 in a deficit)


def reserve_months(w) -> tuple[int, int, int]:
    """(months of expenses the reserve holds, last month's income, its expenses), in rounded dollars: 3, or 6 after a
    deficit month; (3, 0, 0) before the first month."""
    if not w.history:
        return 3, 0, 0
    last = w.history[-1]
    income, expenses = round(last["income"]), round(last["expenses"])
    return (3 if income >= expenses else 6), income, expenses


def reserve(w) -> float:
    """The treasury to keep back: RESERVE_MIN or last month's expenses x 3 (x 6 after a deficit month), in rounded
    dollars, as fleet.tools.reserve_why shows it (agents/scripted.py keeps its own). Smoke v16 seed 1, month 6: a $0.70
    deficit doubled the reserve."""
    mult, _, expenses = reserve_months(w)
    return max(RESERVE_MIN, mult * expenses)
