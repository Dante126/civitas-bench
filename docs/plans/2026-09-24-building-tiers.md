# Building Tiers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cities: Skylines-style building tiers (water pump → pumping station → water works, police station → police HQ, …)
so a growing city has to move to bigger, more efficient infrastructure instead of spamming tier-1 buildings.

**Architecture:** `catalog.TIER_LINES` holds one ladder per building family; tier 1 is the existing `CATALOG` /
`SERVICES` dict. A placed facility stores its tier in `world.facility_state[(x, y)]["tier"]` and every system reads its
active spec through `world.facility_spec(x, y)`. Pressure comes from efficiency (bigger tiers give more per $), crowding
(tier-1 production buildings clustered together lose output) and a downtown rule (level 4–5 tiles get only 60 % of
tier-1 police / fire / health cover). Tiers are built fresh (`"tier": 2`) or reached by upgrading in place
(`upgrade_facility {"module": "tier"}`).

**Tech Stack:** Python 3.12, numpy, pytest; vanilla JS viewer (three.js + canvas pixel renderer).

**Design doc:** `docs/plans/2026-09-24-building-tiers-design.md` (read it first).

**Not a git repo:** where a plan normally says "commit", run the quick suite instead (`pytest -q -m "not slow"`) as a
checkpoint. All commands run from the repo root with the venv active (`source .venv/bin/activate`).

---

### Task 0: Record the before-numbers

**Step 1:** Run the benchmark and save the table.

```bash
civitas bench -c configs/baseline.toml --seeds 0-4 --workers 5 | tee /tmp/claude-tiers-before.txt
```

Expected: city ≈ 77.3 ± 2.2, population ≈ 14.0k (the reference numbers in `CLAUDE.md`). Keep the file; Task 9
compares against it.

**Step 2:** Run the full suite to confirm a green start: `pytest -q` → all pass.

---

### Task 1: Catalog — structs, tier lines, crowding constants, overrides

**Files:**
- Modify: `civitas/sim/catalog.py`
- Create: `tests/test_tiers.py`

**Step 1: Write the failing tests** (`tests/test_tiers.py`)

```python
"""Building tiers: ladders, crowding, downtown cover, upgrades in place."""
import numpy as np
import pytest

from civitas.core.actions import apply_action
from civitas.core.env import CivitasEnv, GameConfig
from civitas.sim import catalog as C
from civitas.sim.grid import distance_to
from civitas.sim.world import World


def test_lines_start_with_the_existing_specs_and_are_tagged():
    assert C.TIER_LINES["pump"][0] is C.CATALOG["pump"]
    assert C.TIER_LINES["clinic"][1] is C.SERVICES["hospital"]
    assert C.TIER_LINES["outlet"][1] is C.CATALOG["treatment"]
    assert C.TIER_BY_NAME["treatment"] == ("outlet", 2)
    assert C.TIER_BY_NAME["pumping_station"] == ("pump", 2)
    assert C.BASE_TIER[C.TREATMENT] == 2 and C.BASE_TIER[C.HOSPITAL] == 2 and C.BASE_TIER[C.PUMP] == 1
    for line, tiers in C.TIER_LINES.items():
        for n, t in enumerate(tiers, 1):
            assert t["tier"] == n and t["line"] == line and t["struct"] in C.CHAR
            assert t["dept"] in ("utilities", "police", "fire", "health", "education")
            if n > 1:
                assert t["unlock"] in C.UNLOCK_POP


@pytest.mark.parametrize("line,key", [("pump", "water"), ("tower", "water"), ("coal", "power"), ("gas", "power"),
                                      ("wind", "power"), ("solar", "power"), ("landfill", "collect"),
                                      ("police_station", "capacity"), ("fire_station", "responses"),
                                      ("elementary_school", "seats"), ("high_school", "seats"), ("clinic", "capacity")])
def test_higher_tiers_give_more_per_dollar_of_upkeep(line, key):
    tiers = C.TIER_LINES[line]
    per = [t[key] / t["upkeep"] for t in tiers]
    assert all(b > a for a, b in zip(per, per[1:])), per


def test_footprint_depends_on_tier():
    assert C.footprint(C.PUMP) == (1, 1) and C.footprint(C.PUMP, 3) == (2, 2)
    assert C.footprint(C.HOSPITAL) == (2, 2) and C.footprint(C.CASINO) == (2, 2)


def test_tier_and_old_overrides():
    old2, old1 = C.TIER_LINES["pump"][1]["water"], C.CATALOG["pump"]["water"]
    try:
        C.apply_overrides({"TIERS.pump.2.water": 999, "CATALOG.pump.water": 300})
        assert C.TIER_LINES["pump"][1]["water"] == 999 and C.TIER_LINES["pump"][0]["water"] == 300
    finally:
        C.TIER_LINES["pump"][1]["water"], C.CATALOG["pump"]["water"] = old2, old1


def test_upgrade_cost_and_gain():
    t1, t2 = C.TIER_LINES["pump"][:2]
    assert C.upgrade_cost(t1, t2) == round((t2["cost"] - t1["cost"]) * C.TIER_UPGRADE_PREMIUM)
    assert C.tier_gain(t1, t2) == {"water": t2["water"] - t1["water"]}
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_tiers.py -q`
Expected: FAIL (`AttributeError: module 'civitas.sim.catalog' has no attribute 'TIER_LINES'`).

**Step 3: Implement in `civitas/sim/catalog.py`**

3a. New structs after `AIRPORT = 40`:

```python
NUCLEAR = 41
CREMATORIUM = 42
```

3b. `POWER_PLANTS` gets `"nuclear": NUCLEAR`. `STRUCT_NAME` gets `NUCLEAR: "nuclear_plant", CREMATORIUM: "crematorium"`.
`CHAR` gets `NUCLEAR: "u", CREMATORIUM: "z"` (lower case like the hospital's `h`; `rciom` stay reserved for zones).
`FOOTPRINT` gets `NUCLEAR: (2, 2)`.

3c. `CATALOG`: add nuclear and retune the treatment plant (today it is worse than an outlet: 220 sewage for $6,000):

```python
    "nuclear": {"cost": 40000, "upkeep": 400, "power": 2000, "pollution": 0.0, "needs_river": True, "water_use": 100},
    ...
    "treatment": {"cost": 5000, "upkeep": 50, "sewage": 700, "needs_river": True, "discharge": 0.2},
```

3d. `_UPKEEP_KEY` gets `NUCLEAR: "nuclear"`; `DEPT_OF_STRUCT`'s utilities tuple gets `NUCLEAR` and add `CREMATORIUM: "health"`.

3e. New section right after `LANDMARK_BY_STRUCT` / `TOURIST_SPEND` (it needs `CATALOG`, `SERVICES`, `UNLOCK_POP`):

```python
# --- building tiers ------------------------------------------------------------
# Cities: Skylines style ladders. Tier 1 of each line is the existing CATALOG / SERVICES dict (the same object, so
# "CATALOG.pump.water" overrides still apply); higher tiers unlock at milestones and give more output per $ of upkeep
# and per tile. A placed facility keeps its tier in facility_state; world.facility_spec() returns the active spec.
# Build a tier fresh with "tier": n, or move one up with upgrade_facility {"module": "tier"}, which costs the price
# difference x TIER_UPGRADE_PREMIUM and grows into free tiles next to the building.
TIER_UPGRADE_PREMIUM = 1.2
GAIN_KEYS = ("power", "water", "sewage", "capacity", "seats", "collect", "responses", "rate", "holding")


def _base(spec: dict, name: str, struct: int, dept: str, unlock: str | None = None, footprint=(1, 1)) -> dict:
    """Tag an existing spec as the first building of a line (or a tier that already exists, like the hospital)."""
    spec.update(name=name, struct=struct, dept=dept, unlock=unlock, footprint=footprint)
    return spec


def _up(prev: dict, name: str, unlock: str, struct: int | None = None, footprint=(1, 1), **changes) -> dict:
    """A higher tier: the previous tier's spec with changes."""
    return {**prev, "name": name, "struct": prev["struct"] if struct is None else struct, "unlock": unlock,
            "footprint": footprint, **changes}


_K, _S = CATALOG, SERVICES
_u = "utilities"
TIER_LINES = {
    # power
    "coal": [_base(_K["coal"], "coal", COAL, _u),
             _up(_K["coal"], "advanced_coal", "advanced_coal", cost=9000, upkeep=120, power=500, pollution=12.0)],
    "gas": [_base(_K["gas"], "gas", GAS, _u),
            _up(_K["gas"], "combined_cycle", "combined_cycle", cost=12000, upkeep=90, power=450, pollution=6.0)],
    "solar": [_base(_K["solar"], "solar", SOLAR, _u),
              _up(_K["solar"], "solar_plant", "solar_plant", footprint=(2, 2), cost=16000, upkeep=50, power=400)],
    "wind": [_base(_K["wind"], "wind", WIND, _u),
             _up(_K["wind"], "advanced_turbine", "advanced_turbine", cost=5000, upkeep=20, power=100)],
    "hydro": [_base(_K["hydro"], "hydro", DAM, _u, unlock="hydro_dam")],
    "nuclear": [_base(_K["nuclear"], "nuclear", NUCLEAR, _u, unlock="nuclear", footprint=(2, 2))],
    # water
    "pump": [_base(_K["pump"], "pump", PUMP, _u),
             _up(_K["pump"], "pumping_station", "pumping_station", cost=4000, upkeep=45, water=650)],
    "tower": [_base(_K["tower"], "tower", TOWER, _u),
              _up(_K["tower"], "inland_water_plant", "inland_water_plant", cost=7000, upkeep=70, water=600)],
    # sewage
    "outlet": [_base(_K["outlet"], "outlet", OUTLET, _u),
               _base(_K["treatment"], "treatment", TREATMENT, _u, unlock="treatment_plant")],
    # garbage, deathcare, health
    "landfill": [_base(_S["landfill"], "landfill", LANDFILL, "health")],
    "incinerator": [_base(_S["incinerator"], "incinerator", INCINERATOR, "health", unlock="incinerator")],
    "cemetery": [_base(_S["cemetery"], "cemetery", CEMETERY, "health")],
    "clinic": [_base(_S["clinic"], "clinic", CLINIC, "health"),
               _base(_S["hospital"], "hospital", HOSPITAL, "health", unlock="hospital", footprint=(2, 2))],
    # safety
    "police_station": [_base(_S["police_station"], "police_station", POLICE, "police")],
    "jail": [_base(_S["jail"], "jail", JAIL, "police", unlock="jail")],
    "fire_station": [_base(_S["fire_station"], "fire_station", FIRE, "fire")],
    "shelter": [_base(_S["shelter"], "shelter", SHELTER, "fire")],
    # education
    "elementary_school": [_base(_S["elementary_school"], "elementary_school", SCHOOL, "education")],
    "high_school": [_base(_S["high_school"], "high_school", HIGHSCHOOL, "education", unlock="high_school")],
    "university": [_base(_S["university"], "university", UNIVERSITY, "education", unlock="university",
                         footprint=(2, 2))],
    "library": [_base(_S["library"], "library", LIBRARY, "education", unlock="library")],
}
_L = TIER_LINES
_L["pump"].append(_up(_L["pump"][1], "water_works", "water_works", footprint=(2, 2), cost=12000, upkeep=100,
                      water=2000, min_quality=0.85))
_L["outlet"].append(_up(_L["outlet"][1], "advanced_treatment", "advanced_treatment", footprint=(2, 2), cost=14000,
                        upkeep=110, sewage=2000, discharge=0.05))
_L["landfill"] += [_up(_L["landfill"][0], "recycling_centre", "recycling_centre", cost=6000, upkeep=50, collect=900,
                       fill=None, pollution=1.5)]
_L["landfill"] += [_up(_L["landfill"][1], "waste_complex", "waste_complex", footprint=(2, 2), cost=20000, upkeep=150,
                       collect=3000, power=60, pollution=5.0)]
_L["cemetery"].append(_up(_L["cemetery"][0], "crematorium", "crematorium", struct=CREMATORIUM, cost=5000, upkeep=35,
                          rate=120, plots=None))
_L["clinic"].append(_up(_L["clinic"][1], "medical_centre", "medical_centre", footprint=(2, 2), cost=20000,
                        upkeep=220, capacity=22000, radius=20))
_L["police_station"].append(_up(_L["police_station"][0], "police_hq", "police_hq", cost=9000, upkeep=130,
                                capacity=14000, radius=22, holding=100))
_L["jail"].append(_up(_L["jail"][0], "prison", "prison", footprint=(2, 2), cost=14000, upkeep=120, capacity=1200))
_L["fire_station"].append(_up(_L["fire_station"][0], "fire_hq", "fire_hq", cost=9000, upkeep=130, capacity=900,
                              responses=15, radius=20))
_L["elementary_school"].append(_up(_L["elementary_school"][0], "large_elementary", "large_elementary", cost=6000,
                                   upkeep=80, seats=1200))
_L["high_school"].append(_up(_L["high_school"][0], "large_high_school", "large_high_school", cost=11000, upkeep=130,
                             seats=2000, radius=26))

UNLOCK_POP.update({
    "pumping_station": 1000, "advanced_turbine": 1000, "recycling_centre": 1000, "crematorium": 1000,
    "large_elementary": 1000,
    "inland_water_plant": 5000, "advanced_coal": 5000, "combined_cycle": 5000, "police_hq": 5000, "fire_hq": 5000,
    "large_high_school": 5000,
    "water_works": 10000, "advanced_treatment": 10000, "solar_plant": 10000, "nuclear": 10000, "waste_complex": 10000,
    "medical_centre": 10000, "prison": 10000,
})

TIER_BY_NAME: dict[str, tuple[str, int]] = {}   # tier name -> (line, tier); a line's name is its tier 1
LINE_OF_STRUCT: dict[int, str] = {}             # struct -> line
BASE_TIER: dict[int, int] = {}                  # struct -> lowest tier using it (facilities without a stored tier)
for _line, _tiers in TIER_LINES.items():
    for _n, _t in enumerate(_tiers, 1):
        _t["line"], _t["tier"] = _line, _n
        TIER_BY_NAME[_t["name"]] = (_line, _n)
        LINE_OF_STRUCT.setdefault(_t["struct"], _line)
        BASE_TIER.setdefault(_t["struct"], _n)
TIERED_STRUCTS = frozenset(LINE_OF_STRUCT)
STRUCTS_OF_DEPT: dict[str, list[int]] = {}
for _st, _line in LINE_OF_STRUCT.items():
    STRUCTS_OF_DEPT.setdefault(TIER_LINES[_line][0]["dept"], []).append(_st)
PRODUCER_STRUCTS = tuple(sorted(UTILITY_TYPES | {INCINERATOR, LANDFILL}))   # may make power / water / sewage
POLLUTER_STRUCTS = (COAL, GAS, LANDFILL, INCINERATOR)

# crowding: production buildings of one group within `radius` tiles of each other lose output
CROWD_STEP, CROWD_FLOOR = 0.15, 0.4          # -15% per neighbour, at least 40% output
CROWD_GROUPS = {
    "river intake": {"lines": ("pump",), "radius": 4, "why": "pumps share this stretch of river"},
    "river outfall": {"lines": ("outlet",), "radius": 4, "why": "outfalls overload this stretch of river"},
    "aquifer": {"lines": ("tower",), "radius": 5, "why": "wells draw down the same groundwater"},
    "wind wake": {"lines": ("wind",), "radius": 2, "why": "turbines steal each other's wind"},
}
# downtown: level 4-5 buildings get only this share of tier-1 police / fire / health cover
DOWNTOWN_LEVEL = 4
T1_DOWNTOWN = 0.6


def footprint(struct: int, tier: int | None = None) -> tuple[int, int]:
    line = LINE_OF_STRUCT.get(struct)
    if line is None:
        return FOOTPRINT.get(struct, (1, 1))
    return tuple(TIER_LINES[line][(tier or BASE_TIER[struct]) - 1]["footprint"])


def upgrade_cost(cur: dict, nxt: dict) -> int:
    return round(max(0, nxt["cost"] - cur["cost"]) * TIER_UPGRADE_PREMIUM)


def tier_gain(cur: dict, nxt: dict) -> dict:
    """What moving from one tier to the next adds, e.g. {'water': 400}."""
    return {k: nxt[k] - (cur.get(k) or 0) for k in GAIN_KEYS if nxt.get(k) and nxt[k] > (cur.get(k) or 0)}
```

Note: `PRODUCER_STRUCTS` needs `UTILITY_TYPES`, which is defined near the top from `POWER_PLANTS` etc., so nuclear is
already in it.

3f. `apply_overrides`: before the `elif parts[0] in g` branch add

```python
        elif parts[0] == "TIERS" and len(parts) == 4:
            TIER_LINES[parts[1]][int(parts[2]) - 1][parts[3]] = value
```

and extend the docstring example with `"TIERS.pump.2.water": 800`.

**Step 4: Run tests**

Run: `pytest tests/test_tiers.py -q` → PASS. Then `pytest -q -m "not slow"`: expect failures only where the
treatment plant numbers changed (fix those test expectations if any assert exact values) — nothing else should move yet.

---

### Task 2: World — facility specs, tier-aware footprints, upgrades, frame helpers

**Files:**
- Modify: `civitas/sim/world.py` (multi-tile section, ~lines 192–241)
- Modify: `civitas/sim/events.py:123-125` (flood marks the anchor, not every tile)
- Test: `tests/test_tiers.py`

**Step 1: Failing tests** (append to `tests/test_tiers.py`)

```python
def _bank_world(w=None):
    """Straight river at x=31-33; the east bank x=34 is riverbank.

    Roads: the main street (row y) crosses the river to x=36; a road along x=35 north of the street (1x1 bank sites at
    x=34, rows 2..y-1); a road along x=36 south of the street (free 2x2 bank blocks at x=34-35); a stub on row y+8
    from x=34 to 36 (a bank tile at (34, y+7) that touches it).
    """
    w = w or World(seed=0)
    w.terrain[:] = C.LAND
    w.terrain[:, 31:34] = C.WATER
    w.dist_water = distance_to(w.terrain == C.WATER)
    y = w.highway[1]
    w.struct[y, 1:37] = C.ROAD
    w.struct[2:y, 35] = C.ROAD
    w.struct[y + 1:46, 36] = C.ROAD
    w.struct[y + 8, 34:37] = C.ROAD
    w.refresh()
    return w, y


def test_facility_spec_defaults_and_stored_tiers():
    w, y = _bank_world()
    w.struct[5, 34] = C.TREATMENT                 # placed by hand, no facility_state
    assert w.facility_tier(34, 5) == 2 and w.facility_spec(34, 5)["name"] == "treatment"
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    assert w.facility_tier(35, y + 3) == 3        # any tile of the building
    assert w.facility_spec(35, y + 3)["name"] == "water_works"
    assert sorted(w.parts(35, y + 3)) == [(34, y + 2), (34, y + 3), (35, y + 2), (35, y + 3)]
    assert w.facility_spec(0, 0) is None


def test_upgrade_building_moves_up_the_line_and_keeps_state():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, 5)
    w.facility_state[(34, 5)].update(built=-30, modules=["expansion"])
    nxt = w.upgrade_building(34, 5)
    assert nxt["name"] == "pumping_station"
    assert w.facility_state[(34, 5)] == {"built": -30, "modules": ["expansion"], "tier": 2}
    with pytest.raises(ValueError, match="2x2"):
        w.upgrade_building(34, 5)                 # the x=35 road and the river leave no room


def test_upgrade_grows_into_free_tiles_and_can_move_the_anchor():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, y + 7, tier=2)
    w.upgrade_building(34, y + 7)                 # row y+8 is road, so it grows north
    assert (34, y + 6) in w.facility_state and (34, y + 7) not in w.facility_state
    assert w.facility_tier(34, y + 6) == 3
    assert sorted(w.parts(35, y + 7)) == [(34, y + 6), (34, y + 7), (35, y + 6), (35, y + 7)]
    with pytest.raises(ValueError, match="top tier"):
        w.upgrade_building(34, y + 6)


def test_upgrade_can_change_the_building_type():
    w, y = _bank_world()
    w.place_building(C.OUTLET, 34, 8)
    w.upgrade_building(34, 8)
    assert w.struct[8, 34] == C.TREATMENT and w.facility_tier(34, 8) == 2


def test_remove_and_multi_buildings():
    w, y = _bank_world()
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    i = (y + 2) * w.w + 34
    assert [i, 2, 2] in w.multi_buildings()
    lv = w.frame_levels()
    assert lv[y + 2, 34] == lv[y + 3, 35] == 3 and w.level[y + 2, 34] == 0
    w.remove_building(35, y + 3)
    assert (w.struct[y + 2:y + 4, 34:36] == C.NONE).all() and (34, y + 2) not in w.facility_state
```

**Step 2:** `pytest tests/test_tiers.py -q` → FAIL (`place_building() got an unexpected keyword argument 'tier'`).

**Step 3: Implement in `civitas/sim/world.py`**

Replace `parts`, `footprint_tiles`, `place_building` and add the new helpers (keep `building_network`,
`remove_building`, `n_buildings` as they are):

```python
    # ----------------------------------------------- multi-tile buildings & tiers
    def _anchor_of(self, x: int, y: int) -> tuple[int, int]:
        a = int(self.anchor[y, x])
        return (x, y) if a < 0 else (a % self.w, a // self.w)

    def facility_tier(self, x: int, y: int) -> int:
        ax, ay = self._anchor_of(x, y)
        return self.facility_state.get((ax, ay), {}).get("tier") or C.BASE_TIER.get(int(self.struct[ay, ax]), 1)

    def facility_spec(self, x: int, y: int) -> dict | None:
        """Active tier spec of the facility covering (x, y) (see catalog.TIER_LINES), or None."""
        ax, ay = self._anchor_of(x, y)
        line = C.LINE_OF_STRUCT.get(int(self.struct[ay, ax]))
        return C.TIER_LINES[line][self.facility_tier(ax, ay) - 1] if line else None

    def footprint_at(self, x: int, y: int) -> tuple[int, int]:
        ax, ay = self._anchor_of(x, y)
        spec = self.facility_spec(ax, ay)
        return tuple(spec["footprint"]) if spec else C.FOOTPRINT.get(int(self.struct[ay, ax]), (1, 1))

    def parts(self, x: int, y: int) -> list[tuple[int, int]]:
        """All tiles of the building at (x, y) (just (x, y) for one-tile buildings)."""
        a = int(self.anchor[y, x])
        if a < 0:
            return [(x, y)]
        ax, ay = a % self.w, a // self.w
        fw, fh = self.footprint_at(ax, ay)
        return [(ax + i, ay + j) for j in range(fh) for i in range(fw)]

    def footprint_tiles(self, struct: int, x: int, y: int, terrain: int = C.LAND,
                        tier: int | None = None) -> list[tuple[int, int]]:
        """Tiles a new building would cover; raises ValueError if they are not all free."""
        fw, fh = C.footprint(struct, tier)
        ...  # body unchanged from here on
```

```python
    def place_building(self, struct: int, x: int, y: int, terrain: int = C.LAND,
                       tier: int | None = None) -> list[tuple[int, int]]:
        tiles = self.footprint_tiles(struct, x, y, terrain, tier)
        for tx, ty in tiles:
            self.struct[ty, tx] = struct
            self.level[ty, tx] = 0
            self.anchor[ty, tx] = y * self.w + x if len(tiles) > 1 else -1
        self.facility_state[(x, y)] = {"built": self.month}
        if struct in C.TIERED_STRUCTS:
            self.facility_state[(x, y)]["tier"] = tier or C.BASE_TIER[struct]
        return tiles

    def next_tier(self, x: int, y: int) -> tuple[dict, dict, tuple[int, int], list[tuple[int, int]]]:
        """(current spec, next spec, new top-left, new tiles) for moving the facility at (x, y) one tier up.

        A bigger tier grows into any free block that contains the old footprint (top-left first). Raises ValueError
        if the facility has no higher tier or there is no room."""
        ax, ay = self._anchor_of(x, y)
        cur = self.facility_spec(ax, ay)
        if cur is None:
            raise ValueError(f"no upgradable facility at ({x},{y})")
        tiers = C.TIER_LINES[cur["line"]]
        name = cur["name"].replace("_", " ")
        if cur["tier"] >= len(tiers):
            raise ValueError(f"the {name} at ({ax},{ay}) is already the top tier of its line")
        nxt = tiers[cur["tier"]]
        old = set(self.parts(ax, ay))
        (ow, oh), (fw, fh) = self.footprint_at(ax, ay), tuple(nxt["footprint"])
        terrain = self.terrain[ay, ax]
        for dy in range(0, oh - fh - 1, -1):
            for dx in range(0, ow - fw - 1, -1):
                x0, y0 = ax + dx, ay + dy
                tiles = [(x0 + i, y0 + j) for j in range(fh) for i in range(fw)]
                if all(self.in_bounds(tx, ty) and ((tx, ty) in old or (self.terrain[ty, tx] == terrain
                                                                       and self.struct[ty, tx] == C.NONE))
                       for tx, ty in tiles):
                    return cur, nxt, (x0, y0), tiles
        raise ValueError(f"upgrading the {name} at ({ax},{ay}) to a {nxt['name'].replace('_', ' ')} needs a free "
                         f"{fw}x{fh} block of {'river' if terrain == C.WATER else 'land'} tiles around it")

    def upgrade_building(self, x: int, y: int) -> dict:
        """Move the facility at (x, y) one tier up its line (keeps age, modules and stock). Returns the new spec."""
        cur, nxt, (x0, y0), tiles = self.next_tier(x, y)
        ax, ay = self._anchor_of(x, y)
        for tx, ty in self.parts(ax, ay):
            self.anchor[ty, tx] = -1
        state = self.facility_state.pop((ax, ay), None) or {"built": self.month}
        for tx, ty in tiles:
            self.struct[ty, tx] = nxt["struct"]
            self.level[ty, tx] = 0
            self.anchor[ty, tx] = y0 * self.w + x0 if len(tiles) > 1 else -1
        state["tier"] = nxt["tier"]
        self.facility_state[(x0, y0)] = state
        return nxt

    def multi_buildings(self) -> list[list[int]]:
        """[top-left flat index, width, height] of every building bigger than one tile (for frames)."""
        flat = np.arange(self.h * self.w).reshape(self.h, self.w)
        ys, xs = np.nonzero(self.anchor == flat)
        return [[int(y) * self.w + int(x), *self.footprint_at(int(x), int(y))] for y, x in zip(ys.tolist(), xs.tolist())]

    def frame_levels(self) -> np.ndarray:
        """Zone levels plus the tier of every tiered facility (the viewer draws bigger models for higher tiers)."""
        lv = self.level.copy()
        for st in C.TIERED_STRUCTS:
            for x, y in self.facilities(st):
                t = self.facility_tier(x, y)
                for px, py in self.parts(x, y):
                    lv[py, px] = t
        return lv
```

Note on the `range` bounds in `next_tier`: for a 1x1 → 2x2 upgrade, `oh - fh - 1 = -2`, so `dy` runs over `0, -1`.

**Step 4: `civitas/sim/events.py:123-125`** — mark the building, not each tile:

```python
        for y, x in zip(*np.nonzero(zone & np.isin(self.struct, list(C.UTILITY_TYPES)))):
            self._fstate(*self._anchor_of(int(x), int(y)))["offline_until"] = self.month + C.FLOOD_OFFLINE_MONTHS
            offline += 1
```

(`is_offline(x, y)` is always called with anchors after Task 3.)

**Step 5:** `pytest tests/test_tiers.py -q` → PASS; `pytest -q -m "not slow"` → all pass.

---

### Task 3: Production — per-building specs, crowding, nuclear cooling, water works, pollution, upkeep, metrics

**Files:**
- Modify: `civitas/sim/world.py` (`refresh()` producers block ~331–361 and sewage discharge ~404; `_compute_environment`
  ~446–452; `dept_upkeep`; `_metrics`)
- Test: `tests/test_tiers.py`

**Step 1: Failing tests**

```python
def _street(length=40):
    """A straight street east of the highway (row y). Row y+1 is free; on row y-1 the town hall sits somewhere in
    x=11..27, so tests only use x<=10 there."""
    w = World(seed=0)
    y = w.highway[1]
    w.struct[y, 1:length] = C.ROAD
    w.refresh()
    return w, y


def _out(w, x, y, key="water"):
    return next(p[key] for p in w.plant_list if (p["x"], p["y"]) == (x, y))


def test_crowded_pumps_lose_output():
    w, y = _street()
    full = C.CATALOG["pump"]["water"]
    w.place_building(C.PUMP, 5, y + 1)
    w.refresh()
    assert _out(w, 5, y + 1) == pytest.approx(full)
    for x in (7, 9):
        w.place_building(C.PUMP, x, y + 1)
    w.refresh()
    assert _out(w, 7, y + 1) == pytest.approx(full * (1 - 2 * C.CROWD_STEP))
    for x in (11, 13, 15):
        w.place_building(C.PUMP, x, y + 1)
    w.refresh()
    assert _out(w, 9, y + 1) == pytest.approx(full * C.CROWD_FLOOR)       # 4 neighbours within 4 tiles
    p = next(p for p in w.plant_list if (p["x"], p["y"]) == (9, y + 1))
    assert p["crowd"] == pytest.approx(C.CROWD_FLOOR) and p["neighbours"] == 4


def test_spread_out_pumps_and_one_big_plant_are_not_crowded():
    w, y = _street()
    w.place_building(C.PUMP, 5, y + 1)
    w.place_building(C.PUMP, 15, y + 1)
    w.place_building(C.PUMP, 25, y + 1, tier=3)
    w.refresh()
    assert _out(w, 5, y + 1) == pytest.approx(C.CATALOG["pump"]["water"])
    assert _out(w, 25, y + 1) == pytest.approx(C.TIER_LINES["pump"][2]["water"])
    assert all("crowd" not in p for p in w.plant_list)


def test_water_works_filter_dirty_river_water():
    w, y = _bank_world()
    w.river_row[:] = 1.0
    w.place_building(C.PUMP, 34, 5)
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    w.refresh()
    q = {(p["x"], p["y"]): p["quality"] for p in w.plant_list if "quality" in p}
    assert q[(34, 5)] == pytest.approx(0.2) and q[(34, y + 2)] == pytest.approx(0.85)


def test_nuclear_plant_needs_cooling_water():
    w, y = _bank_world()
    w.place_building(C.NUCLEAR, 34, y + 2)
    w.refresh()
    w.refresh()
    assert _out(w, 34, y + 2, "power") == 0.0             # no water, no power
    w.place_building(C.PUMP, 34, 5, tier=2)
    w.refresh()
    w.refresh()
    assert _out(w, 34, y + 2, "power") == pytest.approx(C.CATALOG["nuclear"]["power"])
    assert sum(n.water_demand for n in w.networks) >= C.CATALOG["nuclear"]["water_use"]


def test_upkeep_and_pollution_follow_the_tier():
    w, y = _street()
    base = w.dept_upkeep()["utilities"]
    w.place_building(C.COAL, 5, y - 1)
    t1, t2 = C.TIER_LINES["coal"]
    assert w.dept_upkeep()["utilities"] - base == pytest.approx(t1["upkeep"])
    w._compute_environment()
    dirty = w.plant_pollution[y - 1, 5]
    w.upgrade_building(5, y - 1)
    assert w.dept_upkeep()["utilities"] - base == pytest.approx(t2["upkeep"])
    w._compute_environment()
    assert w.plant_pollution[y - 1, 5] < dirty
    assert w.metrics()["tiers"]["coal"] == [0, 1]
```

**Step 2:** Run → FAIL (no crowding, `KeyError: 'tiers'` etc.).

**Step 3: Implement**

3a. In `refresh()`, after the consumer block (after `w_need *= ...`), add a cooling-water grid, and replace the whole
producers block (from `# producers` down to the `self.disconnected_plants = [...]` assignment) with:

```python
        # producers: one entry per building (multi-tile plants count once)
        plants_by_net: dict[int, list] = {i: [] for i in range(n)}
        u_eff = self.funding_eff("utilities")
        self.crowd = self._crowd_factors()
        cool = np.zeros(s.shape)                       # nuclear plants draw cooling water from the water network
        found = []
        for st in C.PRODUCER_STRUCTS:
            for x, y in self.facilities(st):
                spec = self.facility_spec(x, y)
                if any(spec.get(k) for k in ("power", "water", "sewage")) or spec.get("water_use"):
                    found.append((y, x, spec))
        found.sort(key=lambda t: (t[0], t[1]))       # row-major, like the old per-tile scan
        for y, x, spec in found:
            line = spec["line"]
            net = self.building_network(x, y)
            off = self.is_offline(x, y)
            cf, cn = self.crowd.get((x, y), (1.0, 0))
            f = 0.0 if off else u_eff * self.plant_output_factor(x, y, line) * cf
            if spec.get("water_use"):
                cool[y, x] = spec["water_use"]
                if not self.watered[y, x]:
                    f = 0.0
            info = {"kind": spec["name"], "tier": spec["tier"], "x": x, "y": y, "network": net,
                    "power": float(spec.get("power") or 0) * f, "water": float(spec.get("water") or 0) * f}
            if off:
                info["offline"] = True
            if cn:
                info["crowd"], info["neighbours"] = round(cf, 2), cn
            age = (self.month - self.facility_state.get((x, y), {}).get("built", self.month)) / 12.0
            if age >= 1:
                info["age_years"] = round(age, 1)
            info["sewage"] = float(spec.get("sewage") or 0) * f
            if line == "pump":
                q = max(0.2, 1.0 - 0.8 * max(self.pollution[y, x], self._river_near(x, y)))
                info["quality"] = round(float(max(q, spec.get("min_quality", 0.0))), 2)
            elif line == "tower":
                info["quality"] = 1.0
            if net >= 0:
                plants_by_net[net].append(info)
        self.plant_list = [p for lst in plants_by_net.values() for p in lst]
        self.disconnected_plants = [{"kind": spec["name"], "x": x, "y": y} for y, x, spec in found
                                    if self.building_network(x, y) < 0]
```

`_river_near(x, y)` for a 2x2 water works at its top-left tile: the anchor is on the bank for east-bank blocks. That's
fine for now; do not over-engineer.

3b. In the per-network distribution loop, water demand includes cooling but sewage does not. Change the tuple to:

```python
            for res_key, need, served_grid in (("power", p_need, powered), ("water", w_need + cool, watered),
                                               ("sewage", w_need, sewered)):
```

3c. Discharge per sewage plant (was `C.CATALOG[p["kind"]]["discharge"]`):

```python
                        factor = self.facility_spec(p["x"], p["y"])["discharge"]
```

3d. Add the crowding helper to `World` (next to `_main_net`):

```python
    def _crowd_factors(self) -> dict[tuple[int, int], tuple[float, int]]:
        """(output factor, neighbours) per production building in a crowd group (catalog.CROWD_GROUPS)."""
        out = {}
        for g in C.CROWD_GROUPS.values():
            structs = {t["struct"] for line in g["lines"] for t in C.TIER_LINES[line]}
            boxes = []
            for st in structs:
                for x, y in self.facilities(st):
                    fw, fh = self.footprint_at(x, y)
                    boxes.append((x, y, x + fw - 1, y + fh - 1))
            for i, (x0, y0, x1, y1) in enumerate(boxes):
                n = sum(1 for j, (a0, b0, a1, b1) in enumerate(boxes)
                        if j != i and max(a0 - x1, x0 - a1, 0) <= g["radius"] and max(b0 - y1, y0 - b1, 0) <= g["radius"])
                out[(x0, y0)] = (max(C.CROWD_FLOOR, 1.0 - C.CROWD_STEP * n), n)
        return out
```

Also initialise `self.crowd = {}` in `__init__` next to `self.networks = []` (observation reads it).

3e. `_compute_environment`: replace the `plant_src` block (the two `for kind ...` loops and the filter loop) with

```python
        plant_src = np.zeros(s.shape)
        for st in C.POLLUTER_STRUCTS:                        # per building, spread over its tiles; filters halve it
            for x, y in self.facilities(st):
                tiles = self.parts(x, y)
                p = (self.facility_spec(x, y).get("pollution") or 0.0) * self.module_factor(x, y, "filter") / len(tiles)
                for px, py in tiles:
                    plant_src[py, px] = p
```

3f. `dept_upkeep`: replace the utilities and per-department service sums:

```python
        spec_upkeep = lambda structs: sum(self.facility_spec(x, y)["upkeep"] for st in structs
                                          for x, y in self.facilities(st))
        out["utilities"] = spec_upkeep(C.UTILITY_TYPES) * self.funding.get("utilities", 1.0)
        ...
        for dept in ("police", "fire", "health", "education"):
            levees = C.upkeep_of(C.LEVEE) * self.n_buildings(C.LEVEE) if dept == "fire" else 0.0
            out[dept] = (spec_upkeep(C.STRUCTS_OF_DEPT[dept]) + levees) * self.funding.get(dept, 1.0)
```

3g. `_metrics`: replace the per-tile `util_structs` / `clean` computation and add tier info:

```python
        utils = [(x, y) for st in C.UTILITY_TYPES for x, y in self.facilities(st)]
        clean = sum(float(self.facility_spec(x, y).get("power") or 0) for x, y in utils
                    if self.facility_spec(x, y)["line"] in ("solar", "wind"))
        ...
            "utility_upkeep": round(sum(self.facility_spec(x, y)["upkeep"] for x, y in utils), 2),
        ...
            "tiers": self.tier_counts(),
            "downtown_uncovered": dict(st.get("downtown_uncovered", {})),
```

and the helper

```python
    def tier_counts(self) -> dict[str, list[int]]:
        """Buildings per tier for every line that has more than one tier, e.g. {'pump': [3, 1, 0]}."""
        out: dict[str, list[int]] = {}
        for st, line in C.LINE_OF_STRUCT.items():
            if len(C.TIER_LINES[line]) < 2:
                continue
            for x, y in self.facilities(st):
                counts = out.setdefault(line, [0] * len(C.TIER_LINES[line]))
                counts[self.facility_tier(x, y) - 1] += 1
        return out
```

3h. `civitas/sim/economy.py:plant_output_factor` — the third argument is now the line (for power lines it equals the
old kind, so `test_trade.py` keeps passing). Only rename the parameter to `line` and its uses; no behaviour change.

**Step 4:** `pytest tests/test_tiers.py -q` → PASS; `pytest -q -m "not slow"`. Expect possible failures in tests that
put several pumps / outlets / turbines next to each other and assert exact supply — check each one. If a test is about
something else (e.g. river flow), move the buildings ≥ 5 tiles apart. Do not weaken assertions about the tested
behaviour.

---

### Task 4: Services — per-building specs, downtown rule, recycling, crematoria

**Files:**
- Modify: `civitas/sim/services.py` (`coverage`, `_compute_services`, `_full_landfills`, `_update_crime`,
  `_fire_events`, `_garbage_and_health`, `_deathcare`), `civitas/sim/world.py:_landfill_fill_pct`
- Test: `tests/test_tiers.py`

**Step 1: Failing tests**

```python
def test_downtown_needs_tier_two_cover():
    w, y = _street(30)
    w.struct[y + 1, 8:14] = C.RES
    w.level[y + 1, 8:14] = 5
    w.place_building(C.POLICE, 10, y - 1)
    w.refresh()
    w._compute_services()
    assert w.police_cov[y + 1, 10] == pytest.approx(C.T1_DOWNTOWN)
    assert w.stats["downtown_uncovered"]["police"] == 6
    w.level[y + 1, 8:14] = 3                            # not downtown: full tier-1 cover
    w._compute_services()
    assert w.police_cov[y + 1, 10] == pytest.approx(1.0)
    w.level[y + 1, 8:14] = 5
    w.upgrade_building(10, y - 1)                        # police HQ
    w._compute_services()
    assert w.police_cov[y + 1, 10] == pytest.approx(1.0)
    assert w.stats["downtown_uncovered"]["police"] == 0


def test_service_capacity_radius_and_responses_follow_the_tier():
    w, y = _street()
    w.place_building(C.FIRE, 10, y + 1, tier=2)
    w.place_building(C.JAIL, 30, y + 1)
    w.upgrade_building(30, y + 1)                        # prison (2x2)
    w.refresh()
    w._compute_services()
    fire = w.service_info["fire_station"][0]
    assert fire["capacity"] == C.TIER_LINES["fire_station"][1]["capacity"] and fire["name"] == "fire_hq"
    w._update_crime()
    assert w.stats["jail_capacity"] == C.TIER_LINES["jail"][1]["capacity"]


def test_recycling_centres_never_fill_and_crematoria_need_no_plots():
    w, y = _street(30)
    w.struct[y + 1, 2:30] = C.RES
    w.level[y + 1, 2:30] = 3
    w.place_building(C.LANDFILL, 5, y - 1, tier=2)
    w.place_building(C.CREMATORIUM, 8, y - 1)
    w.refresh()
    w._compute_services()
    for _ in range(6):
        w._garbage_and_health()
    assert w.stats["garbage_collected"] > 0
    assert w._fstate(5, y - 1).get("fill", 0.0) == 0.0 and w._full_landfills() == set()
    w.deaths_month = 50.0
    w._deathcare()
    assert w.stats["deathcare_capacity"] >= C.TIER_LINES["cemetery"][1]["rate"]
    assert w.death_backlog == pytest.approx(0.0)
```

**Step 2:** Run → FAIL.

**Step 3: Implement**

3a. `coverage()` — radius / capacity may be `None` ("from each facility's tier spec"); add `cap_key` and `min_tier`:

```python
    def coverage(self, struct: int, radius: int | None, capacity: float | None, demand: np.ndarray,
                 dept: str | None = None, full: float = C.COVERAGE_FULL,
                 skip: set | None = None, traffic: bool = False, cap_key: str = "capacity",
                 min_tier: int = 1) -> tuple[np.ndarray, list[dict]]:
        """Coverage grid (0..1) of all facilities of one type.

        Full strength up to `full` x radius (road steps), fading to 0 at the radius. Scaled by capacity / load of
        the nearest facility and by the department's funding level. radius / capacity None: each facility's own
        (tier) spec, capacity read from `cap_key`. min_tier: only facilities of at least this tier.
        """
        sources = [s for s in self.facilities(struct) if self.building_network(*s) >= 0 and not (skip and s in skip)
                   and self.facility_tier(*s) >= min_tier]
        if not sources:
            return np.zeros((self.h, self.w)), []
        eff = self.funding_eff(dept) if dept else 1.0
        specs = [self.facility_spec(x, y) or {} for x, y in sources]
        radii = np.array([float(radius if radius is not None else sp["radius"]) for sp in specs])
        caps = np.array([(capacity if capacity is not None else sp[cap_key]) * eff * self.module_factor(x, y, "expansion")
                         for (x, y), sp in zip(sources, specs)])
        td, tl = self._road_field(tuple(sources), int(radii.max()))
        r = radii[np.maximum(tl, 0)]
        reach = td <= r
        load = np.bincount(tl[reach], weights=demand[reach], minlength=len(sources))
        ratio = np.where(load > caps, caps / np.maximum(load, 1e-9), 1.0)
        f = np.clip((r - td) / np.maximum(1e-9, (1.0 - full) * r), 0.0, 1.0) * reach if full < 1 else reach * 1.0
        cov = f * ratio[np.maximum(tl, 0)] * min(1.0, eff)
        if traffic:                     # emergency vehicles stuck in traffic
            cov = cov * self.congestion_penalty
        info = []
        for k, ((x, y), sp) in enumerate(zip(sources, specs)):
            it = {"x": x, "y": y, "load": round(float(load[k]), 1), "capacity": round(float(caps[k]), 1)}
            if len(C.TIER_LINES.get(sp.get("line"), ())) > 1:
                it.update(name=sp["name"], tier=sp["tier"])
            info.append(it)
        return np.clip(cov, 0.0, 1.0), info
```

3b. `_compute_services` — every `S["x"]["radius"], S["x"]["capacity"]` pair becomes `None, None` (with
`cap_key="seats"` for schools, `cap_key="collect"` for garbage; the university keeps `None, 1e9`). Police, fire and
health get the downtown rule:

```python
        dev_hi = developed & (self.level >= C.DOWNTOWN_LEVEL)
        police, info["police_station"] = self.coverage(C.POLICE, None, None, people, "police", traffic=True)
        police_hi, _ = self.coverage(C.POLICE, None, None, people, "police", traffic=True, min_tier=2)
        fire, info["fire_station"] = self.coverage(C.FIRE, None, None, developed.astype(float), "fire", traffic=True)
        fire_hi, _ = self.coverage(C.FIRE, None, None, developed.astype(float), "fire", traffic=True, min_tier=2)
        clinic, info["clinic"] = self.coverage(C.CLINIC, None, None, res, "health", traffic=True)
        hosp, info["hospital"] = self.coverage(C.HOSPITAL, None, None, res, "health", traffic=True)
        # downtown: tall buildings need tier-2 services (ladder trucks, HQ detectives, hospital care)
        down = lambda cov, hi: np.where(dev_hi, np.maximum(C.T1_DOWNTOWN * cov, hi), cov)
        self.police_cov, self.fire_cov = down(police, police_hi), down(fire, fire_hi)
        self.health_cov = down(np.maximum(clinic, hosp), hosp)
        self.downtown_gap = {"police": dev_hi & (police_hi < 0.3), "fire": dev_hi & (fire_hi < 0.3),
                             "health": dev_hi & (hosp < 0.3)}
        self.stats["downtown_uncovered"] = {k: int(v.sum()) for k, v in self.downtown_gap.items()}
```

Initialise `self.downtown_gap = {}` in `_init_services`.

Landfill `fill_pct` and the jail / cemetery info from specs:

```python
        for it in info["landfill"]:
            cap = self.facility_spec(it["x"], it["y"]).get("fill")
            if cap:
                it["fill_pct"] = round(100 * self._fstate(it["x"], it["y"]).get("fill", 0.0) / cap, 1)
        ...
        info["jail"] = [{"x": x, "y": y, "capacity": self.facility_spec(x, y)["capacity"]}
                        for x, y in self.facilities(C.JAIL)]
        info["cemetery"] = [{"x": x, "y": y, "plots_left": int(S["cemetery"]["plots"] - self._fstate(x, y).get("used", 0))}
                            for x, y in self.facilities(C.CEMETERY)]
        info["crematorium"] = [{"x": x, "y": y, "rate": self.facility_spec(x, y)["rate"]}
                               for x, y in self.facilities(C.CREMATORIUM)]
```

and `seats`:

```python
        seats = lambda struct: sum(self.facility_spec(x, y)["seats"] * self.module_factor(x, y, "expansion")
                                   for x, y in self.facilities(struct) if self.building_network(x, y) >= 0)
```

(update the three callers to `seats(C.SCHOOL)` etc.).

3c. `_full_landfills`:

```python
    def _full_landfills(self) -> set:
        out = set()
        for s in self.facilities(C.LANDFILL):
            cap = self.facility_spec(*s).get("fill")         # recycling centres / waste complexes never fill
            if cap and self._fstate(*s).get("fill", 0.0) >= cap:
                out.add(s)
        return out
```

3d. `_update_crime` jail capacity:

```python
        cap = sum(self.facility_spec(x, y)["capacity"] for x, y in self.facilities(C.JAIL)) + \
            sum(self.facility_spec(x, y)["holding"] for x, y in self.facilities(C.POLICE))
```

3e. `_fire_events`:

```python
        responses = int(sum(self.facility_spec(x, y)["responses"] for x, y in self.facilities(C.FIRE))
                        * self.funding_eff("fire"))
```

3f. `_garbage_and_health`: only landfills with a `fill` accumulate; use the largest radius; the incinerator coverage
reads its spec:

```python
        full = self._full_landfills()
        landfills = [s for s in self.facilities(C.LANDFILL) if self.building_network(*s) >= 0 and s not in full]
        if landfills and collected.sum() > 0:
            radius = max(self.facility_spec(*s)["radius"] for s in landfills)
            td, tl = self._road_field(tuple(landfills), radius)
            burn_share = 0.0
            if self.facilities(C.INCINERATOR):
                inc_cov, _ = self.coverage(C.INCINERATOR, None, None, prod, "health", full=1.0, cap_key="collect")
                burn_share = float((inc_cov * stock).sum() / max(1e-9, (self.garbage_cov * stock).sum()))
            per = np.bincount(tl[td <= radius], weights=collected[td <= radius], minlength=len(landfills))
            for k, (x, y) in enumerate(landfills):
                cap = self.facility_spec(x, y).get("fill")
                if not cap:
                    continue                                  # recycles what it collects
                st = self._fstate(x, y)
                before = st.get("fill", 0.0)
                st["fill"] = min(cap, before + float(per[k]) * (1.0 - min(1.0, burn_share)))
                if before < cap * 0.9 <= st["fill"]:
                    self.event(f"Landfill at ({x},{y}) is 90% full")
                if before < cap <= st["fill"]:
                    self.event(f"Landfill at ({x},{y}) is FULL and stopped collecting garbage")
```

3g. `_deathcare`: crematoria add capacity and are used before cemetery plots:

```python
        crems = [s for s in self.facilities(C.CREMATORIUM) if self.building_network(*s) >= 0]
        crem_cap = sum(self.facility_spec(x, y)["rate"] for x, y in crems) * eff
        morgue = len(self.facilities(C.HOSPITAL)) * C.HOSPITAL_MORGUE
        cap += crem_cap + morgue
        want = self.death_backlog + deaths
        handled = min(want, cap)
        left = handled - morgue - crem_cap                   # the rest goes to cemetery plots
```

(replace the old `cap += len(...) * C.HOSPITAL_MORGUE` and `left = ...` lines; keep the cemetery loop; update the
backlog event text to "build cemeteries or a crematorium").

3h. `world.py:_landfill_fill_pct`:

```python
    def _landfill_fill_pct(self) -> float | None:
        caps = [(self._fstate(x, y).get("fill", 0.0), self.facility_spec(x, y).get("fill"))
                for x, y in self.facilities(C.LANDFILL)]
        caps = [(f, c) for f, c in caps if c]
        if not caps:
            return None
        return round(100 * sum(f for f, _ in caps) / sum(c for _, c in caps), 1)
```

**Step 4:** `pytest tests/test_tiers.py tests/test_services.py -q` → PASS; `pytest -q -m "not slow"` → all pass.

---

### Task 5: Actions — `tier` parameter, kind names, `upgrade_facility module="tier"`, docs

**Files:**
- Modify: `civitas/core/actions.py`
- Test: `tests/test_tiers.py`

**Step 1: Failing tests**

```python
def _funded(w):
    for d in w.dept_funds:
        w.dept_funds[d] = 200000
    return w


def test_higher_tiers_are_gated_by_milestones_and_cost_their_price():
    w, y = _bank_world(World(seed=0, milestones=True))
    _funded(w)
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "tier": 2, "x": 34, "y": 5})
    assert not r.ok and "1,000 residents" in r.message
    w.milestones_reached.append("Town")
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pumping_station", "x": 34, "y": 5})
    assert r.ok and r.cost == C.TIER_LINES["pump"][1]["cost"]
    assert w.facility_spec(34, 5)["name"] == "pumping_station"


def test_two_by_two_riverbank_build_and_riverbank_rule():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "water_works", "x": 34, "y": y + 2}).ok
    r = apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 37, "y": y + 6})
    assert not r.ok and "riverbank" in r.message


def test_upgrade_action_charges_the_difference_and_reports_blocks():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_water", "kind": "pump", "x": 34, "y": 5}).ok
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    t1, t2 = C.TIER_LINES["pump"][:2]
    assert r.ok and r.cost == C.upgrade_cost(t1, t2) and "pumping station" in r.message
    r = apply_action(w, "utilities", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    assert not r.ok and "2x2" in r.message
    r = apply_action(w, "police", {"type": "upgrade_facility", "x": 34, "y": 5, "module": "tier"})
    assert not r.ok


def test_old_kinds_and_service_actions_still_work():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "utilities", {"type": "build_sewage", "kind": "treatment", "x": 34, "y": 11}).ok
    assert w.facility_tier(34, 11) == 2
    r = apply_action(w, "health", {"type": "build_hospital", "x": 37, "y": y + 2})
    assert r.ok and w.facility_spec(37, y + 2)["name"] == "hospital"
    r = apply_action(w, "health", {"type": "build_hospital", "tier": 3, "x": 37, "y": y + 5})
    assert r.ok and w.facility_spec(37, y + 5)["name"] == "medical_centre"
    r = apply_action(w, "education", {"type": "build_school", "kind": "elementary", "tier": 2, "x": 37, "y": y + 9})
    assert r.ok and w.facility_spec(37, y + 9)["name"] == "large_elementary"
    assert apply_action(w, "police", {"type": "build_police_station", "tier": 2, "x": 37, "y": y + 11}).ok


def test_health_upgrades_landfills_and_cemeteries():
    w, y = _bank_world()
    _funded(w)
    assert apply_action(w, "health", {"type": "build_cemetery", "x": 37, "y": y + 2}).ok
    assert apply_action(w, "health", {"type": "upgrade_facility", "x": 37, "y": y + 2, "module": "tier"}).ok
    assert w.struct[y + 2, 37] == C.CREMATORIUM
    assert apply_action(w, "health", {"type": "demolish", "x": 37, "y": y + 2}).ok


def test_action_docs_list_tiers_and_crowding():
    from civitas.core.actions import action_docs
    docs = action_docs()
    assert "pumping_station" in docs["build_water"][1] and "crowd" in docs["build_water"][1].lower()
    assert '"tier"' in docs["upgrade_facility"][1] or "tier" in docs["upgrade_facility"][0]
    assert "police_hq" in docs["build_police_station"][1]
```

**Step 2:** Run → FAIL.

**Step 3: Implement in `civitas/core/actions.py`**

3a. Resolution helper (put it above `_place_facility`):

```python
POWER_LINES = ("coal", "gas", "solar", "wind", "hydro", "nuclear")
WATER_LINES = ("pump", "tower")
SEWAGE_LINES = ("outlet",)


def _tier_spec(world: World, kind: str, p: dict, lines: tuple) -> dict:
    """Spec for a build action: kind is a line ('pump') or a tier name ('pumping_station'); "tier" picks a tier."""
    names = [t["name"] for line in lines for t in C.TIER_LINES[line]]
    kind = str(kind).lower().strip().replace(" ", "_")
    if kind not in C.TIER_BY_NAME or C.TIER_BY_NAME[kind][0] not in lines:
        raise ActionError(f"kind must be one of {names}")
    line, tier = C.TIER_BY_NAME[kind]
    tiers = C.TIER_LINES[line]
    if "tier" in p:
        tier = _int(p, "tier")
        if not 1 <= tier <= len(tiers):
            raise ActionError(f"{line} has tiers 1-{len(tiers)}: " + ", ".join(f"{t['tier']} {t['name']}" for t in tiers))
    spec = tiers[tier - 1]
    if spec.get("unlock"):
        _require(world, spec["unlock"])
    return spec
```

3b. `_place_facility` now takes the spec:

```python
def _place_facility(world: World, p: dict, spec: dict) -> tuple[str, float]:
    x, y = _point(world, p)
    name = spec["name"].replace("_", " ")
    terrain = C.WATER if spec.get("on_river") else C.LAND
    try:
        tiles = world.footprint_tiles(spec["struct"], x, y, terrain, spec["tier"])
    except ValueError as e:
        if spec.get("on_river"):
            raise ActionError(f"a {name} must be built on a river tile (not a bridge) touching a road")
        raise ActionError(str(e))
    if spec.get("needs_river") and not any(world.is_riverbank(tx, ty) for tx, ty in tiles):
        raise ActionError(f"a {name} must be on the riverbank (a land tile next to water)")
    _spend(world, "utilities", spec["cost"])
    world.place_building(spec["struct"], x, y, terrain, spec["tier"])
    net = world.building_network(x, y)
    warn = "" if world.networks[net].outside_connected else \
        " WARNING: that road network is not connected to the highway"
    res = next(k for k in ("power", "water", "sewage") if spec.get(k))
    return (f"built {name} (tier {spec['tier']}) at ({x},{y}){_size(spec)} adding {spec[res]} {res} "
            f"to network {net}{warn}"), spec["cost"]


def build_power_plant(world: World, p: dict) -> tuple[str, float]:
    return _place_facility(world, p, _tier_spec(world, p.get("kind", ""), p, POWER_LINES))


def build_water(world: World, p: dict) -> tuple[str, float]:
    return _place_facility(world, p, _tier_spec(world, p.get("kind", ""), p, WATER_LINES))


def build_sewage(world: World, p: dict) -> tuple[str, float]:
    return _place_facility(world, p, _tier_spec(world, p.get("kind", ""), p, SEWAGE_LINES))
```

(`hydro_dam` and `treatment_plant` locks now come from the specs' `unlock`.)

3c. `_build`, `_size`, `_place_service`, `build_school`:

```python
def _build(world: World, struct: int, dept: str, cost: float, p: dict, terrain: int = C.LAND,
           tier: int | None = None) -> tuple[int, int]:
    """Validate the footprint, charge the budget, place the building. Returns its top-left tile."""
    x, y = _point(world, p)
    try:
        world.footprint_tiles(struct, x, y, terrain, tier)
    except ValueError as e:
        raise ActionError(str(e))
    _spend(world, dept, cost)
    world.place_building(struct, x, y, terrain, tier)
    return x, y


def _size(what) -> str:
    """what: a struct code or a tier spec."""
    fw, fh = tuple(what["footprint"]) if isinstance(what, dict) else C.FOOTPRINT.get(what, (1, 1))
    return f" ({fw}x{fh} tiles from its top-left corner)" if fw * fh > 1 else ""


def _place_service(world: World, p: dict, kind: str) -> tuple[str, float]:
    spec = _tier_spec(world, kind, p, (C.TIER_BY_NAME[kind][0],))
    x, y = _build(world, spec["struct"], spec["dept"], spec["cost"], p, tier=spec["tier"])
    net = world.building_network(x, y)
    warn = "" if world.networks[net].outside_connected else \
        " WARNING: that road network is not connected to the highway"
    return f"built {spec['name'].replace('_', ' ')} at ({x},{y}){_size(spec)}{warn}", spec["cost"]
```

Delete `SERVICE_UNLOCK` (the specs carry `unlock`). `build_landmark` keeps calling `_size(spec["struct"])`.

```python
def build_school(world: World, p: dict) -> tuple[str, float]:
    kind = str(p.get("kind", "elementary")).lower().replace(" school", "")
    kind = SCHOOL_KINDS.get(kind.replace("_school", ""), kind)
    if C.TIER_BY_NAME.get(kind, ("",))[0] not in SCHOOL_KINDS.values():
        raise ActionError(f"kind must be one of {list(SCHOOL_KINDS)} (or a tier name such as large_elementary)")
    return _place_service(world, p, kind)
```

3d. Owners (replace `UPGRADE_OWNER` and the `OWNED_STRUCTS` service entries):

```python
UPGRADE_OWNER = {**{st: C.TIER_LINES[line][0]["dept"] for st, line in C.LINE_OF_STRUCT.items()},
                 C.TRAIN: "transport", C.HARBOUR: "transport"}
...
OWNED_STRUCTS = {
    "utilities": set(C.UTILITY_TYPES),
    **{d: set(C.STRUCTS_OF_DEPT[d]) for d in ("police", "fire", "health", "education")},
    ...
```

3e. `upgrade_facility`:

```python
def upgrade_facility(world: World, p: dict, role: str = "utilities") -> tuple[str, float]:
    x, y = _point(world, p)
    st_code = int(world.struct[y, x])
    if UPGRADE_OWNER.get(st_code) != role:
        raise ActionError(f"no {role} facility at ({x},{y}) (found {C.STRUCT_NAME.get(st_code, 'nothing')})")
    module = str(p.get("module", "")).lower()
    if module == "tier" or (not module and "tier" in p):
        return _upgrade_tier(world, x, y, role)
    if module not in C.MODULES:
        raise ActionError(f"module must be 'tier' or one of {list(C.MODULES)}")
    spec = world.facility_spec(x, y)
    kind = spec["line"] if spec else C.STRUCT_NAME[st_code]
    name = spec["name"] if spec else kind
    allowed = C.MODULES[module]["for"]
    if allowed != "any" and kind not in allowed:
        raise ActionError(f"{module} does not apply to a {name} (only {', '.join(allowed)})")
    x, y = world._anchor_of(x, y)
    state = world.facility_state.setdefault((x, y), {"built": world.month})
    mods = set(state.get("modules", ()))
    if module in mods and module != "overhaul":
        raise ActionError(f"the {name} at ({x},{y}) already has a {module}")
    base = spec["cost"] if spec else C.TRAIN_STATION["cost"]
    cost = round(base * C.MODULES[module]["cost_share"])
    _spend(world, role, cost)
    if module == "overhaul":
        state["built"] = world.month
        return f"overhauled the {name} at ({x},{y}): back to full output", cost
    mods.add(module)
    state["modules"] = sorted(mods)
    what = "halves its pollution" if module == "filter" else "+50% capacity"
    return f"added {module} to the {name} at ({x},{y}) ({what})", cost


def _upgrade_tier(world: World, x: int, y: int, role: str) -> tuple[str, float]:
    try:
        cur, nxt, (x0, y0), _ = world.next_tier(x, y)
    except ValueError as e:
        raise ActionError(str(e))
    if nxt.get("unlock"):
        _require(world, nxt["unlock"])
    cost = C.upgrade_cost(cur, nxt)
    _spend(world, role, cost)
    world.upgrade_building(x, y)
    gain = ", ".join(f"+{v:g} {k}" for k, v in C.tier_gain(cur, nxt).items())
    return (f"upgraded the {cur['name'].replace('_', ' ')} to a {nxt['name'].replace('_', ' ')} (tier {nxt['tier']}) "
            f"at ({x0},{y0}){_size(nxt)}" + (f": {gain}" if gain else "")), cost
```

Note: `MODULES["filter"]["for"]` lists `landfill` and `incinerator`; those are line names too, so filters still apply
to every landfill-line tier. `MODULES["overhaul"]["for"]` lists power lines; advanced coal / combined cycle are in the
same lines, so they can be overhauled.

3f. Docs. Add helpers above `action_docs()`:

```python
_GAIN_LABEL = {"power": "+{} power", "water": "+{} water", "sewage": "{} sewage", "capacity": "capacity {}",
               "seats": "{} seats", "collect": "collects {}/mo", "responses": "{} call-outs/mo", "rate": "{}/mo",
               "holding": "{} cells"}


def _tiers_doc(line: str) -> str:
    """'tier 2 pumping_station (1,000+ residents): $4,000, +650 water, upkeep $45/mo; ...'"""
    out = []
    for t in C.TIER_LINES[line][1:]:
        fw, fh = t["footprint"]
        bits = [f"${t['cost']:,}"] + [_GAIN_LABEL[k].format(f"{t[k]:,}") for k in C.GAIN_KEYS
                                       if t.get(k) and t[k] != C.TIER_LINES[line][0].get(k)]
        bits.append(f"upkeep ${t['upkeep']}/mo")
        extra = {"water_works": "filters dirty river water", "advanced_treatment": "removes 95% of the pollution",
                 "recycling_centre": "never fills up", "waste_complex": "never fills up",
                 "crematorium": "no plots to fill"}.get(t["name"])
        out.append(f"tier {t['tier']} {t['name']} ({C.UNLOCK_POP[t['unlock']]:,}+ residents"
                   + (f", {fw}x{fh}" if fw * fh > 1 else "") + "): " + ", ".join(bits + ([extra] if extra else [])))
    return "; ".join(out)


def _crowd_doc(line: str) -> str:
    g = next((g for g in C.CROWD_GROUPS.values() if line in g["lines"]), None)
    return (f" Crowding: each other {line} (any tier) within {g['radius']} tiles cuts output {int(C.CROWD_STEP * 100)}% "
            f"(min {int(C.CROWD_FLOOR * 100)}%) - {g['why']}; spread them out or upgrade.") if g else ""


DOWNTOWN_DOC = (f" Downtown (level {C.DOWNTOWN_LEVEL}-5 buildings) gets only {int(C.T1_DOWNTOWN * 100)}% cover from "
                "tier-1 buildings; a tier-2+ building in range gives full cover.")
```

Then:
- `_svc_doc(kind)` appends `" Higher tiers (add \"tier\": 2 or upgrade): " + _tiers_doc(line)` when the line has more
  than one tier, plus `DOWNTOWN_DOC` for `police_station`, `fire_station`, `clinic`, `hospital`.
- `build_power_plant` / `build_water` / `build_sewage` docs: after the existing per-kind text, append
  `_tiers_doc(...)` for every multi-tier line in their group, a line for nuclear (`$40,000, +2000 power, 2x2
  riverbank, uses 100 water for cooling, 10,000+ residents`) and `_crowd_doc` for pump / tower / outlet / wind.
  Every kind value in the examples still works. Build the text from the specs (no hard-coded numbers except the
  unlock labels coming from `UNLOCK_POP`).
- `upgrade_facility` doc: example stays; prepend
  `'module "tier": move the building one tier up its line (see the build actions) for the price difference x1.2; '
  'bigger tiers grow into free tiles around it; keeps age and modules. '` before the modules list.

**Step 4:** `pytest tests/test_tiers.py -q` → PASS; `pytest -q -m "not slow"` → all pass. Also check
`civitas prompt -c configs/baseline.toml --role utilities | wc -w` is not more than ~250 words above the value before this
task (run it before editing the docs to know the before-value).

---

### Task 6: Frames carry tiers and tier-based footprints

**Files:**
- Modify: `civitas/core/env.py:192-222` (`frame`)
- Test: `tests/test_tiers.py`

**Step 1: Failing test**

```python
def test_frames_carry_tiers_and_multi_tile_upgrades():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    w.place_building(C.PUMP, 34, y + 2, tier=3)
    w.place_building(C.POLICE, 37, y + 5, tier=2)
    f = env.frame()
    i = (y + 2) * w.w + 34
    assert f["levels"][i] == "3" and f["levels"][i + w.w + 1] == "3"
    assert f["levels"][(y + 5) * w.w + 37] == "2"
    assert [i, 2, 2] in f["multi"]
```

**Step 2:** Run → FAIL.

**Step 3: Implement** in `frame()`:

```python
            "levels": "".join(str(int(v)) for v in w.frame_levels().ravel()),
            ...
            "multi": w.multi_buildings(),
```

**Step 4:** `pytest tests/test_tiers.py -q` → PASS.

---

### Task 7: Observations — tiers, crowding, upgrade offers, downtown gaps

**Files:**
- Modify: `civitas/sim/world.py` (add `upgrade_options`), `civitas/core/observation.py`
- Test: `tests/test_tiers.py`

**Step 1: Failing test**

```python
def test_prompts_show_tiers_crowding_upgrades_and_downtown():
    env = CivitasEnv(GameConfig(seed=0, months=12, milestones=False))
    w, y = _bank_world(env.world)
    for yy in (3, 5, 7):
        w.place_building(C.PUMP, 34, yy)
    w.place_building(C.POLICE, 37, y + 1)
    w.struct[y + 3:y + 8, 37] = C.RES                # a column of towers east of the x=36 road
    w.level[y + 3:y + 8, 37] = 5
    w.refresh()
    w._compute_services()
    text = env.observe("utilities")["text"]
    assert "crowd 0.7" in text and "pumping station" in text.lower()
    obs = env.observe("police")
    assert obs["police"]["downtown_uncovered"] == 5
    assert obs["police"]["upgrades"][0]["to"] == "police_hq"
    assert "police hq" in obs["text"].lower()
```

**Step 2:** Run → FAIL.

**Step 3: Implement**

3a. `World.upgrade_options` (next to `next_tier`):

```python
    def upgrade_options(self, structs, need=None, k: int = 4) -> list[dict]:
        """Unlocked tier upgrades with room to grow for facilities of these types, most needed first.

        need: a per-tile grid (summed around each building) or {(x, y): value}."""
        out = []
        for st in structs:
            for x, y in self.facilities(st):
                try:
                    cur, nxt, _, _ = self.next_tier(x, y)
                except ValueError:
                    continue
                if nxt.get("unlock") and not self.unlocked(nxt["unlock"]):
                    continue
                o = {"x": x, "y": y, "from": cur["name"], "to": nxt["name"], "tier": nxt["tier"],
                     "cost": C.upgrade_cost(cur, nxt), "gain": C.tier_gain(cur, nxt)}
                cf = self.crowd.get((x, y), (1.0, 0))[0]
                if cf < 1.0:
                    o["crowd"] = round(cf, 2)
                if isinstance(need, dict):
                    o["need"] = round(float(need.get((x, y), 0.0)), 1)
                elif need is not None:
                    o["need"] = round(float(need[max(0, y - 6):y + 8, max(0, x - 6):x + 8].sum()), 1)
                out.append(o)
        out.sort(key=lambda o: (-o.get("need", 0.0), o.get("crowd", 1.0), o["cost"]))
        return out[:k]
```

3b. `observation.py`, utilities branch (where `obs["networks"]` is set):

```python
        obs["upgrades"] = world.upgrade_options(C.UTILITY_TYPES)
```

and in `_candidate_sites`, when `world.unlocked("water_works") or world.unlocked("advanced_treatment") or
world.unlocked("nuclear")`:

```python
        out["riverbank_2x2"] = world.site_candidates(np.ones(world.struct.shape), radius=1, k=3, riverbank=True, size=2)
```

(build the dict in a variable `out`, then return it).

3c. Service sections:

```python
# _police_section
    gap = world.downtown_gap.get("police")
    out["downtown_uncovered"] = int(gap.sum()) if gap is not None else 0
    out["upgrades"] = world.upgrade_options([C.POLICE, C.JAIL], need=people * gap if gap is not None else None, k=3)
# _fire_section (dev as defined there)
    gap = world.downtown_gap.get("fire")
    out["downtown_uncovered"] = int(gap.sum()) if gap is not None else 0
    out["upgrades"] = world.upgrade_options([C.FIRE], need=dev * gap if gap is not None else None, k=3)
# _health_section
    gap = world.downtown_gap.get("health")
    out["downtown_uncovered"] = int(gap.sum()) if gap is not None else 0
    fills = {(l["x"], l["y"]): l.get("fill_pct", 0.0) for l in world.service_info.get("landfill", [])}
    plots = {(c["x"], c["y"]): 100 - 100 * c["plots_left"] / C.SERVICES["cemetery"]["plots"]
             for c in world.service_info.get("cemetery", [])}
    out["crematoria"] = world.service_info.get("crematorium", [])
    out["upgrades"] = {
        "care": world.upgrade_options([C.CLINIC, C.HOSPITAL], need=res * gap if gap is not None else None, k=2),
        "garbage": world.upgrade_options([C.LANDFILL], need=fills, k=2),
        "deathcare": world.upgrade_options([C.CEMETERY], need=plots, k=2),
    }
# _education_section
    out["upgrades"] = world.upgrade_options([C.SCHOOL, C.HIGHSCHOOL],
                                            need=res * (n[0].sum() + n[1].sum()) / total, k=3)
```

3d. Rendering. Add helper:

```python
def _upgrades(lst: list[dict]) -> str:
    return "; ".join(f"({u['x']},{u['y']}) {u['from'].replace('_', ' ')} -> {u['to'].replace('_', ' ')} "
                     f"${u['cost']:,} ({', '.join(f'+{v:g} {k}' for k, v in u['gain'].items())})"
                     + (f" crowd {u['crowd']}" if "crowd" in u else "") for u in lst)
```

- Networks line: the plant string gets `+ (f" crowd {p['crowd']}" if "crowd" in p else "")` (the `kind` is already
  the tier name).
- After the networks block, if `obs.get("upgrades")`:
  `L.append('Tier upgrades available (upgrade_facility {"module":"tier"}): ' + _upgrades(obs["upgrades"]))`.
- `riverbank_2x2` (if present): `L.append("  riverbank 2x2 blocks (water works / advanced treatment / nuclear): " +
  _sites(cs["riverbank_2x2"]))`.
- `_facilities`: prefix the tier name when present:
  `out.append((f"{f['name'].replace('_', ' ')} " if f.get("tier", 1) > 1 else "") + f"({f['x']},{f['y']})" + ...)`.
- Police / fire / health role blocks: after the stations line,
  `if d.get("downtown_uncovered"): L.append(f"Downtown (level 4-5) tiles without tier-2 cover: {d['downtown_uncovered']} "
  f"(tier-1 buildings give them only {int(C.T1_DOWNTOWN * 100)}%)")` and
  `if d.get("upgrades"): L.append('Tier upgrades (upgrade_facility {"module":"tier"}): ' + _upgrades(d["upgrades"]))`.
  Health renders its three lists on one line: `"; ".join(f"{k}: {_upgrades(v)}" for k, v in d["upgrades"].items() if v)`,
  and crematoria in the deathcare line: `f"; crematoria: {_facilities(d['crematoria'], ('rate',))}"` when non-empty.
- Education block: the same upgrades line.
- `MAP_LEGEND`: add `u nuclear plant  z crematorium` and "(bigger tiers of a building keep its letter; the facility
  lists name the tier)".

**Step 4:** `pytest tests/test_tiers.py -q` → PASS; `pytest -q -m "not slow"` → PASS.

**Step 5: Prompt budget check**

```bash
for r in utilities police fire health education; do civitas prompt -c configs/baseline.toml --role $r | wc -w; done
```

Compare with the same command before Task 5 (keep a note of it). Each role should grow by at most ~150 words at
month 0; the upgrade lines are empty at month 0, so also check a mid-game prompt: run
`civitas run -c configs/baseline.toml --seed 0` and look at a month-100 utilities prompt in the run's
trajectories file, if the scripted runner logs prompts; otherwise build one in a Python shell by ticking an env with
scripted agents for 100 months and calling `env.observe("utilities")["text"]`.

---

### Task 8: Scripted baselines use tiers

**Files:**
- Modify: `civitas/agents/scripted.py` (`ScriptedUtilities`, `_ServiceAgent`, `ScriptedPolice`, `ScriptedFire`,
  `ScriptedHealth`, `ScriptedEducation`)

The scripted agents read only the observation plus catalog constants (as today with `C.CATALOG`).

**Step 1: Utilities.** Add at the top of `ScriptedUtilities.act`, after `pick` is defined:

```python
        locked = set(obs.get("progress", {}).get("locked", []))

        def best(line, left, reserve=1000):
            """Highest unlocked 1x1 tier of a line we can afford (tiers above 1 keep a reserve)."""
            for t in reversed(C.TIER_LINES[line]):
                if (tuple(t["footprint"]) == (1, 1) and t.get("unlock") not in locked
                        and left >= t["cost"] + (reserve if t["tier"] > 1 else 0)):
                    return t
            return None
```

Fix the overhaul block (plant kinds are tier names now):

```python
        for p in net["plants"]:
            line, tier = C.TIER_BY_NAME.get(p["kind"], (p["kind"], 1))
            if p.get("age_years", 0) >= 7 and line in ("coal", "gas", "solar", "wind"):
                cost = round(C.TIER_LINES[line][tier - 1]["cost"] * C.MODULES["overhaul"]["cost_share"])
                ...
```

Before the power / water / sewage build blocks, one tier upgrade per turn when it fixes a shortage or crowding:

```python
        need_s = 1.2 * net["sewage_demand"] + 5
        short = {"power": net["power_supply"] < need_p, "water": net["water_supply"] < need_w,
                 "sewage": net["water_supply"] > 0 and net["sewage_supply"] < need_s}
        for u in obs.get("upgrades", []):
            res = next((k for k in ("power", "water", "sewage") if k in u["gain"]), None)
            if res and (short[res] or u.get("crowd", 1.0) <= 0.7) and budget >= u["cost"] + 1000:
                acts.append({"type": "upgrade_facility", "x": u["x"], "y": u["y"], "module": "tier"})
                budget -= u["cost"]
                gap = {"power": need_p, "water": need_w, "sewage": need_s}[res] - net[f"{res}_supply"]
                short[res] = short[res] and u["gain"][res] < gap
                why.append(f"upgrade {u['from']} -> {u['to']}")
                break
```

Then gate each build block on `short[...]` instead of the raw comparison and choose tiers with `best`:

```python
        if short["power"]:
            t = (best("coal", budget - 500) if c["population"] > 600 else None) or best("wind", budget)
            s = t and pick(sites.get("far_from_homes" if t["line"] == "coal" else "windy", []) or sites.get("near_homes", []))
            if t and s:
                acts.append({"type": "build_power_plant", "kind": t["name"], "x": s["x"], "y": s["y"]})
                budget -= t["cost"]
                why.append(f"power {net['power_supply']}<{need_p:.0f}: {t['name']}")
            else:
                ...  # existing message to the mayor
        if short["water"]:
            banks = [b for b in sites.get("riverbank_next_to_road", []) if b["network"] == net["id"]]
            t = best("pump", budget)
            s = pick(banks)
            if s and t:
                acts.append({"type": "build_water", "kind": t["name"], "x": s["x"], "y": s["y"]})
                budget -= t["cost"]
                why.append(f"{t['name']} on riverbank")
            elif not banks and (t := best("tower", budget)):
                ...  # existing tower branch with kind t["name"] and cost t["cost"]
            else:
                ...  # existing messages
        if short["sewage"]:
            t = best("outlet", budget)
            banks = [b for b in sites.get("riverbank_for_sewage", []) if b["network"] == net["id"]]
            s = pick(banks)
            if s and t:
                acts.append({"type": "build_sewage", "kind": t["name"], "x": s["x"], "y": s["y"]})
                budget -= t["cost"]
                why.append(f"sewage {net['sewage_supply']}<{need_s:.0f}: {t['name']}")
            elif s:
                msgs.append({"to": "mayor", "text": f"Utilities needs ${C.CATALOG['outlet']['cost']:,} for sewage capacity."})
```

Remove the old `need_s` / `locked` lines inside the sewage block (they moved up).

**Step 2: Service agents.** Add to `_ServiceAgent`:

```python
    def _upgrade(self, obs, acts, why, ups, reason, min_need=0.0, only=None) -> bool:
        """Upgrade the most needed facility one tier (optionally only those whose current tier is in `only`)."""
        left = (obs["your_budget"] or 0.0) - sum(a.get("_cost", 0) for a in acts)
        for u in ups or []:
            if (only is None or u["from"] in only) and u.get("need", 0.0) > min_need and left >= u["cost"] + 500:
                acts.append({"type": "upgrade_facility", "x": u["x"], "y": u["y"], "module": "tier", "_cost": u["cost"]})
                why.append(f"{reason}: {u['from']} -> {u['to']}")
                return True
        return False
```

Police (before the new-station branch):

```python
        if d.get("downtown_uncovered", 0) > 0:
            self._upgrade(obs, acts, why, d.get("upgrades"), "downtown needs an HQ", only=("police_station",))
        if c["population"] > 1500 and d["inmates"] > 0.8 * max(1, d["jail_capacity"]):
            if not self._upgrade(obs, acts, why, d.get("upgrades"), "jail full", min_need=-1, only=("jail",)):
                ...  # existing build_jail call
```

Fire: the same downtown upgrade with `only=("fire_station",)` before its station branch.

Health:

```python
        ups = d.get("upgrades", {})
        if d.get("downtown_uncovered", 0) > 0:
            self._upgrade(obs, acts, why, ups.get("care"), "downtown care", only=("hospital",))
        # a nearly full landfill becomes a recycling centre instead of opening another landfill
        recycled = self._upgrade(obs, acts, why, ups.get("garbage"), "landfill filling up", min_need=70)
        if not recycled and pop >= 100 and (...existing garbage condition...):
            ...  # existing build_landfill
        # cemeteries running out of plots become crematoria
        if low_plots and self._upgrade(obs, acts, why, ups.get("deathcare"), "cemetery full", min_need=80):
            pass
        elif pop >= 300 and (...existing deathcare condition...):
            ...  # existing build_cemetery
```

Education: when enrolment is low but homes are in range (seats short, not coverage), upgrade instead of building:

```python
        el = d["elementary"]
        if d["elementary"]["students"] > 30 and (e["elementary"] or 0) < 90:
            if not ((el.get("homes_in_range_pct") or 0) >= 85
                    and self._upgrade(obs, acts, why, d.get("upgrades"), "more seats", only=("elementary_school",))):
                ...  # existing elementary build
        # the same pattern for high schools with only=("high_school",)
```

Also fix the stale hospital character in `_build`'s `own` map: `"build_hospital": "h"` (it was `"M"`, which is now
mixed use). This changes hospital spacing slightly; note it in the benchmark call-out.

**Step 3:** `pytest -q -m "not slow"` → PASS. Then `civitas run -c configs/baseline.toml --seed 0` and check the
decisions in the run log include `upgrade_facility` with `module: tier` and tier-2 builds after the Town milestone.

---

### Task 9: Benchmark — run, tune the scripted agents, record the numbers

**Step 1:** `civitas bench -c configs/baseline.toml --seeds 0-4 --workers 5 | tee /tmp/claude-tiers-after.txt`

**Step 2:** Compare with `/tmp/claude-tiers-before.txt`. Checks:
- no bankruptcies; city score mean inside 60–80 (`tests/test_benchmark.py` enforces this), aim ≈ 77;
- final metrics `tiers` show upgrades happening (e.g. pumps at tier 2+, recycling centres);
- building counts dropped versus the before-run (count characters in the last frame as in the brainstorming session);
- `time` per episode still ≈ 4 s.

**Step 3: Tune the scripted agents only** (reserves, the crowd threshold 0.7, when to prefer upgrades), not the balance
constants, unless a tier is clearly never worth it (then adjust that tier's numbers in `catalog.py` and note it). Re-run
Step 1 after each change.

**Step 4:** `pytest -q` (full, includes the slow benchmark tests) → PASS.

**Step 5:** Also run `civitas run -c configs/region_trio.toml --seed 0` → finishes, no exceptions, ~12 s.

---

### Task 10: Viewer — tier models, nuclear plant, crematorium, names

**Files:**
- Modify: `civitas/viewer/recipes.js` (`special`), `civitas/viewer/pixel.js` (`_objParams`, construction check ~line 101),
  `civitas/viewer/scene.js` (`SPECIALS`, `_specials`, `_buildSpecial`), `civitas/viewer/app.js` (`NAMES`, `MINI`, tile name)
- Check: `civitas/viewer/classic.js` and `region.js` for per-character colour tables; add `u` and `z` where such tables exist.

**Step 1: `scene.js`**
- `SPECIALS` string: append `uz`.
- In `_specials`, include the tier in the key so an upgrade rebuilds the model:

```js
      const lv = Math.max(1, +(frame.levels[i] || 0));
      if (SPECIALS.includes(ch) && !part.has(i)) want.set(`${i}:${ch}:${lv}`, { i, ch, lv, size: size.get(i) });
```

  and destructure `lv` in the loop and call `this._buildSpecial(ch, i, lv)`.
- `_buildSpecial(ch, i, lv = 1)`: add branches for `u` (two cooling towers as lathes / tapered cylinders, a reactor
  dome, a steam plume emitter via `g.userData.stacks`) and `z` (small chapel box, pitched roof, one chimney). At the end,
  before `return g`, add `if (lv >= 2) this._tierExtras(g, ch, lv, bx);` where

```js
  // higher tiers (frame level = tier): an annex block with an accent roof, a second one at tier 3
  _tierExtras(g, ch, lv, bx) {
    const c = TIER_ACCENT[ch];
    if (!c) return;
    bx(0.34, 0.22 + 0.12 * lv, 0.3, 0.3, 0, -0.3, "#d9dde2");
    bx(0.36, 0.04, 0.32, 0.3, 0.22 + 0.12 * lv, -0.3, c);
    if (lv >= 3) { bx(0.3, 0.3 + 0.1 * lv, 0.26, -0.32, 0, -0.32, "#cfd4da"); bx(0.32, 0.04, 0.28, -0.32, 0.3 + 0.1 * lv, -0.32, c); }
  }
```

  with a module-level `const TIER_ACCENT = { E: "#4a4c50", G: "#d9822b", S: "#1f3a73", U: "#1f8a8a", V: "#2fa3b3",
  "*": "#3d7a4a", K: "#1f4fa8", J: "#7d8388", F: "#8e1c1c", h: "#3fd0c9", L: "#6b7a3a", N: "#f5d547", B: "#f5d547" };`
  (export it from `recipes.js` and import it in `scene.js` so both renderers share it).

**Step 2: `recipes.js`**
- `export function special(b, ch, v = 0, season = null, lv = 1)`; add `u` and `z` branches mirroring scene.js (same
  argument order via the `bx` / `cy` wrappers already defined there); at the end `if (lv >= 2) tierExtras(b, ch, lv);`
  with the same geometry as `_tierExtras`, using `b.box` through the local `bx` wrapper and `{ win: WIN.factory(lv) }`
  on the annex.

**Step 3: `pixel.js:_objParams`** (special branch):

```js
      const lv = Math.max(1, +(f.levels[i] || 0));
      return { key: `s|${ch}|${fw}|${fh}|${q}|${lv}`, fw, fh,
               build: (b) => { b.scale(k, kz).turn(q); R.special(b, ch, i % 7, season, lv); }, build0: ch !== "H" };
```

and line ~101 so an upgrade plays the construction animation:

```js
        const built = ("RCIOM".includes(is) && +L[i] > 0) || (SPECIALS.includes(is) && is !== "H" && (is !== was || L[i] !== prev.levels[i]));
```

(check the surrounding code: `built && !had` probably drives the animation; if so, make `had` false when the level
changed for a special.)

**Step 4: `app.js`**
- `NAMES`: `u: "nuclear plant", z: "crematorium"`; `MINI`: `u: "#9fd3c7", z: "#6a6a6a"`.
- Tier names and the tile title (line ~511):

```js
const TIER_NAMES = { E: { 2: "advanced coal plant" }, G: { 2: "combined-cycle gas plant" }, S: { 2: "solar power plant" },
  W: { 2: "advanced wind turbine" }, U: { 2: "pumping station", 3: "water works" }, V: { 2: "inland water plant" },
  "*": { 3: "advanced treatment plant" }, L: { 2: "recycling centre", 3: "waste processing complex" },
  h: { 3: "medical centre" }, K: { 2: "police HQ" }, J: { 2: "prison" }, F: { 2: "fire HQ" },
  N: { 2: "large elementary school" }, B: { 2: "large high school" } };
const nameOf = (ch, lvl) => (TIER_NAMES[ch] && TIER_NAMES[ch][lvl]) || NAMES[ch] || ch;
```

  and in the tile-title function use `nameOf(t.ch, t.lvl)` instead of `NAMES[t.ch] || t.ch` for non-zone tiles.

**Step 5: Look at it.** Run a baseline episode and serve it:

```bash
civitas run -c configs/baseline.toml --seed 0
civitas serve
```

Use the playwright skill (or open http://localhost:8765) to screenshot the replay at month 120 in pixel view and 3D view
(`X`): tier-2/3 buildings show annexes, 2x2 water works / prisons draw as one model, hovering shows "pumping station",
no console errors. Also load an old replay from `runs/` made before this change: facilities read as tier 1.

---

### Task 11: Docs and the LLM smoke test

**Files:**
- Modify: `README.md` (building tiers section + table from the design doc, `tier` parameter, `upgrade_facility
  module "tier"`, crowding and downtown rules, `TIERS.<line>.<tier>.<field>` overrides, reference numbers)
- Modify: `docs/MECHANICS.md` (add a status row "11 | Building tiers: ladders per service, upgrades in place, crowding,
  downtown cover [`catalog.TIER_LINES`, `world.facility_spec`]" and a "Tiers" paragraph under Details)
- Modify: `CLAUDE.md` (a short "Tiers" paragraph in Architecture: `TIER_LINES`, `facility_spec()`, tier in
  `facility_state` not `level`, frames overlay tiers via `frame_levels()`; don't read `C.CATALOG[kind]` /
  `C.SERVICES[kind]` for placed buildings; update Reference numbers from Task 9)

**Step 1:** Write the docs.

**Step 2: LLM smoke test** (per the user's standing preference after observation changes):

```bash
civitas check-llm
civitas run -c configs/smoke.toml --set game.months=6 --set game.act_every=3
```

Then check the run's `trajectories.jsonl` for parse errors and `finish_reason=length`, and that the utilities agent's
prompt shows tier docs without errors from the model about unknown kinds. If `ARCEE_API_KEY` is missing, say so in the
final report instead of skipping silently.

**Step 3:** Final `pytest -q` → PASS. Report the before / after benchmark table and call out that this is a
benchmark change.
