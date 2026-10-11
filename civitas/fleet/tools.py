"""Fleet tools: what a tool-calling agent can look up and do on its turn.

`ToolBox(env, role)` exposes OpenAI-style function schemas (`schemas()`) and runs calls (`call(name, arguments)`,
never raises). Look-ups read the world through the observation builders; action tools are the role's existing
actions (schemas derived from `actions.action_docs()`) run through `apply_action`; turn-control tools write the
fleet memory (`FleetState`) and collect council messages and the end-of-turn report.
"""
from __future__ import annotations

import copy
import json
import math
import re
import weakref
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Callable, NamedTuple

import numpy as np

from ..bench import live
from ..bench.collab import BUILDS, MONEY, TEXT_CHARS, _amount as asked_amount, _tiles as text_tiles, action_tile, \
    answer, is_request, replies
from ..core.actions import (NO_DENSITY, OWNED_STRUCTS, POWER_LINES, REZONE_COST, SCHOOL_KINDS, SEWAGE_LINES,
                            WATER_LINES, _kind_name, _line, _point, _rect, action_docs, apply_action, zone_cost,
                            zone_density, zone_price, zone_tiles)
from ..core.observation import (MAP_LEGEND, _far_from_homes, _residents, _road_extensions, _suggest_stops,
                                _uncrowded_first, _zone_suggestions, build_observation)
from ..core.region import DEAL_ACTIONS
from ..core.money import RESERVE_MIN, reserve, reserve_months as _reserve_months   # re-exported
from ..core.roles import ROLES
from ..sim import catalog as C
from ..sim.grid import box_sum, label_components, road_bfs, shift, spread_to_tiles
from ..sim.world import own_supply
from .fmt import STYLES, TOOL_MESSAGE_CHARS, a_or_an, advice, briefing_style, cash, cut, is_open, money, months, pct, \
    signed
from .memory import DEPARTMENTS, HISTORY_KINDS, LIMITS, FleetState

RESULT_CHARS = TOOL_MESSAGE_CHARS - 200     # the agent cuts tool messages at TOOL_MESSAGE_CHARS; lists are trimmed below
MAX_SITES = 6
MAX_RADIUS = 4
MAX_MONTHS = 12
MAX_ENTRIES = 20             # history entries / council messages per call
REPORT_MAX_CHARS = 1000      # end_turn reports are cut to this
REPEAT_ERROR_CHARS = 300     # the earlier error quoted when a repeated failed action or bad call is refused
INSPECTS_FREE = 3            # inspects run in a turn before their results point to map_overview (smoke v9-v11:
                             # planning made 5.4 a turn, other roles 0.4-1.2)
INSPECTS_AFTER_MAP = 2       # inspects run after map_overview before their results say its map shows the layout
                             # (smoke v12-v16: planning ran it on 35 of 40 turns, then inspected 3.8 times a turn)
OWN_NAMED = 2                # this turn's own build_road calls named when a road rebuilds them
BLOCKERS_NAMED = 3           # tiles named per kind when a road is blocked
DEZONES_NAMED = 2            # dezone rects suggested for the lots in its way
ZONES_BESIDE = 2             # zone rects suggested beside a rectangle with no empty land
ZONE_LEFT_SHOWN = 100        # a planning zone leaving less than this says what the rest pays for (_zone_left)
CONTENT_KINDS = 5            # kinds of tile named when a failed zone / dezone says what the rectangle holds
RIVER_BANDS = 3              # row bands the Map block describes the river in
ROAD_LINE_MIN = 3            # the Map block's road lines: straight runs of connected road at least this long,
ROAD_LINES = 8               # the longest this many named, then "+N more"
# branch roads (branch_sites), offered where no dead end extends: off a road tile, straight through at most
# BRANCH_CLEAR zoned tiles (planning zones strips 2 deep beside a road) and on for BRANCH_MIN-BRANCH_LEN tiles of
# empty land or river (bridges; observation's extensions run 3-15), opening at least BRANCH_OPENS land tiles; listed
# branches that run the same way start more than BRANCH_APART tiles apart
BRANCH_CLEAR, BRANCH_MIN, BRANCH_LEN, BRANCH_OPENS, BRANCH_APART = 2, 3, 15, 5, 4
# the most land a road_extension site opens (opens_land: empty land without road access within 2 tiles of it): a dead
# end and its BRANCH_LEN-tile extension, 2 tiles round, 20 x 5 (a branch's BRANCH_LEN tiles, 19 x 5)
OPENS_MAX = (BRANCH_LEN + 5) * 5
DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
BUILT_NAMES = {"R": "home", "C": "shop", "I": "factory", "O": "office", "M": "mixed-use block"}   # a developed lot
# what planning's road_extension price buys (road_price: the road and zoning the land it opens), for the mayor's needs
# line and the report
ROAD_NAME = "road into open land and zoning it"
GAP_MIN, GAP_SHARE = 100, 0.1   # a department within max($100, 10% of its plan target) of it counts as funded
SPEND_MONTHS = 3             # the mayor's Departments block: what each department spent over this many months
IDLE_MULT, IDLE_MIN = 2, 1000   # a department keeps max(IDLE_MULT x that, its dearest ready first building,
                                # IDLE_MIN); funds above it by IDLE_MIN or more are idle
USED_SHARE = 0.5             # below its target only if it spent more than this share of the most it held (or never
                             # held what it keeps): DeptMoney.wants_more
# what a department's money is for (DeptMoney.kind), most urgent first: the mayor's Departments block and the to-fund
# line go in this order (Funding.order), turn order within a kind. The first three have a needs line (NEED_KINDS);
# "first/supply" is a first building (Need.first), utilities' supply (first_build names a utilities building only
# while it is wanted) or a levee; "growth" is planning's road and zoning
KINDS = ("first/supply", "growth", "another", "blocked", "nothing", "idle")
KIND_RANK = {k: i for i, k in enumerate(KINDS)}
NEED_KINDS = KINDS[:3]
# the kinds a top-up that leaves the treasury under the reserve is refused for (ToolBox._reserve_refusal): another
# building, and money a department has no use for now (nothing ready, or idle already). Review: $3,000 to transport with
# nothing ready left $500 of a $3,000 reserve unrefused, and then $400 for another police station was refused. Blocked
# first buildings pass: the money waits for planning's lot
RESERVE_REFUSED = (KINDS[2], KINDS[4], KINDS[5])
# a loan over what first/supply, growth and the reserve need by at least this much is refused (ToolBox._loan_refusal):
# a loan rounded up to the next $1,000 isn't. Review replay of smoke v20 seed 0, month 9: $5,000 borrowed where $3,850
# was for first/supply, growth and the reserve
LOAN_SLACK = 1000
FUNDED_SHOWN = 4             # still_to_fund: entries named before "+N more"
ROAD_REMOVERS = {"bulldoze"}  # the only actions that remove road tiles (demolish takes buildings, never roads)
# buildings that link the road networks they touch to the region, as World.refresh() counts them (metro stations
# don't) -> their name in a refusal
LINKS = {C.TRAIN: "train station", C.HARBOUR: "harbour", C.AIRPORT: "airport"}
LINK_OWNERS = {r for r, owned in OWNED_STRUCTS.items() if owned & set(LINKS)}     # roles whose demolish takes one
# guarded action that removes roads or links -> what its refusal and confirm parameter say they'd be cut off from
CUT_FROM = {"bulldoze": "highway", "demolish": "region"}
# guarded action -> its confirm parameter's description
CONFIRM = {**{a: f"true only to repeat a {a} refused for cutting roads off the {c}, once you are sure"
              for a, c in CUT_FROM.items()},
           "allocate_budget": "true only to repeat an allocate_budget refused this turn with the same department and "
                              "amount, once you are sure; on any other call it doesn't count",
           "take_loan": "true only to repeat a take_loan refused this turn with the same amount, once you are sure; on "
                        "any other call it doesn't count",
           "upgrade_road": "true only to repeat an upgrade_road refused for coming before the city's first bus line, "
                           "once you are sure"}
# the mayor's guarded money actions: their guard always runs, and confirm: true overrides a refusal only on a repeat of
# the exact call (action, department, amount) refused this turn, one override per refusal (ToolBox._refused). v19: all
# 3 take-back refusals were overridden at once and 3 of 7 duplicate refusals; confirm was a free pass
EARNED_CONFIRM = {"allocate_budget", "take_loan"}
EARNED_NOTE = " (confirm: true counts only when repeating this exact call after it was refused)"

# purpose -> path into build_observation(..., site_hints=True)
SITE_PURPOSES = {
    "mayor": {},
    "planning": {"zone_R": ("zoning_suggestions", "R"), "zone_C": ("zoning_suggestions", "C"),
                 "zone_I": ("zoning_suggestions", "I"), "zone_O": ("zoning_suggestions", "O"),
                 # no mixed-use ranking exists; shops below flats want what shops want (near homes and the centre)
                 "zone_M": ("zoning_suggestions", "C"),
                 "road_extension": ("road_suggestions",), "casino": ("casino_sites",)},
    "utilities": {"coal_or_gas": ("candidate_sites", "far_from_homes"), "wind": ("candidate_sites", "windy"),
                  "pump": ("candidate_sites", "riverbank_next_to_road"),
                  "sewage": ("candidate_sites", "riverbank_for_sewage"),
                  "near_homes": ("candidate_sites", "near_homes"),
                  "riverbank_2x2": ("candidate_sites", "riverbank_2x2")},
    "transport": {"bus_stops": ("transport", "suggested_stops"),
                  "train_station": ("transport", "train_station_sites"),
                  "harbour": ("transport", "harbour_sites"), "metro_station": ("transport", "metro_station_sites"),
                  "congested_roads": ("transport", "congested_roads")},
    "police": {"station": ("police", "candidate_sites"), "jail": ("police", "jail_sites")},
    "fire": {"station": ("fire", "candidate_sites"), "shelter": ("fire", "shelter_sites"),
             "levee": ("fire", "levee_lines")},
    "health": {"clinic": ("health", "clinic_sites"), "hospital": ("health", "hospital_sites"),
               "landfill": ("health", "landfill_sites"),
               # a cemetery serves the whole city (no radius), so it reuses the landfill list: roadside lots away
               # from homes, ranked by uncovered garbage nearby (which doesn't matter for a cemetery)
               "cemetery": ("health", "landfill_sites")},
    "education": {"school": ("education", "school_sites"), "high_school": ("education", "high_school_sites"),
                  "university": ("education", "university_sites"),
                  # a library speeds up schooling in the homes within its radius, so it reuses the school list:
                  # roadside lots near homes no school covers yet
                  "library": ("education", "school_sites")},
}
SITE_UNLOCKS = {"zone_O": ("office_zone",), "zone_M": ("mixed_use",), "casino": ("casino",),
                "riverbank_2x2": ("water_works", "advanced_treatment", "nuclear"), "train_station": ("train_station",),
                "harbour": ("harbour",), "metro_station": ("metro",), "jail": ("jail",), "hospital": ("hospital",),
                "high_school": ("high_school",), "university": ("university",), "library": ("library",)}
# purpose -> (short reason for the briefing, full note for find_sites) when its list is empty
SITE_NOTES = {"levee": ("no flood forecast", "no flood is forecast within 3 months; riverbank stretches to protect "
                        "appear with a flood warning"),
              "congested_roads": ("no road over capacity", "no road carries more than its capacity right now; an "
                                  "avenue pays only on a stretch at 1.0x or more"),
              "bus_stops": ("the first bus line is offered at 1,500 residents",
                            "no stops are suggested below 1,500 residents: a bus line costs more than it carries in a "
                            "smaller town; from 1,500 the first line is listed here, and later ones whenever transit "
                            "falls under 12% and a new line would carry more trips")}
# purpose -> the population its list waits for (empty below it, with its SITE_NOTES reason): observation's suggested
# stops have no floor, so from month 1 transport read "can't pay (cheapest: bus_stops $1,500)" and the mayor "needs
# $1,500 for a bus line", against both playbooks (no bus line before ~1,500 residents)
SITE_MIN_POP = {"bus_stops": 1500}
# bus lines, as the scripted transport builds them: another line only once every line carries BUS_FULL of its seats,
# and at most BUS_LINES_MAX lines. Riders pick the line with the best share (max over lines), so a second line on
# the same stops buys nothing: the 120-month LLM runs built lines at ~210 of 1,500 seats. Congested stretches are
# listed from CONGESTED_LOAD x capacity (the scripted transport upgrades only there; observation lists from 0.8)
BUS_FULL, BUS_LINES_MAX, CONGESTED_LOAD = 0.9, 4, 1.0
# ... or while transit is under TRANSIT_GOAL (transport_score counts the share up to 12%: scoring.transport_score) and
# the next line (new_line_stops: today's biggest job centres and the homes the lines don't reach) would add at least
# BUS_GAIN of the trips (transit_estimate: traffic's gravity model, read-only). A trip rides the best line covering both
# its ends, so a line on the same stops adds nothing, and a line through hubs that moved adds riders. v23 120-month
# runs: one 7-stop line from month 12-18 took transit to 15-21%, then the city outgrew it (4-6% by month 48, riders
# 124-196 of 1,500 seats) and the full-buses rule offered no second line. The rule after it (40% of residents beyond
# every stop) still missed hubs that moved: region run 20261005, north, month 66: 61% of residents near a stop,
# transit 9.6%, no line offered; 1-3% at month 108-120 in every 120-month LLM run (scripted fleet: 10-15%)
TRANSIT_GOAL, BUS_GAIN = 0.12, 0.01
# transport's stations (first_build names one, priced, when nothing else is ready). A train station or harbour links the
# city to the region: without one, R and C demand stay capped from CAP_POP_CONNECTION residents (World.demand_caps).
# Their lists offer one only when it pays (link_room): no link yet, a first train station (the scripted transport's
# rule), or freight at FREIGHT_FULL of its capacity or more (the scripted harbour's rule); else another adds only freight
# room nobody uses. v1 seed 3, month 96: "Next: a harbour ($10k)" with freight at 15% of a harbour's capacity
STATION_PURPOSES = ("train_station", "harbour", "metro_station")
LINK_PURPOSES = ("train_station", "harbour")
FREIGHT_FULL = 0.7
RIVERBANK_PURPOSES = {"pump", "sewage", "riverbank_2x2"}      # empty when no road reaches free riverbank
BANK_TOP_UP = {"pump": "pump", "sewage": "outlet"}   # one-tile riverbank purposes site_lists tops up -> crowd line
# utilities' first-building purposes -> the supply their building adds to (near_homes: its cheapest, the water tower)
# and, for the one-tile riverbank ones, their _bank_needed name: blocked or funded, one counts only while the city lacks
# it or demand reaches BANK_BUSY x that supply now or within AHEAD_MONTHS (_wanted, _busy)
SUPPLY_OF = {"pump": ("water", "pump"), "sewage": ("sewage", "sewage outlet"), "near_homes": ("water", None),
             "coal_or_gas": ("power", None), "wind": ("power", None)}
# utilities' supplies, most urgent first: a lot develops to level 1 only with power, to 2 with water, to 3 with sewage
# (economy.max_levels). Its needs line names one building per short supply (short_supplies) in this order
SUPPLIES = ("power", "water", "sewage")
# what a short supply wants added (supply_need): max(SUPPLY_MARGIN x demand, demand + SUPPLY_GROWTH) less the city's
# own output, and the building named is the cheapest ready one whose output covers that (pick_supply). Smoke v9-v19
# (22 runs): power demand grew a median 62 from month 0 to month 3 (46-192), water 50 (34-144; sewage demand is
# water's without the cooling), so a first plant must cover about that: a wind turbine's 40 (x its site's wind, 1.4
# at best: 56) doesn't, a coal plant's 250, a pump's 250, a tower's 120 and an outlet's 250 do. Later, half again the
# demand is inside the 10-60% margin the utilities playbook asks and a quarter's growth (x1.3-1.5 a quarter from
# month 6 to 12 in those runs). Smoke v20 seed 1: the line named only the cheapest ($2,200 tower), the mayor gave that,
# and power stayed at 0-18% until month 10
SUPPLY_MARGIN = 1.5
SUPPLY_GROWTH = {"power": 60, "water": 50, "sewage": 50}
BANK_BUSY = C.SUPPLY_BUSY
# ... now or within this many months at its growth over the last as many months (demand_ahead): a quarter, the next
# turn at act_every 3, so utilities builds (and asks) ahead of a supply that will run short before then
AHEAD_MONTHS = 3
JAIL_BUSY = 0.8              # a first jail is needed once inmates pass this share of the cells (_cells_busy)
RIVER_PURPOSES = RIVERBANK_PURPOSES | {"harbour", "levee"}    # purposes that belong on the river
SCARCE_BANK = 2              # at most this many free riverbank lots by the roads: other purposes leave them alone
# purposes whose lists skip lots with 3+ homes within 4 tiles (observation._far_from_homes): never empty for lack
# of need (the landfill ranking adds 0.01 everywhere, the jail's is flat), only for lack of such a lot
AWAY_FROM_HOMES = {"jail", "landfill", "cemetery"}
# purposes whose lists rank free 2x2 roadside blocks (observation: site_candidates(..., size=2)) with 0.01 need added
# everywhere: never empty for lack of need, only for lack of a block. Casino's too, but it is planning's (its own
# reason); riverbank_2x2 is a riverbank purpose
TWO_BY_TWO = {"hospital", "university"}
# (role, purpose) -> (what its building serves, the coverage grid, the list's radius): the lists observation ranks by
# uncovered need within that radius (people, buildings, residents or school-age residents x (1 - coverage), times a
# positive weight). Empty with real need, the free roadside tiles are all too far from it (_why_empty). A library
# reads the school list. Not the shelter: supplementary, and its coverage stays about 0 all game
NEAR_NEED = {("police", "station"): ("people", "police_cov", 6), ("fire", "station"): ("buildings", "fire_cov", 6),
             ("health", "clinic"): ("residents", "health_cov", 5),
             ("education", "school"): ("children", "school_cov", 6),
             ("education", "high_school"): ("teenagers", "high_cov", 8),
             ("education", "library"): ("children", "school_cov", 6)}
# real need: at least NEED_SHARE of what the building serves under NEED_COVERED coverage (the observation's
# uncovered_buildings test). Summing 1 - coverage flagged well-covered cities (coverage fades from 60% of the radius and
# with congestion): scripted month 120 read 11.5% for fire (score 87.4). Under 0.3, mature scripted cities (months
# 72-120, seeds 0-1) sit at police 5-17%, fire 0-5%, clinics 2-12%, schools 10-27%; month 60 had clinics at 12%
# (health 91.0) and fire at 10% (76.4), so 15%, not 10%. The 16 misread turns of v9-v14 were all over 50%
NEED_SHARE, NEED_COVERED = 0.15, 0.3
# the best site's most useful field for the briefing's Ready sites line, first match wins
SITE_FIELDS = [("wind", "wind {}"), ("homes_within_5", "{} homes within 5"), ("need", "need {}"),
               ("opens_land", "opens {} land tiles"), ("load_vs_capacity", "load {}x capacity"), ("tiles", "{} tiles"),
               ("pollution", "pollution {}")]
SITE_FIELDS_HIDDEN = {"sewage": {"pollution"}}    # the outfall's own water quality doesn't matter
SITE_FIELDS_NONZERO = {"near_homes"}              # "0 homes within 5" says nothing near homes (unlike coal_or_gas)
# purpose -> what its sites are for (tier names, or OFF_TIER_SPECS keys): the cheapest unlocked one prices the site
# (site_price). Planning's purposes aren't (its road_extension is priced apart, as no building: road_price)
SITE_BUILDS = {
    "utilities": {"coal_or_gas": ("coal", "gas"), "wind": ("wind",), "pump": ("pump",), "sewage": ("outlet",),
                  "near_homes": ("tower", "solar", "wind"),
                  "riverbank_2x2": ("water_works", "advanced_treatment", "nuclear")},
    "transport": {"bus_stops": ("bus_line",), "train_station": ("train_station",), "harbour": ("harbour",),
                  "metro_station": ("metro_station",), "congested_roads": ("avenue_tile",)},
    "police": {"station": ("police_station",), "jail": ("jail",)},
    "fire": {"station": ("fire_station",), "shelter": ("shelter",), "levee": ("levee_tile",)},
    "health": {"clinic": ("clinic",), "hospital": ("hospital",), "landfill": ("landfill",), "cemetery": ("cemetery",)},
    "education": {"school": ("elementary_school",), "high_school": ("high_school",), "university": ("university",),
                  "library": ("library",)},
}
# prices outside the tier lines -> (what the action charges, looked up when priced as the actions do, since a balance
# override may replace it; unlock key). A bus line: create_bus_line. Per tile (PER_TILE): build_levee charges each
# levee tile; upgrade_road each street tile it makes an avenue on land (the cheapest class upgrade; x3 on bridges)
OFF_TIER_SPECS = {"train_station": (lambda: C.TRAIN_STATION["cost"], "train_station"),
                  "harbour": (lambda: C.HARBOUR_SPEC["cost"], "harbour"),
                  "metro_station": (lambda: C.METRO_STATION["cost"], "metro"),
                  "bus_line": (lambda: C.BUS_LINE_COST, None),
                  "levee_tile": (lambda: C.CATALOG["levee"]["cost"], None),
                  "avenue_tile": (lambda: C.ROAD_UPGRADE_COST[C.ROAD_CLASSES["avenue"]], "avenue")}
# the same names -> their monthly upkeep before funding (upkeep_of), looked up when asked like the prices: a bus line
# its line's with the 2 stops create_bus_line needs at least (each stop adds BUS_STOP_UPKEEP: $60 alone understated a
# line by 27% or more), a levee tile its catalog entry's, an avenue tile what an avenue costs beyond the street it was
# (World.road_upgrade_upkeep; x2 on bridges)
OFF_TIER_UPKEEP = {"train_station": lambda: C.TRAIN_STATION["upkeep"], "harbour": lambda: C.HARBOUR_SPEC["upkeep"],
                   "metro_station": lambda: C.METRO_STATION["upkeep"],
                   "bus_line": lambda: C.BUS_LINE_UPKEEP + 2 * C.BUS_STOP_UPKEEP,
                   "levee_tile": lambda: C.CATALOG["levee"]["upkeep"],
                   "avenue_tile": lambda: C.ROAD_UPKEEP[C.ROAD_CLASSES["avenue"]] - C.ROAD_UPKEEP[1]}
PER_TILE = {"levee_tile", "avenue_tile"}
BRIDGE_UPGRADE = 3           # upgrade_road charges a bridge tile this many times a land tile's upgrade
# priced purposes that are never a department's first build: cheapest() skips them, so the mayor's needs line and the
# budget line name the primary price (their Ready lines keep theirs; affordable, they still keep the budget line
# away, which would be false). At $1,500 each they undercut the first build and were named instead. A library only
# multiplies school graduation (LIBRARY_BOOST): useless without the $2,500 school's seats. A shelter only matters in
# disasters (the playbook adds one past 3,000 residents): smoke v9's mayor read "needs $1,500 for a shelter", funded
# fire that much, and fire never reached the $2,500 station (0% coverage, fire score 31.4). A cemetery comes after
# the $2,000 landfill in the playbook. Train stations, harbours and metro stations are never transport's first build
# either: the playbook wants bus lines first (at 5,000+ residents, $3,000 read "can't pay (cheapest: harbour $10k)"
# with bus stops ready)
SUPPLEMENTARY = {"library", "shelter", "cemetery", "train_station", "harbour", "metro_station"}
# base-tier buildings that are never a first building (Price.first False: "a university", kind "another" on the
# mayor's lines, never a blocked or funded first building). civitas-v2 review, eval_v2 seeds 4, 5, 12, 18, 20 at month
# 96: with both school levels at 90% the needs line read "needs $12k for its first university", first/supply, so it
# could use the treasury down to $3,000 and passed the loan guard; a university counts in no enrolment the score reads
NOT_FIRST = {"university"}
# (dept, purpose) -> the World._metrics coverage the mayor's needs line shows for another building of a kind the city
# has (buildings_of); utilities show the share of their supply used, a kind without either its ready sites. Smoke
# v16: "needs $2,500 for a police station (5 ready sites)" read the same with a station built, and the mayor funded
# second stations while education never got its school
COVERAGE_OF = {("police", "station"): "police_coverage", ("fire", "station"): "fire_coverage",
               ("health", "clinic"): "health_coverage", ("health", "hospital"): "health_coverage",
               ("health", "landfill"): "garbage_coverage", ("education", "school"): "school_coverage",
               ("education", "high_school"): "high_coverage", ("transport", "bus_stops"): "transit_share"}
COVERAGE_LABEL = {"transit_share": "transit"}       # else "coverage"

# every role's actions, and the region's deal actions (a region mayor's tools: fleet/region.py)
ALL_ACTIONS = {a for spec in ROLES.values() for a in spec["actions"]} | set(DEAL_ACTIONS)
# a build refused for its tile, as World.footprint_tiles, build_train_station and build_harbour word it
NOT_EMPTY = re.compile(r"\((\d+),(\d+)\) is not an empty (land|river|riverbank) tile")
BUDGET_SHORT = re.compile(r"costs \$([\d,]+) but the \w+ budget only has \$([\d,]+)")    # actions._spend
# int in the doc example, but the handler reads a float (actions._num)
NUMBER_PARAMS = {("set_tax", "rate"), ("set_funding", "percent"), ("allocate_budget", "amount"),
                 ("take_loan", "amount"), ("repay_loan", "amount"), ("set_fare", "fare"), ("set_district_tax", "delta")}
_JSON_TYPE = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
# build actions -> the tier lines their "kind" can name
_BUILD_LINES = {"build_power_plant": POWER_LINES, "build_water": WATER_LINES, "build_sewage": SEWAGE_LINES,
                "build_school": tuple(SCHOOL_KINDS.values())}
# documented parameters the examples don't show
_EXTRA_PARAMS = {
    "build_park": {k: {"type": "integer"} for k in ("x1", "y1", "x2", "y2")},
    "enact_policy": {"district": {"type": "string", "description": "apply the policy in one district only"}},
    "repeal_policy": {"district": {"type": "string", "description": "repeal it in one district only"}},
    "upgrade_facility": {"tier": {"type": "integer", "description": 'target tier (with module "tier")'}},
}
_TIER_PARAM = {"type": "integer", "description": "tier within the building's line (1 = base); see the description"}


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    properties: dict = field(default_factory=dict)
    required: tuple = ()
    strict: bool = True           # reject unknown arguments (action tools pass them to the handler)
    newest_last: bool = False     # trimming keeps the tail of long lists (logs, messages)
    lists: tuple = ()             # string arguments that also take a JSON list, joined with commas (send_message's to)
    trim_hint: str = ""           # how to ask for less when a result is trimmed

    def schema(self) -> dict:
        params = {"type": "object", "properties": self.properties}
        if self.required:
            params["required"] = list(self.required)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}


# ------------------------------------------------------------------ helpers
def _py(v):
    """JSON-safe copy: numpy scalars to Python, floats to 2 dp, tuples to lists, keys to str."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 2) if math.isfinite(v) else None
    return v


def _size(v) -> int:
    return len(json.dumps(v))


def _fit(res: dict, limit: int = RESULT_CHARS, newest_last: bool = False, hint: str = "") -> dict:
    """Drop items from the biggest lists until the JSON, with its "truncated" note counting them, fits (from the
    end, or from the front for newest-last lists)."""
    note = lambda n: f"{n} {'older' if newest_last else 'more'} list items left out" + (f"; {hint}" if hint else "")
    dropped = 0
    while _size({**res, "truncated": note(dropped)} if dropped else res) > limit:
        lists = []

        def walk(v):
            if isinstance(v, list) and v:
                lists.append(v)
            for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()):
                walk(x)
        walk(res)
        if not lists:
            break
        big = max(lists, key=_size)
        n = max(1, len(big) // 10)
        if newest_last:
            del big[:n]
        else:
            del big[-n:]
        dropped += n
    if dropped:
        res["truncated"] = note(dropped)
    return res


def _coerce(name: str, v, spec: dict):
    t = spec.get("type")
    bad = lambda what: ToolError(f"argument '{name}' must be {what}, got {v!r}")
    if t in ("integer", "number"):
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                raise bad("an integer" if t == "integer" else "a number")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise bad("an integer" if t == "integer" else "a number")
        v = int(v) if float(v).is_integer() else v       # fractions pass: handlers round where they need to
    elif t == "string":
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise bad("a string")
        v = str(v)
    elif t == "boolean":
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        if not isinstance(v, bool):
            raise bad("true or false")
    elif t in ("array", "object"):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        if not isinstance(v, list if t == "array" else dict):
            raise bad("a JSON array" if t == "array" else "a JSON object")
    if "enum" in spec and v not in spec["enum"]:
        raise ToolError(f"argument '{name}' must be one of {', '.join(map(str, spec['enum']))}; got {v!r}")
    return v


def to_schema(targets: list[str]) -> dict:
    """send_message's `to`: a string naming one address or several separated by commas (a JSON list is joined:
    Tool.lists). An enum would refuse the several."""
    return {"type": "string", "description": f"one of {', '.join(targets)}; or several roles separated by commas "
                                             "(\"police,fire\"), each getting its own message"}


def recipients(to: str, targets: list[str]) -> list[str]:
    """The addresses `to` names (send_message): one, or several separated by commas, each one of `targets` (case
    aside), repeats dropped; "all" only alone. ToolError otherwise."""
    lower = {t.lower(): t for t in targets}
    names = list(dict.fromkeys(p.strip() for p in str(to).split(",") if p.strip()))
    bad = [n for n in names if n.lower() not in lower]
    if not names or bad:
        raise ToolError(f"argument 'to' must be one of {', '.join(targets)}, or several of them separated by commas; "
                        f"got {to!r}")
    names = list(dict.fromkeys(lower[n.lower()] for n in names))
    if "all" in names and len(names) > 1:
        raise ToolError("'all' already reaches every role: send to 'all' alone, or list the roles")
    return names


def _items_schema(example: list) -> dict:
    """Item schema from a doc example, e.g. stops [[5, 24], ...] -> arrays of integers (models sent {"x", "y"}
    objects to an untyped list)."""
    if not example:
        return {}
    first = example[0]
    if isinstance(first, list):
        return {"type": "array", "items": _items_schema(first) or {}}
    return {"type": _JSON_TYPE[type(first)]}


def _action_tool(name: str, doc: tuple[str, str], fn: Callable, confirm: bool = False) -> Tool:
    """The tool for action `name`, with a confirm parameter when the action is guarded for the role (guarded())."""
    example, text = doc
    props = {}
    for k, v in json.loads(example).items():
        if k != "type":
            t = "number" if (name, k) in NUMBER_PARAMS else _JSON_TYPE[type(v)]
            props[k] = {"type": t, **({"items": _items_schema(v)} if isinstance(v, list) else {})}
    lines = _BUILD_LINES.get(name) or ((C.TIER_BY_NAME[name[6:]][0],) if name[6:] in C.TIER_BY_NAME else ())
    if name.startswith("build_") and any(len(C.TIER_LINES[ln]) > 1 for ln in lines):
        props["tier"] = dict(_TIER_PARAM)
    props.update(_EXTRA_PARAMS.get(name, {}))
    if confirm:
        props["confirm"] = {"type": "boolean", "description": CONFIRM[name]}
    return Tool(name, text, fn, props, strict=False)


def _site(s) -> dict:
    s = {"x": s[0], "y": s[1]} if isinstance(s, (list, tuple)) else dict(s)
    if "x" not in s:                           # a rectangle or line: its first corner
        s = {"x": s["x1"], "y": s["y1"], **s}
    s.pop("network", None)
    return s


def _on_connected_road(w, s) -> bool:
    """False for a hint that names a road network off the highway (the riverbank lists take any network): a
    building there would get no power, water or customers."""
    net = s.get("network") if isinstance(s, dict) else None
    return net is None or bool(w.networks[net].outside_connected)


def _distance(s: dict, near: tuple[int, int]) -> int:
    """From `near` to the nearest tile of a site, rectangle or line (a road extension or branch runs either way)."""
    nx, ny = near
    if "x2" not in s:
        return abs(s["x"] - nx) + abs(s["y"] - ny)
    (xa, xb), (ya, yb) = sorted((s["x1"], s["x2"])), sorted((s["y1"], s["y2"]))
    return abs(min(max(nx, xa), xb) - nx) + abs(min(max(ny, ya), yb) - ny)


def _connected_roads(w) -> np.ndarray:
    """Road tiles on networks connected to the highway."""
    labels = [i for i, n in enumerate(w.networks) if n.outside_connected]
    return np.isin(w.road_labels, labels) if labels else np.zeros(w.struct.shape, bool)


def _outside_labels(w, labels: np.ndarray, drop=frozenset()) -> set[int]:
    """The road components linked to the region, as World.refresh() finds them: those holding a highway tile, and
    those next to (8-neighbourhood) a train station, harbour or airport (LINKS) not anchored at a tile in `drop`."""
    out = {int(labels[y, x]) for y, x in zip(*np.nonzero(w.struct == C.HIGHWAY))} - {-1}
    for x, y in (a for st in LINKS for a in w.facilities(st) if a not in drop):
        for px, py in w.parts(x, y):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if w.in_bounds(px + dx, py + dy) and labels[py + dy, px + dx] >= 0:
                        out.add(int(labels[py + dy, px + dx]))
    return out


def cut_off(w, tiles, drop=frozenset()) -> np.ndarray:
    """The highway-connected road tiles (the removed ones excluded) that removing the road tiles among `tiles`, and
    the link buildings anchored at the tiles in `drop`, would cut off the region: the road network relabelled
    without them as World.refresh() labels it (4-neighbour components, linked by a highway tile or a station /
    harbour / airport)."""
    before = _connected_roads(w)
    gone = np.zeros(before.shape, bool)
    for x, y in tiles:
        gone[y, x] = w.struct[y, x] == C.ROAD
    if not (gone & before).any() and not drop:
        return np.zeros_like(before)
    labels, _ = label_components(((w.struct == C.ROAD) | (w.struct == C.HIGHWAY)) & ~gone)
    return before & ~gone & ~np.isin(labels, list(_outside_labels(w, labels, drop)))


def upgrade_cost(w, p: dict) -> float | None:
    """What upgrade_road charges for these arguments (actions.upgrade_road's rule: the class difference per tile
    below the target, BRIDGE_UPGRADE times on a bridge); None when it would fail anyway (bad class or line, locked,
    off the road, nothing to upgrade)."""
    to = str(p.get("to", p.get("kind", "avenue"))).lower()
    if to not in ("avenue", "highway") or not w.unlocked(to):
        return None
    cls = C.ROAD_CLASSES[to]
    try:
        pts = _line(w, p)
    except Exception:
        return None
    cost = 0.0
    for x, y in pts:
        if w.struct[y, x] != C.ROAD:
            return None
        cur = int(w.road_class[y, x])
        if cur < cls:
            per = C.ROAD_UPGRADE_COST[cls] - (C.ROAD_UPGRADE_COST.get(cur, 0) if cur > 1 else 0)
            cost += per * (BRIDGE_UPGRADE if w.terrain[y, x] == C.WATER else 1)
    return cost or None


def guarded(role: str, action: str) -> bool:
    """Whether `role`'s `action` can be refused before it runs (its schema then has a confirm parameter): a bulldoze,
    and a demolish by a role that can take a train station, harbour or airport (transport; LINK_OWNERS), that would
    cut roads off (confirm: true lets it run); the mayor's allocate_budget identical to one that succeeded this turn,
    taking a department below what it keeps (DeptMoney.keep), or topping up another building under the reserve, and
    its take_loan while the treasury holds RESERVE_MIN, its own top-ups this turn took it under, or no first
    building, supply or planning road waits (EARNED_CONFIRM: confirm: true counts only on a repeat of the refused
    call); transport's upgrade_road that would leave it short of the city's first bus line while one is ready
    (confirm: true lets it run)."""
    return (action in ROAD_REMOVERS or (action == "demolish" and role in LINK_OWNERS)
            or (role == "mayor" and action in EARNED_CONFIRM) or (role == "transport" and action == "upgrade_road"))


def _allocation(p: dict) -> tuple[str, float] | None:
    """An allocate_budget's department and amount as actions.allocate_budget reads them; None if it can't."""
    try:
        return str(p.get("department", "")).lower(), float(p["amount"])
    except (KeyError, TypeError, ValueError):
        return None


def _money_key(name: str, p: dict) -> tuple:
    """(action, department, amount) of an EARNED_CONFIRM call as its handler reads them: what an earned confirm must
    repeat (ToolBox._refused). The department is None for a loan; both are None when they can't be read (the guards
    then refuse nothing: the action reports it)."""
    if name == "allocate_budget":
        return (name, *(_allocation(p) or (None, None)))
    try:
        return name, None, float(p["amount"])
    except (KeyError, TypeError, ValueError):
        return name, None, None


def _touching(mask: np.ndarray, diagonal: bool = True) -> np.ndarray:
    """Tiles with a neighbour in `mask`: 8-neighbourhood (as `adjacent_network` and the build actions check road
    access), or 4 (as `is_riverbank` checks water)."""
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dx or dy) and (diagonal or not (dx and dy)):
                out |= shift(mask, dy, dx, False)
    return out


def _empty_land(w) -> np.ndarray:
    return (w.struct == C.NONE) & (w.terrain == C.LAND)


def _roadside(w) -> np.ndarray:
    """Empty land tiles touching a highway-connected road: where a new building could go."""
    return _touching(_connected_roads(w)) & _empty_land(w)


def _free_roadside(w) -> int:
    return int(_roadside(w).sum())


def _riverbank_by_roads(w) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Riverbank land tiles (next to water, as the build actions check `is_riverbank`) that touch a
    highway-connected road, road tiles excluded: the lots a pump or outlet could use. And where those roads meet
    the river: bridgeheads, else their tiles on the bank (roads along it), else the bank lots beside them (a road
    passing the water diagonally); empty when no connected road reaches the river."""
    road, water = _connected_roads(w), w.terrain == C.WATER
    bank = (w.terrain == C.LAND) & _touching(water, diagonal=False)
    lots = bank & _touching(road) & (w.struct != C.ROAD) & (w.struct != C.HIGHWAY)
    heads = road & bank & _touching(road & water, diagonal=False)
    contact = next((m for m in (heads, road & bank) if m.any()), lots)
    ys, xs = np.nonzero(contact)
    return lots, [(int(x), int(y)) for y, x in zip(ys, xs)]


def free_bank(w) -> np.ndarray:
    """The free riverbank lots beside highway-connected roads: where a pump or outlet can go."""
    return _riverbank_by_roads(w)[0] & _empty_land(w)


def _bank_needed(w) -> list[str]:
    """What the city still lacks that takes a riverbank lot: a pump (neither a pump nor a water tower yet; a tower
    supplies water without the bank) and a sewage outlet (neither an outlet nor a treatment plant)."""
    return [name for name, structs in (("pump", (C.PUMP, C.TOWER)), ("sewage outlet", (C.OUTLET, C.TREATMENT)))
            if not any(w.facilities(s) for s in structs)]


def _ends(points: list) -> list:
    """The first and last point (reading order): a bridge's two ends, a riverside road's two ends."""
    return points[:1] + points[-1:] if len(points) > 1 else points


def _box(mask) -> tuple[list[int], list[int]] | None:
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else ([int(xs.min()), int(xs.max())], [int(ys.min()), int(ys.max())])


def _span(a: int, b: int) -> str:
    return str(a) if a == b else f"{a}-{b}"


def _some(items: list[str], n: int, sep: str = ", ") -> str:
    """The first n items, and how many more there are."""
    return sep.join(items[:n]) + (f" +{len(items) - n} more" if len(items) > n else "")


def _roads(w) -> dict | None:
    """Extent and size of the highway-connected road network."""
    roads = _connected_roads(w)
    box = _box(roads)
    return box and {"x": box[0], "y": box[1], "tiles": int(roads.sum())}


def _roads_text(r: dict | None) -> str:
    return (f"highway-connected roads: x {_span(*r['x'])}, y {_span(*r['y'])}, {r['tiles']} tiles" if r else
            "no roads connected to the highway yet")


def _road_lines(w) -> list[dict]:
    """The maximal straight runs of highway-connected road, along rows ({"y", "x": [x0, x1]}) and down columns
    ({"x", "y": [y0, y1]}), of at least ROAD_LINE_MIN tiles: the highway tile, bridges and avenues count, and a
    crossing is in both its lines. Longest first, then by the first tile's y and x, rows before columns. Smoke
    v9-v11: given only the network's box (x 0-25, y 19-29; really y=19, y=24 and x=19), planning rebuilt existing
    road 13 times and made 5.4 inspect calls a turn."""
    roads = _connected_roads(w)
    runs = []
    for down, grid in ((False, roads), (True, roads.T)):
        for i, row in enumerate(grid):
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
            for a, b in zip(edges[::2].tolist(), (edges[1::2] - 1).tolist()):      # each run's first and last tile
                if b - a + 1 >= ROAD_LINE_MIN:
                    y, x = (a, i) if down else (i, a)
                    runs.append(((a - b, y, x, down), {"x": i, "y": [a, b]} if down else {"y": i, "x": [a, b]}))
    return [ln for _, ln in sorted(runs, key=lambda r: r[0])]


def _road_line_text(ln: dict) -> str:
    return f"y={ln['y']} x {_span(*ln['x'])}" if isinstance(ln["x"], list) else f"x={ln['x']} y {_span(*ln['y'])}"


def _on_road_line(ln: dict, x: int, y: int) -> bool:
    return (y == ln["y"] and ln["x"][0] <= x <= ln["x"][1]) if isinstance(ln["x"], list) else (
        x == ln["x"] and ln["y"][0] <= y <= ln["y"][1])


class Leg(NamedTuple):
    """One straight leg of a build_road line: along a row (axis "y": `at` = y, x from `lo` to `hi`), down a column
    (axis "x": `at` = x), or a single tile (axis None, along a row)."""
    axis: str | None
    at: int
    lo: int
    hi: int

    @property
    def tiles(self) -> set[tuple[int, int]]:
        return {(self.at, t) if self.axis == "x" else (t, self.at) for t in range(self.lo, self.hi + 1)}

    def along(self, other: Leg) -> bool:
        """On the same row or column as `other` and sharing a tile with it (a single tile: any leg through it)."""
        return (self.axis == other.axis or None in (self.axis, other.axis)) and bool(self.tiles & other.tiles)

    def text(self) -> str:
        if self.axis is None:
            return f"({self.lo},{self.at})"
        return _road_line_text({"x": self.at, "y": [self.lo, self.hi]} if self.axis == "x" else
                               {"y": self.at, "x": [self.lo, self.hi]})


def _legs(w, args: dict) -> list[Leg]:
    """The legs of the line build_road draws for `args` (_line: an L runs along its row first, then down x2; both
    legs hold the corner)."""
    (x1, y1), (x2, y2) = _point(w, args, "x1", "y1"), _point(w, args, "x2", "y2")
    if (x1, y1) == (x2, y2):
        return [Leg(None, y1, x1, x1)]
    return ([Leg("y", y1, min(x1, x2), max(x1, x2))] if x1 != x2 else []) + (
        [Leg("x", x2, min(y1, y2), max(y1, y2))] if y1 != y2 else [])


def _and(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def _contents(w, tiles: list[tuple[int, int]]) -> str:
    """What the tiles hold by kind, largest first, the first CONTENT_KINDS kinds then "+N other tiles": road (bridges
    too), highway, zoned <letter> (with how many are developed), river, empty land, and any other structure by name
    (tiles, so a 2x2 hospital is "4 hospital"). "all 14 tiles are road", "10 zoned R (6 developed), 4 road"."""
    n, dev = Counter(), Counter()
    for x, y in tiles:
        s = int(w.struct[y, x])
        k = (f"zoned {C.LETTER_BY_ZONE[s]}" if s in C.ZONES else C.STRUCT_NAME[s].replace("_", " ") if s != C.NONE
             else "river" if w.terrain[y, x] == C.WATER else "empty land")
        n[k] += 1
        dev[k] += bool(s in C.ZONES and w.level[y, x] > 0)
    name = lambda k: k + (f" ({dev[k]} developed)" if dev[k] else "")
    if len(n) == 1:
        (k, c), = n.items()
        bare = k in ("road", "highway", "river", "empty land") or k.startswith("zoned ")
        return f"all {c} tiles are {name(k)}" if c > 1 else f"the tile is {name(k) if bare else a_or_an(k)}"
    kinds = n.most_common()
    rest = sum(c for _, c in kinds[CONTENT_KINDS:])
    return ", ".join(f"{c} {name(k)}" for k, c in kinds[:CONTENT_KINDS]) + (
        f" +{rest} other tile{'s' * (rest > 1)}" if rest else "")


def _zone_beside(w, tiles: list[tuple[int, int]], spare=()) -> list[str]:
    """Zone rects for the free roadside land (empty land touching a highway-connected road) on the lines just outside
    a rectangle's sides (`tiles` in reading order, as _rect gives them), over its span: each trimmed to its first and
    last such tile (one line, so always a valid zone), kept with 2 such tiles or half its side (a 1-tile side only
    beside a 1-tile rectangle, so a row's end caps aren't offered as single lots), the most tiles first (ties: north,
    west, east, south). While a pump or outlet is missing (_bank_needed) the free riverbank lots are left out, and a
    side whose rect would still cover one is dropped (v7: zoning took the last bank lots and utilities had nowhere to
    put them); so are the `spare` tiles (funded_spots)."""
    (xa, ya), (xb, yb) = tiles[0], tiles[-1]
    free, sides = _roadside(w), []
    bank = free_bank(w) if _bank_needed(w) else np.zeros_like(free)
    for x, y in spare:
        bank[y, x] = True
    free &= ~bank
    for side in ([(x, ya - 1) for x in range(xa, xb + 1)], [(xa - 1, y) for y in range(ya, yb + 1)],
                 [(xb + 1, y) for y in range(ya, yb + 1)], [(x, yb + 1) for x in range(xa, xb + 1)]):
        hits = [(x, y) for x, y in side if w.in_bounds(x, y) and free[y, x]]
        if hits and (len(hits) >= 2 or 2 * len(hits) >= len(side) > 1 or len(tiles) == 1):
            (x1, y1), (x2, y2) = hits[0], hits[-1]                 # along the side, so x1 <= x2 and y1 <= y2
            if not bank[y1:y2 + 1, x1:x2 + 1].any():
                sides.append((-len(hits), f"x1={x1},y1={y1},x2={x2},y2={y2}"))
    return [r for _, r in sorted(sides, key=lambda s: s[0])[:ZONES_BESIDE]]


def zone_fit_rect(w, args: dict, funds: float,
                  avoid=frozenset()) -> tuple[tuple[int, int, int, int], int, float] | None:
    """The sub-rectangle of a zone's area (`args` as sent) with the most tiles `funds` pay for, as actions.zone charges
    them (then the most with road access, then the smallest), covering none of the `avoid` tiles: ((x1, y1, x2, y2),
    tiles, cost); None when the funds don't pay for one tile. zone_fit names it; planning's Next line sizes its zone
    call with it (avoiding the lots funded departments and utilities are about to build on)."""
    letter = str(args.get("zone", "")).upper()[:1]
    density = zone_density(letter, str(args.get("density", "low")).lower())
    rect = _rect(w, args)
    (xa, ya), (xb, yb) = rect[0], rect[-1]
    tiles, rezone = zone_tiles(w, letter, density, rect)
    grids = [np.zeros((yb - ya + 1, xb - xa + 1), int) for _ in range(4)]    # new, re-zoned, road access, avoid
    for x, y in tiles:
        grids[0][y - ya, x - xa] = 1
        grids[2][y - ya, x - xa] = int(bool(w.access[y, x]))
    for x, y in rezone:
        grids[1][y - ya, x - xa] = 1
    for x, y in avoid:
        if xa <= x <= xb and ya <= y <= yb:
            grids[3][y - ya, x - xa] = 1
    sums = [np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0))).tolist() for g in grids]
    box = lambda t, a, b, c, d: t[d + 1][c + 1] - t[b][c + 1] - t[d + 1][a] + t[b][a]
    H, W = grids[0].shape
    best = None
    for y1 in range(H):
        for y2 in range(y1, H):
            for x1 in range(W):
                for x2 in range(x1, W):
                    n, r, a, bad = (box(t, x1, y1, x2, y2) for t in sums)
                    cost = zone_cost(n, r, density)
                    if n + r == 0 or bad or short_of(funds, cost):
                        continue
                    key = (n + r, a, -(x2 - x1 + 1) * (y2 - y1 + 1))
                    if best is None or key > best[0]:
                        best = (key, (x1 + xa, y1 + ya, x2 + xa, y2 + ya), n + r, cost)
    return None if best is None else best[1:]


def zone_fit(w, args: dict, funds: float, ask: bool = True) -> str:
    """For a zone its department's funds can't pay (`args` as sent): the price per tile it was charged, as actions.zone
    charges it (zone_density: industry and mixed use at the high-density price; REZONE_COST per zoned tile changing
    density), how many new tiles `funds` buy, and the sub-rectangle of the requested area with the most tiles `funds`
    pay for (zone_fit_rect); for R/C/O at high density, whether "low" zones it all;
    then to ask the mayor for the rest, or with `ask` False (mayor_ask) to say so in the end_turn report instead.
    "" when the funds don't pay for one tile, or on any error. v19 (120 months, seeds 0-1): 43 zones failed on "costs
    $48 but the planning budget only has $46; ask the mayor for funds", 41 of them with money for some tiles, and
    planning asked the mayor instead of zoning less; industry zoned "low" was charged high density unseen. v23 (120
    months) seed 0: with "zone fewer tiles rather than waiting for funds" planning asked the mayor 18 times (v19: 37),
    spent $7.8k (v19: $13.3k) and zoned 40% less under full residential demand, so the hint asks for the rest too."""
    try:
        letter = str(args.get("zone", "")).upper()[:1]
        density = zone_density(letter, str(args.get("density", "low")).lower())
        rect = _rect(w, args)
        tiles, rezone = zone_tiles(w, letter, density, rect)
        price = zone_price(density)
        best = zone_fit_rect(w, args, funds)
        if best is None:
            return ""
        why = (" (industry and mixed use are always charged the high-density price)" if letter in NO_DENSITY
               else f" ({density} density)")
        buys = int((funds + 1e-9) // price)
        (x1, y1, x2, y2), n, cost = best
        out = (f" - zone {letter} costs {money(price)}/tile{why}"
               + (f", {money(REZONE_COST)} per zoned tile changing density" if rezone else "")
               + f"; {money(math.floor(funds + 1e-9))} pays for {buys} new tile{'s' * (buys != 1)}: zone "
               f"x1={x1},y1={y1},x2={x2},y2={y2} ({n} tile{'s' * (n != 1)}, {money(cost)}) now")
        if letter not in NO_DENSITY and density == "high":
            low_tiles, low_re = zone_tiles(w, letter, "low", rect)
            low = zone_cost(len(low_tiles), len(low_re), "low")
            if not short_of(funds, low):
                out += f', or zone it all with density "low" for {money(low)}'
        full = zone_cost(len(tiles), len(rezone), density)
        rest = money(math.ceil(full - funds))
        if full <= funds:
            return out
        return out + (f"; then ask the mayor for {rest} for the rest (send_message)" if ask   # mayor_ask: review
                      else f"; the rest ({rest}) waits: say so in your end_turn report")
    except Exception:
        return ""


def _river_cols(w) -> tuple[int, int, int, int] | None:
    """(x0, x1, y0, y1): the river's columns within the rows the connected roads span (the whole map before
    there are roads, or when the river misses those rows); None without a river."""
    road_rows = np.nonzero(_connected_roads(w).any(axis=1))[0]
    y0, y1 = (int(road_rows.min()), int(road_rows.max())) if len(road_rows) else (0, w.h - 1)
    cols = np.nonzero((w.terrain[y0:y1 + 1] == C.WATER).any(axis=0))[0]
    if not len(cols):
        cols = np.nonzero((w.terrain == C.WATER).any(axis=0))[0]
        y0, y1 = 0, w.h - 1
    return (int(cols.min()), int(cols.max()), y0, y1) if len(cols) else None


def _river_span(w) -> str | None:
    """Where the river runs, within the rows the connected roads span; None without a river."""
    r = _river_cols(w)
    return r and f"the river runs at x {r[0]}-{r[1]} in rows {r[2]}-{r[3]}"


def _edge(w, x: int, y: int) -> str:
    return "west" if x == 0 else "east" if x == w.w - 1 else "north" if y == 0 else "south" if y == w.h - 1 else "inner"


def map_facts(w) -> dict:
    """Orientation for the briefing's Map block and map_overview: size, highway entries, the river per row band
    (RIVER_BANDS; each [[x0, x1, first row], [x0, x1, last row]], the columns at the band's ends), the
    highway-connected roads' extent and straight lines (_road_lines; map_lines names ROAD_LINES), the road tiles off
    the highway network (None when there are none) and the built-up box."""
    hw = [{"x": int(x), "y": int(y), "edge": _edge(w, x, y)} for y, x in zip(*np.nonzero(w.struct == C.HIGHWAY))]
    water = w.terrain == C.WATER
    rows = np.nonzero(water.any(axis=1))[0]
    at = lambda y: [int(np.nonzero(water[y])[0].min()), int(np.nonzero(water[y])[0].max()), int(y)]
    river = [[at(band[0]), at(band[-1])] for band in np.array_split(rows, min(RIVER_BANDS, len(rows)))
             if len(band)] if len(rows) else []
    zoned = np.isin(w.struct, list(C.ZONES))
    built = (w.struct != C.NONE) & (w.struct != C.ROAD) & (w.struct != C.HIGHWAY)
    flat = np.arange(w.h * w.w).reshape(w.h, w.w)
    anchors = built & ~zoned & ((w.anchor < 0) | (w.anchor == flat))       # multi-tile buildings count once
    bb = _box(built)
    loose = (w.struct == C.ROAD) & ~_connected_roads(w)
    lb = _box(loose)
    return {"size": [w.w, w.h], "highway": hw, "river": river, "roads": _roads(w), "road_lines": _road_lines(w),
            "unconnected": lb and {"x": lb[0], "y": lb[1], "tiles": int(loose.sum())},
            "built": bb and {"x": bb[0], "y": bb[1], "zoned": int(zoned.sum()),
                             "developed": int((zoned & (w.level > 0)).sum()), "buildings": int(anchors.sum())}}


MAP_PARTS = ("size", "highway", "river", "roads", "road_lines", "unconnected", "built")
# the mayor's Map block: how far the roads reach, roads cut off, the built-up area. Size, highway entry, river and
# road lines made room for the departments' spending, idle and blocked money: with them the worst case (2 departments
# idle, 5 blocked) took 7,788 of the 7,450 characters test_briefings_stay_bounded_at_the_limits allows; without, 7,420
MAYOR_MAP = ("roads", "unconnected", "built")


def map_lines(f: dict, parts: tuple = MAP_PARTS, road_lines: int = ROAD_LINES) -> list[str]:
    """The map_facts lines, only `parts` (MAP_PARTS keys, in MAP_PARTS order), naming the longest `road_lines` road
    lines (none: no road lines line)."""
    W, H = f["size"]
    end = lambda e: f"x {_span(e[0], e[1])} at row {e[2]}"
    b = f["built"]
    lines = {
        "size": [f"- {W}x{H} tiles: x = column 0-{W - 1} (0 = west), y = row 0-{H - 1} (0 = north)"],
        "highway": ["- highway entry: " + (", ".join(f"({h['x']},{h['y']}) on the {h['edge']} edge"
                                                     for h in f["highway"]) or "none")],
        "river": ["- river: " + ", ".join(end(t) + (f" to {end(e)}" if e != t else "") for t, e in f["river"])
                  if f["river"] else "- no river"],
        "roads": ["- " + _roads_text(f["roads"])],
        "road_lines": ["- road lines: " + _some([_road_line_text(ln) for ln in lns], road_lines, "; ")]
                      if (lns := f.get("road_lines")) and road_lines > 0 else [],
        "unconnected": [f"- roads not connected to the highway: {u['tiles']} tiles (x {_span(*u['x'])}, y "
                        f"{_span(*u['y'])}) - reconnect them"] if (u := f.get("unconnected")) else [],
        "built": [f"- built-up area: x {_span(*b['x'])}, y {_span(*b['y'])} ({b['zoned']} zoned tiles, "
                  f"{b['developed']} developed, {b['buildings']} building{'s' * (b['buildings'] != 1)})"
                  if b else "- built-up area: none yet"]}
    return [ln for p in MAP_PARTS if p in parts for ln in lines[p]]


TIER_NOTE = " (higher tiers keep the letter)."


def _legend(chars: set) -> str:
    """MAP_LEGEND cut to the characters on the map (the tier note moves to map_overview's "read")."""
    items = MAP_LEGEND.removeprefix("Legend: ").replace(TIER_NOTE, "").split("  ")
    return "Legend: " + "  ".join(i for i in items if set(i.split(" ", 1)[0].split("/")) & chars)


def _ascii(w, k: int) -> list[str]:
    """world.ascii_map() rows (column rulers, row numbers), or one character per k x k block: the highway entry,
    else its most common feature (ties: the first in reading order), river only where the block has nothing else."""
    if k == 1:
        return w.ascii_map().split("\n")
    xs, ys = range(0, w.w, k), range(0, w.h, k)
    rows = ["    " + "".join(str((x // 10) % 10) for x in xs), "    " + "".join(str(x % 10) for x in xs)]
    for y in ys:
        line = []
        for x in xs:
            block = [w.tile_char(tx, ty) for ty in range(y, min(y + k, w.h)) for tx in range(x, min(x + k, w.w))]
            main = ([c for c in block if c == "H"] or [c for c in block if c not in ".~"]
                    or [c for c in block if c == "~"] or ["."])
            line.append(Counter(main).most_common(1)[0][0])
        rows.append(f"{y:>3} " + "".join(line))
    return rows


def map_overview(w, limit: int = RESULT_CHARS, parts: tuple = MAP_PARTS) -> dict:
    """The whole map as ASCII rows with the map_facts summary, downsampled until the result fits `limit`. At full
    size the summary goes first. The mayor's shrinks to what its Map block (`parts`: MAYOR_MAP) leaves out (size,
    highway, river and road lines), a department's (its Map block has it all) stays whole; then the road lines go,
    the longest kept while they fit, and a department's summary last gives way to a pointer to the Map block.
    Scripted seed 0: from month 66 the map, legend (growing with the city) and summary took 3,803 of the 3,800
    characters (3,894 at month 120), and a department's map_overview went straight to the pointer; now it keeps 7
    of the 12 road lines at month 72 and none at month 120 (3,768)."""
    facts = map_facts(w)
    left = tuple(p for p in MAP_PARTS if p not in parts)
    lines = lambda ps, n=ROAD_LINES: [ln[2:] for ln in map_lines(facts, ps, n)]
    note = ["the rest: see the Map block in your briefing"] if left else []
    steps = [lines(MAP_PARTS)] + [lines(left or MAP_PARTS, n) + note for n in range(ROAD_LINES, -1, -1)]
    if not left:
        steps.append(["see the Map block in your briefing"])
    steps = [list(s) for s in dict.fromkeys(map(tuple, steps))]     # cuts that still name every road line are no step
    read = "top two rows: x (tens, units); rows start with y; higher tiers keep the letter"
    for k in range(1, max(w.w, w.h) + 1):
        grid = _ascii(w, k)
        body = {c for row in grid[2:] for c in row[4:]}
        for summary in steps if k == 1 else steps[:1]:
            res = {"ok": True, "summary": summary, "legend": _legend(body), "read": read, "map": grid}
            if k > 1:
                res["scale"] = f"each character is a {k}x{k} block, shown at its top-left x, y; inspect for detail"
            if _size(res) <= limit:
                return res
    return res


def _plant(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in ("line", "neighbours")}


def overview(env) -> dict:
    """Headline metrics, finances and scores, with the change over the last 3 months (`trend`, None early on). The
    treasury's change over that window is split (`operations_3_months`, cash_flow: treasury_change = operations -
    top_ups + loans + other; None early on), and last month's costs by kind are listed dearest first
    (`costs_last_month`, $0 ones left out)."""
    w = env.world
    m = w.metrics()
    last = w.history[-1] if w.history else m
    pct = lambda v: None if v is None else round(100 * v, 1)
    out = {"ok": True, "month": w.month, "months_left": env.cfg.months - w.month,
           "population": m["population"], "happiness_pct": pct(m["happiness"]),
           "approval_pct": pct(m["approval"]), "unemployment_pct": pct(m["unemployment"]),
           "jobs": {"commercial": m["jobs_com"], "industrial": m["jobs_ind"], "offices": m["jobs_off"]},
           "treasury": math.floor(w.treasury), "debt": round(w.debt),
           "last_month": {"income": round(last["income"]), "expenses": round(last["expenses"])},
           "department_funds": {k: round(v) for k, v in w.dept_funds.items()},
           "tax_pct": {k: round(100 * v, 1) for k, v in w.taxes.items()},
           "demand": {k: round(v, 2) for k, v in (w.demand or {}).items()},
           **{k: {"supply": m[f"{k}_supply"], "demand": m[f"{k}_demand"], "coverage_pct": pct(m[f"{k}_coverage"]),
                  "imported": round(m["service_imports"].get(k, 0.0), 1)}     # part of supply (service imports)
              for k in ("power", "water", "sewage")},
           "coverage_pct": {k: pct(m[f"{k}_coverage"]) for k in ("police", "fire", "health", "garbage", "school")},
           "scores": {k: round(v, 1) for k, v in env.last_scores.items()}}
    past = next((h for h in w.history if h["month"] >= w.month - 3), None)
    out["trend"] = None if past is None else {
        "since_month": past["month"], "population": m["population"] - past["population"],
        "treasury": round(w.treasury - past["treasury"]),
        "happiness_pct": round(100 * (m["happiness"] - past["happiness"]), 1)}
    cf = cash_flow(w)                    # the same window: the mayor's own transfers and loans apart from operations
    out["operations_3_months"] = None if cf.since is None else {
        "since_month": cf.since, "months": cf.months, "treasury_change": round(cf.change), "operations": round(cf.ops),
        "top_ups": round(cf.top_ups), "loans": round(cf.loans), "other": round(cf.other)}
    costs = sorted(last.get("expense_breakdown", {}).items(), key=lambda kv: -kv[1])
    out["costs_last_month"] = {k: round(v) for k, v in costs if round(v)}
    return out


def plan_slice(fleet: FleetState, role: str) -> dict:
    """The whole plan for the mayor; goals, notes and the role's budget target and priority for a department."""
    p = fleet.plan
    if not fleet.plan_versions:
        return {"ok": True, "plan": None, "note": "no plan yet" + (" - write one with update_plan"
                                                                  if role == "mayor" else "")}
    if role == "mayor":
        return {"ok": True, "version": len(fleet.plan_versions), **p}
    out = {"ok": True, "month": p["month"], "goals": p["goals"], "budget": p["budgets"].get(role),
           "priority": p["priorities"].get(role)}
    if p["notes"]:
        out["notes"] = p["notes"]
    return out


def reserve_why(w) -> str:
    """What sets reserve(w), for the text after "the $3,711 reserve": "" at RESERVE_MIN, " (3 x $1,237 expenses)", or
    " (6 x $676 expenses: last month's deficit was $2)"."""
    mult, income, expenses = _reserve_months(w)
    if mult * expenses <= RESERVE_MIN:
        return ""
    deficit = f": last month's deficit was {money(expenses - income)}" if mult == 6 else ""
    return f" ({mult} x {money(expenses)} expenses{deficit})"


def flood_ahead(w) -> bool:
    """The forecast the briefing shows (World.forecast) names a flood: fire's money is then levee money, first/supply
    for the reserve and loan guards (the playbook's flood rule funds fire at once, though fire's needs line may name a
    cheaper station and list no levee site)."""
    return any("flood" in s.lower() for s in w.forecast())


def bills_ahead(w, months_ahead: int) -> float:
    """What operations take from the treasury over `months_ahead` months at their recent pace (cash_flow's
    ops_per_month), 0 while they run a surplus: the loan guard counts it as a need. Review: at $300 with operations
    at -$800/month and a 3-month cadence, a loan was refused as "would pay for another building"."""
    return max(0.0, -cash_flow(w).ops_per_month * months_ahead)


# World._economy's expenses the Finances costs line shows as imports, apart from city services: services bought from
# outside (a utilities shortfall), and in a region the trade deficit and deal payments to neighbours
IMPORT_COSTS = ("service_imports", "regional_trade", "neighbour_deals")


class CashFlow(NamedTuple):
    """The treasury change over overview()'s trend window, split (cash_flow): change = ops - top_ups + loans + other.
    Then last month's costs. Smoke v13-v19: the treasury change mixed in the mayor's own transfers, so 30 of 81 turns
    showed a "loss" while operations ran a surplus, and interest never showed in dollars."""
    since: int | None       # the window starts after this month's history entry (overview's trend since_month); None
                            # before the first month (every field 0)
    months: int             # months of operations in the window (history entries after `since`; 0 at month 1)
    change: float           # the treasury now less at `since`: this turn's moves so far included
    ops: float              # sum of income - expenses over those months
    top_ups: float          # the mayor's allocations, take-backs netted (dept_allocated), this month's so far included
    loans: float            # the debt now less at `since`: borrowed less repaid
    other: float            # change - ops + top_ups - loans: milestone rewards, offer bonuses (0 without them)
    income: float           # last month (the latest history entry)
    expenses: float         # last month, all of it: buildings + services + interest
    buildings: dict         # last month's upkeep of buildings, roads and parks: expense_breakdown entries that are
                            # dept_upkeep keys ("utilities", "roads", "police", ...), above 0, dearest first
    services: float         # the rest of last month's expenses but imports and interest (city services, town hall,
                            # policies)
    interest: float         # last month's loan_interest
    imports: float = 0.0    # last month's IMPORT_COSTS: services bought from outside and region trade and deal
                            # payments (review, region month 36: $173 of imports read as "city services")

    @property
    def ops_per_month(self) -> float:
        """ops / months; last month's income - expenses while the window holds none (month 1); 0 before any."""
        return self.ops / self.months if self.months else self.income - self.expenses


def cash_flow(w) -> CashFlow:
    """The treasury change since overview()'s trend month (the first history entry at most 3 months back), split into
    operations, top-ups and loans, the rest being rewards and bonuses (CashFlow), with last month's costs by kind.
    Loans and this month's top-ups are read off the world, so a split made mid-turn still adds up."""
    past = next((h for h in w.history if h["month"] >= w.month - 3), None)
    if past is None:
        return CashFlow(None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0)
    after = [h for h in w.history if h["month"] > past["month"]]
    change = w.treasury - past["treasury"]
    ops = sum(h["income"] - h["expenses"] for h in after)
    top_ups = sum(sum(h.get("dept_allocated", {}).values()) for h in after) + sum(w.ledger["allocated"].values())
    loans = w.debt - past["debt"]
    last = w.history[-1]
    spent, upkeep = last.get("expense_breakdown", {}), last.get("dept_upkeep", {})
    buildings = dict(sorted(((k, v) for k, v in spent.items() if k in upkeep and v > 0), key=lambda kv: -kv[1]))
    interest = spent.get("loan_interest", 0.0)
    imports = sum(spent.get(k, 0.0) for k in IMPORT_COSTS if k not in upkeep)
    return CashFlow(past["month"], len(after), change, ops, top_ups, loans, change - ops + top_ups - loans,
                    last["income"], last["expenses"], buildings,
                    last["expenses"] - sum(buildings.values()) - interest - imports, interest, imports)


class DeptMoney:
    """What a department holds and spent, and how much of it is idle: it keeps max(IDLE_MULT x its spending over the
    last SPEND_MONTHS months, its dearest ready or blocked first building (first_build), IDLE_MIN), and the rest, when
    it is IDLE_MIN or more, is idle. Smoke v9-v12: planning held $7,000-10,700 unspent all game in 7 of 8 runs,
    spending about $1,000 a quarter, while the mayor took loans in 6 and education got no school in 5. Money moved to
    it this month doesn't count: the mayor acts first, so the rest sat through at least one of its turns (at month 0
    none has had one). Its observation (`build`) is built once, on first use, and only asked for when a site's price
    can matter: funds short of its dearest priced purpose, a per-tile purpose (a stretch has no such bound), or a
    price above what its spending keeps. Planning's is its road into open land and zoning the land it opens
    (road_build), bounded by road_max: live 120-month run, seed 0, month 9: it read "idle $1,243" with nothing priced,
    the mayor took that and $1,000 more, and its $150 extension failed twice on "the planning budget only has $0"."""

    def __init__(self, env, dept: str):
        w = env.world
        self.env, self.dept = env, dept
        self.funds = w.dept_funds[dept]
        window = w.history[-SPEND_MONTHS:]
        self.months = len(window)               # 0 before the first month: nothing shown, nothing idle
        self.spent = sum(h["dept_spent"].get(dept, 0) for h in window)
        # the most it had in one of those months: month-end funds plus that month's spending (the mayor moves money
        # first), so a station bought in the month counts as held
        self.held_max = max((h["dept_funds"].get(dept, 0) + h["dept_spent"].get(dept, 0) for h in window), default=0)
        # without this month's top-ups; the ledger nets the month, so a take-back and a top-up in one turn can leave
        # part of the top-up counted as idle (rare)
        self.held = self.funds - max(0.0, w.ledger["allocated"].get(dept, 0.0))
        self.floor = max(IDLE_MULT * self.spent, IDLE_MIN)         # what it keeps whatever its sites cost
        priced = [pr for p in SITE_PURPOSES[dept] if (pr := site_price(w, dept, p))]
        self.priced = bool(priced)
        self.bound = None if any(pr.per_tile for pr in priced) else max((pr.cost for pr in priced), default=0)
        if dept == "planning":                  # its road and zoning (road_build), no dearer than road_max
            self.bound = road_max(w)
        if dept == "utilities":                 # a building per short supply (short_supplies): each supply's dearest
            dear = {}                           # summed. Smoke v20 seed 1, month 6: $4,400 above the $4,000 coal
            for pr in priced:                   # plant read "funds cover its builds", power 17% and sewage 0%
                s = supply_of(pr.name)
                dear[s] = max(dear.get(s, 0.0), pr.cost)
            self.bound = sum(dear.values())
        if self.bound is not None and dept in TIER_UP_LINES:     # a tier upgrade its next build may name (tier_up,
            self.bound = max(self.bound, upgrade_bound(w, dept))  # school_upgrade): v1 police held $7,095, above its
                                                                  # $4,000 jail, and read idle beside an $7,800 HQ
        # its funded first buildings with no lot of their own (unclaimed_firsts across departments): funding_gaps
        # sets it, so a DeptMoney made alone reads () and its kind is "blocked" only for lack of land
        self.unclaimed: tuple[str, ...] = ()
        # the buildings its needs line names (goal): above 1 only to keep coverage up, up to `want`, as many as its
        # funds and the mayor's money cover (funding_gaps sets it; keep_up_counts); 1 is the line as it always was
        self.count = 1
        self.counted = False                    # whether keep_up_counts set `count` (a DeptMoney made alone: no)
        # sites an earlier department's keep-up buildings take (keep_up_counts sets it, in turn order, before its
        # keep_ups are read): police stations and clinics rank the same uncovered homes, and both named (22,30)
        self.avoid: frozenset = frozenset()

    @cached_property
    def keep_ups(self) -> tuple[tuple[int, KeepUp], ...]:
        """For each building its next build names (build) whose coverage is under KEEP_UP_GOAL: (its index in
        `named`, keep_up), lowest coverage first; () unless every named building is another one of a KEEP_UP purpose
        with ready sites (FirstBuild.sites): a first building, utilities, transport and planning keep their line."""
        if not self.keep_ups_tried:
            return ()
        return tuple((i, ku) for i, ku in self._keep_ups_all if ku.n)

    @cached_property
    def _keep_ups_all(self) -> tuple[tuple[int, KeepUp], ...]:
        b, w = self.build, self.env.world
        out = [(i, ku) for i, (p, spots) in enumerate(zip(b.purposes, b.sites))
               if (ku := keep_up(w, self.dept, p, [s for s in spots if s not in self.avoid])) is not None]
        return tuple(sorted(out, key=lambda t: t[1].now))

    @property
    def stuck(self) -> str:
        """When its next build is another building of a KEEP_UP purpose under KEEP_UP_GOAL and no ready site of it
        adds KEEP_UP_GAIN (keep_up's gap): "ready sites add under 1%: the uncovered homes near (38,28) are out of
        their road reach", then, when DOWNTOWN_SHOWN or more of the coverage is lost on level 4-5 tiles, "; 7 points
        are lost downtown (level 4-5), where tier 1 covers 60% at most: upgrade_facility to tier 2"; "" otherwise
        (also with a tie of which one adds). Late smoke 20261007 seed 2, month 102: no ready fire or police station
        site added coverage (the one built at (37,23) at month 99 served one building, load 1 of 260): the homes
        without cover were on another road, and 7 of the 19 points missing were downtown."""
        ku = next((ku for _, ku in self._keep_ups_all if ku.gap is not None), None) if self.gap else None
        if ku is None:
            return ""
        return f"ready sites add under {100 * KEEP_UP_GAIN:g}%: {stuck_note(ku)}"

    @property
    def gap(self) -> tuple[int, int] | None:
        """keep_up's gap when no ready site of its next build adds coverage (stuck), or when nothing is named because
        none does (FirstBuild.gap: keep_up_room), else None."""
        if not self.keep_ups_tried:
            b = self.build if self.dept in KEEP_UP_GATED else None
            return b.gap if b is not None else None
        if self.keep_ups:
            return None
        return next((ku.gap for _, ku in self._keep_ups_all if ku.gap is not None), None)

    @property
    def keep_ups_tried(self) -> bool:
        """keep_ups ran on its next build (every named building another one of a KEEP_UP purpose with sites). Never
        for planning, utilities or transport, whose build isn't asked for here (no observation built for it)."""
        m = self.env.world.metrics()                # every coverage it builds for at the goal: nothing to ask (and no
        if all(m[COVERAGE_OF[d, p]] >= KEEP_UP_GOAL for d, p in KEEP_UP if d == self.dept):    # observation built)
            return False
        b = self.build
        return not (b is None or b.cost is None or not b.named or b.all_of or len(b.sites) != len(b.named)
                    or any(not n.have for n in b.named))

    @property
    def want(self) -> int:
        """The buildings its coverage asks for (keep_ups' counts summed) when that is 2 or more, else 1."""
        n = sum(ku.n for _, ku in self.keep_ups)
        return n if n > 1 else 1

    def goal_of(self, count: int) -> FirstBuild | None:
        """Its next build for `count` buildings: `build` itself without keep_ups; else the first `count` of
        the keep_ups' buildings (each service's in turn, lowest coverage first), bought together at `count` x the
        price, each Need with its count, its sites' coverage estimate and its upkeep x count: "3 more fire stations
        (coverage 78%, ~91% with them, $165/month)", "falling: 94 -> 86 -> 78%" for the coverage when it fell."""
        b = self.build
        if count < 1 or not self.keep_ups:
            return b
        order = [i for i, ku in self.keep_ups for _ in range(ku.n)][:count]
        kus, named, purposes, sites = dict(self.keep_ups), [], [], []

        def status(ku: KeepUp, k: int) -> str:
            fall = coverage_trend(self.env.world, ku.key)
            return f"coverage {fall or pct(100 * ku.now)}, ~{pct(100 * ku.levels[k])} with {'them' if k > 1 else 'it'}"

        if count == 1 and len(b.named) > 1:     # a tie stays a choice, each with its estimate and best site
            return b._replace(named=tuple(n._replace(status=status(kus[i], 1)) if i in kus else n
                                          for i, n in enumerate(b.named)),
                              sites=tuple(kus[i].sites[:1] if i in kus else spots[:1]
                                          for i, spots in enumerate(b.sites)), keep_up=True)
        for i in dict.fromkeys(order):
            k, n, ku = order.count(i), b.named[i], kus[i]
            named.append(n._replace(count=k, status=status(ku, k), upkeep=n.upkeep * k))
            purposes.append(b.purposes[i])
            sites.append(ku.sites[:k])
        return b._replace(cost=b.cost * len(order), named=tuple(named), all_of=len(named) > 1,
                          purposes=tuple(purposes), sites=tuple(sites), keep_up=True)

    @property
    def goal(self) -> FirstBuild | None:
        """goal_of(count): what its needs line, its ask and the mayor's notes name."""
        return self.goal_of(self.count)

    @cached_property
    def build(self) -> FirstBuild | None:
        return road_build(self.env) if self.dept == "planning" else first_build(self.env, self.dept)

    @cached_property
    def keep(self) -> float:
        if self.bound is not None and self.bound <= self.floor:
            return self.floor
        return max(self.floor, self.build.dearest if self.build else 0)

    @cached_property
    def idle(self) -> int:
        """Funds above what it keeps, rounded down, when IDLE_MIN or more; else 0."""
        if not self.months or self.held - self.floor < IDLE_MIN:       # not idle whatever its sites cost
            return 0
        idle = math.floor(self.held - self.keep)
        return idle if idle >= IDLE_MIN else 0

    @property
    def needs(self) -> FirstBuild | None:
        """Its cheapest ready site when its funds are short of it (the mayor's needs line), else None; with a count
        above 1, its goal when its funds are short of that."""
        if self.count > 1:
            b = self.goal
            return b if short_of(self.funds, b.cost) else None
        if self.bound is not None and not short_of(self.funds, self.bound):
            return None
        b = self.build
        if not (b and b.cost is not None and short_of(self.funds, b.cost)):
            return None
        return self.goal_of(1) if self.keep_ups else b      # the same price: the building where it adds the most

    @property
    def blocked(self) -> tuple[str, ...]:
        """With no ready site, the cheapest first building(s) it has no lot for (the mayor's line names them); ()
        otherwise. Asking builds its observation (planning has no priced purpose: none)."""
        return self.build.blocked if self.priced and self.build else ()

    @property
    def wants_more(self) -> bool:
        """Whether a top-up would be used: before the first month always; then only when it used more than USED_SHARE
        of the most it held over the last SPEND_MONTHS months, or never held what it keeps (its dearest ready or
        blocked first building).
        After a take-back planning held its keep, 2 x its spending, below its $9,000 target, and read "below
        target" again: topped up, then idle, then taken back. More than, not at least: a take-back leaves exactly
        1 / IDLE_MULT = USED_SHARE used over the next quarter (test_a_take_back_starts_no_top_up_cycle)."""
        return (not self.months or self.spent > USED_SHARE * self.held_max
                or self.held_max < self.floor or self.held_max < self.keep)     # floor first: keep may need a build

    @property
    def kind(self) -> str:
        """What its money is for (KINDS): "idle" with idle money; else, with a needs line, "growth" for planning,
        "first/supply" when a named building is a first one (Need.first), for utilities (first_build names a
        utilities building only while it is wanted) or for a levee, "another" otherwise; else "blocked" with a first
        building blocked for lack of land (blocked) or funded first buildings with no lot of their own (unclaimed,
        set by funding_gaps); else "nothing". Smoke v16-v19: first buildings at a $0 target were funded 16 times in
        36 against 36 in 40 with a target."""
        if self.idle:
            return "idle"
        if (b := self.needs) is not None:
            return self._need_kind(b)
        return "blocked" if self.blocked or self.unclaimed else "nothing"

    def _need_kind(self, b: FirstBuild) -> str:
        """"growth" for planning, "first/supply" for a first building (Need.first), utilities or a levee, else
        "another"."""
        if self.dept == "planning":
            return "growth"
        first = self.dept == "utilities" or any(n.first or n.what.removesuffix("s").endswith("levee tile")
                                                for n in b.named)
        return "first/supply" if first else "another"

    @property
    def buys(self) -> str:
        """What its money is for, whether or not its funds cover it: like `kind`, but by its next build (build) rather
        than its needs line, so a top-up that already paid for its first building is still first/supply. The loan
        guard reads it for this turn's top-ups (ToolBox._loan_refusal)."""
        if self.idle:
            return "idle"
        b = self.build
        if b is not None and b.cost is not None and b.named:
            return self._need_kind(b)
        return "blocked" if self.blocked or self.unclaimed else "nothing"

    @property
    def top_up(self) -> int:
        """What its needs line lacks, rounded up to whole dollars (needs.cost - funds); 0 without a needs line."""
        b = self.needs
        return math.ceil(b.cost - self.funds) if b is not None else 0

    @property
    def upkeep(self) -> float:
        """The monthly upkeep its needs line adds (FirstBuild.upkeep): the largest Need.upkeep of the buildings it
        names, their sum when bought together (utilities' supplies); 0 without."""
        b = self.needs
        return b.upkeep if b is not None else 0.0

    @property
    def what(self) -> str:
        """What its needs line buys, without the details (FirstBuild.what): "its first police station", "another clinic
        or another landfill", "a road into open land and zoning it", "its first coal plant and water tower"; "" without
        a needs line."""
        b = self.needs
        return b.what if b is not None else ""

    @property
    def state(self) -> str:
        """Its money for a line with nothing else to show (no idle money, needs line or block): "funds cover its
        builds" when its funds reach its dearest priced purpose (bound), "funds cover its next build" when they reach
        its next build (first_build: a first building it lacks before a cheaper site), else why nothing is named:
        "supply has room" (utilities sites left out while their supply has room: FirstBuild.room), education's
        "enrolment covered: elementary 99%, high-school 92%" (FirstBuild.covered), "only a library
        ready ($1,500)" (FirstBuild.extra), or "nothing ready to buy"; "" for planning (never shown). Review, seed 0,
        month 108: utilities read "nothing ready to buy" with a pump and an outlet ready, water and sewage spare."""
        if self.dept == "planning":
            return ""
        if self.count > 1 and not short_of(self.funds, self.goal.cost):
            return f"funds cover {self.goal.what}"
        if self.priced and self.bound is not None and not short_of(self.funds, self.bound):
            return "funds cover its builds"
        b = self.build
        if b is not None and b.cost is not None and not short_of(self.funds, b.cost):
            return "funds cover its next build"
        if b is not None and b.room:
            return "supply has room"
        if b is not None and b.covered:
            return b.covered + (f"; only {a_or_an(b.extra[0])} ready ({money(b.extra[1])})" if b.extra else "")
        if b is not None and b.extra:
            return f"only {a_or_an(b.extra[0])} ready ({money(b.extra[1])})"
        return "nothing ready to buy"

    @property
    def keep_why(self) -> str:
        """What set `keep`: "2 x its $658 spending" (IDLE_MULT x spent), "for" its dearest ready or blocked first
        building (FirstBuild.keep_for: "for its first university", "for another fire station"; planning "for its road
        into open land and zoning it") or "the $1,000 minimum" (IDLE_MIN). Review, month 60: "(its dearest first
        building)" read beside "needs $2,500 for another fire station"."""
        if self.keep > self.floor:
            if self.dept == "planning":
                return f"for its {ROAD_NAME}"
            return f"for {self.build.keep_for}" if self.build.keep_for else "for its dearest first building"
        if IDLE_MULT * self.spent > IDLE_MIN:
            return f"{IDLE_MULT} x its {money(self.spent)} spending"
        return f"the {money(IDLE_MIN)} minimum"

    def cant_pay(self, left: float) -> str:
        """What `left` in its funds can't pay for, for a take-back's refusal and result: its next build ("a road into
        open land and zoning it ($640)"; first_build / road_build) when `left` is short of it and keep_why doesn't name
        that building already; "" otherwise. Review, fleet_scripted seed 0: "so it can't build next quarter" whenever
        a take-back left less than the keep, also with planning's $640 road still covered by the $950 left (month 12)
        and police's $2,500 station by $3,950 left of a $4,000 keep (month 90)."""
        b = self.build
        if b is None or b.cost is None or not b.named or not short_of(left, b.cost):
            return ""
        if self.keep > self.floor and b.cost >= self.keep:         # keep_why names it ("for another fire station")
            return ""
        return f"{b.what} ({money(b.cost)})"


class FundItem(NamedTuple):
    """One department on the to-fund line (Funding.to_fund): what its needs line lacks."""
    dept: str
    kind: str               # its DeptMoney.kind: one of NEED_KINDS ("first/supply", "growth", "another")
    amount: int             # DeptMoney.top_up (> 0)
    upkeep: float           # DeptMoney.upkeep: the monthly upkeep the building adds (0 for planning's road)
    what: str               # DeptMoney.what: "its first police station", "another clinic or another landfill"
    count: int = 1          # DeptMoney.count: the buildings it buys (more than one to keep coverage up)


class Funding(NamedTuple):
    """The department money picture (funding_gaps): the plan-target gaps (the "<" marker, update_plan's
    below_target), and, needs-based whatever the targets, each department's kind, the order and the top-ups that
    buy what the needs lines name. spend_lines renders it."""
    gaps: dict[str, dict]   # {dept: {"funds", "target", "gap"}}, in turn order; idle departments left out
    total: int              # the gaps summed
    reserve: float          # reserve(w)
    spare: float            # treasury above the reserve (negative below it)
    idle: dict[str, int]    # {dept: idle funds} (DeptMoney), in turn order
    depts: dict[str, DeptMoney]     # every department in the game, in turn order
    # the rest are needs-based; defaults only for tests that build one by hand (never mutate them)
    unclaimed: dict[str, tuple[str, ...]] = {}  # unclaimed_firsts(depts): {dept: ("an elementary school", ...)}
    order: tuple[str, ...] = ()             # every department, by KIND_RANK of its kind, turn order within a kind
    to_fund: list[FundItem] = []            # the departments of NEED_KINDS with a top-up, in `order`
    firsts_at_zero: tuple[str, ...] = ()    # the first/supply departments (in order) whose plan target is 0 or unset
    months_left: int | None = None          # months left in the game (None: not known): the hold's "close that"
    unusable: dict[str, str] = {}           # {dept: wait_why}: a plan target above its funds (target_above) with
                                            # nothing to buy now (buys_nothing); no gap, flagged on its line

    @property
    def spendable(self) -> float:
        """The money there is to move: the treasury above the reserve plus the idle money to take back."""
        return self.spare + sum(self.idle.values())

    @property
    def dip(self) -> float:
        """What first/supply and growth may use, in a hold or not: the treasury plus the idle money above RESERVE_MIN
        (the treasury is spare + reserve), at least spendable since the reserve is at least RESERVE_MIN. Review: as a
        hold's allowance only, $1 more idle money (or treasury past the reserve) left the hold and cut what a first
        school could use from $3,605 to $1 (and the old max(0, treasury - $3,000) + idle let idle money count while
        the treasury itself was under $3,000)."""
        return max(0.0, self.spare + self.reserve + sum(self.idle.values()) - RESERVE_MIN)

    @property
    def hold(self) -> bool:
        """Nothing to spend (spendable <= 0): the treasury is at or under the reserve and the idle money doesn't lift
        it above; spend_lines says what first/supply and growth may still use (dip)."""
        return self.spendable <= 0


def funding_gaps(env, fleet: FleetState) -> Funding:
    """Departments in the game whose funds are below their plan budget target by more than max(GAP_MIN, GAP_SHARE x
    target) and that would use a top-up (DeptMoney.wants_more), and whether the treasury above the reserve covers
    them: the "<" marker and update_plan's below_target. An idle department is no gap: smoke v12 seed 0 read
    "planning: funds $7,834 < target $9,000" and topped it up to $10,721.
    Needs-based, whatever the targets: each department's kind (DeptMoney.kind, with unclaimed_firsts across
    departments in turn order, so site claims don't change), the order (KIND_RANK, turn order within a kind), the
    top-ups (to_fund) and the first buildings at a $0 target. Smoke v16-v19: the to-do line came from targets alone,
    education was at $0 in 15 of 15 plans, and 24 service first buildings went unfunded with money above the reserve.
    Builds each department's observation it needs (DeptMoney.build), about 5 ms in all."""
    w, budgets = env.world, fleet.plan["budgets"]
    depts = {d: DeptMoney(env, d) for d in DEPARTMENTS if d in env.roles}
    idle = {d: m.idle for d, m in depts.items() if m.idle}
    unclaimed = unclaimed_firsts(depts)
    for d, m in depts.items():
        m.unclaimed = unclaimed.get(d, ())
    turn = list(depts)
    keep = reserve(w)
    keep_up_counts(depts, w.treasury - keep + sum(idle.values()))
    kinds = {d: m.kind for d, m in depts.items()}
    gaps, unusable = {}, {}
    for d, m in depts.items():
        if d in idle or not target_above(budgets.get(d), m.funds):
            continue
        if buys_nothing(m):                     # nothing to buy now: no gap, the target flagged (unusable)
            unusable[d] = wait_why(m)
        elif m.wants_more:
            target = budgets[d]
            gaps[d] = {"funds": round(m.funds), "target": round(target), "gap": round(target - m.funds)}
    order = tuple(sorted(depts, key=lambda d: (KIND_RANK[kinds[d]], turn.index(d))))
    to_fund = [FundItem(d, kinds[d], m.top_up, m.upkeep, m.what, m.count) for d in order
               if kinds[d] in NEED_KINDS and (m := depts[d]).top_up > 0]
    zero = tuple(d for d in order if kinds[d] == KINDS[0] and not budgets.get(d))
    return Funding(gaps, sum(g["gap"] for g in gaps.values()), keep, w.treasury - keep, idle, depts, unclaimed, order,
                   to_fund, zero, env.cfg.months - w.month, unusable)


def keep_up_counts(depts: dict[str, DeptMoney], spendable: float) -> None:
    """Sets each department's count (DeptMoney.count) where its coverage asks for 2 or more buildings (want; never
    for idle money): first as many as its own funds pay for, then, while the money to spend (`spendable`: the treasury
    above the reserve plus the idle money) covers every top-up the needs lines ask at those counts, one more at a time
    in turn, most urgent first (kind, then turn order), as long as what the next one adds to its top-up fits in what
    is left. So the to-fund line's cover clause still covers all of them (fund_cover), and a city short of money, or
    in a hold, keeps its one-building lines. The guards read the kinds this leaves: one more building is still
    "another" (RESERVE_REFUSED alike), and a department's keep (what a take-back leaves) doesn't change."""
    taken: set = set()
    for m in depts.values():                    # in turn order: a site goes to the first department picking it
        m.counted, m.avoid = True, frozenset(taken)
        if not m.idle:
            taken |= {s for _, ku in m.keep_ups for s in ku.sites}
    grow = [d for d, m in depts.items() if not m.idle and m.want > 1]
    for d in grow:
        m = depts[d]
        m.count = max(1, min(m.want, math.floor((m.funds + 1e-9) / m.build.cost)))
    if spendable <= 0:
        return
    rank = {d: (KIND_RANK[m.kind], i) for i, (d, m) in enumerate(depts.items())}
    left = spendable - sum(m.top_up for m in depts.values() if m.kind in NEED_KINDS)
    grow = sorted((d for d in grow if depts[d].count < depts[d].want), key=rank.get)
    while grow and left > 0:
        for d in list(grow):
            m = depts[d]
            before = m.top_up
            m.count += 1
            if m.top_up - before > left:
                m.count -= 1
                grow.remove(d)
                continue
            left -= m.top_up - before
            if m.count >= m.want:
                grow.remove(d)


def _to_spend_line(w, f: Funding, cf: CashFlow) -> str:
    """spend_lines' first line: the money to spend, or the hold and when operations close it, then what first/supply
    and growth may use when that is more (Funding.dip)."""
    idle = sum(f.idle.values())
    res = f"the {money(f.reserve)} reserve{reserve_why(w)}"
    pm = cf.ops_per_month
    ops = f"; operations {signed(pm, money)}/month" if cf.since is not None else ""     # none before the first month
    dip = (f"; first/supply and growth may use {money(f.dip)} (treasury{' + idle' if idle else ''} above "
           f"{money(RESERVE_MIN)})")
    firsts = any(i.kind in NEED_KINDS[:2] for i in f.to_fund)
    more = dip if firsts and round(f.dip) > round(f.spendable) else ""        # a hold names it whatever is listed
    if f.spendable > 0 and f.spare >= 0:
        return (f"- to spend: {money(f.spare)} above {res}"
                + (f" + {money(idle)} idle to take back = {money(f.spendable)}" if idle else "") + ops + more)
    if f.spendable > 0:                                    # only because of the idle money
        return (f"- to spend: {money(f.spendable)}: {money(idle)} idle to take back, less the {money(-f.spare)} the "
                f"treasury lacks for {res}{ops}{more}")
    gap = -f.spare
    where = f"is {money(gap)} under {res}" if round(gap) else f"is at {res}"
    n = math.ceil(gap / pm) if round(gap) and round(pm) > 0 else None
    close = f" close that in {months(n)}" if n and (f.months_left is None or n <= f.months_left) else ""
    return (f"- to spend: none (hold): the treasury {where}{ops}{close}"
            + (dip if round(f.dip) else f"; nothing above {money(RESERVE_MIN)}"))


def fund_cover(f: Funding) -> list[str]:
    """The departments the to-fund line's cover clause names, greedy in to_fund order (one that doesn't fit is
    skipped; a cheaper one after it may fit): first/supply and growth against what they may use (Funding.dip),
    another against what is left above the reserve (spendable), each covered item using both up. Review: covering
    first/supply against spendable outside a hold cut their allowance at the hold's edge, so a richer city covered
    less."""
    left_first, left, covered = f.dip, f.spendable, []
    for i in f.to_fund:
        if i.amount <= (left_first if i.kind in NEED_KINDS[:2] else left):
            covered.append(i.dept)
            left_first, left = left_first - i.amount, left - i.amount
    return covered


def _to_fund_line(f: Funding) -> str:
    """spend_lines' second line: the top-ups by kind, and what the money covers (fund_cover)."""
    if not f.to_fund:
        return "- to fund: nothing ready"
    parts = []
    for kind in NEED_KINDS:
        items = [i for i in f.to_fund if i.kind == kind]
        if not items:
            continue
        up = sum(i.upkeep for i in items)
        tail = f"upkeep +{money(up)}/month" if round(up) else ""
        part = f"{kind}: " + ", ".join(f"{i.dept} {money(i.amount)}" + (f" ({i.count} buildings)" if i.count > 1
                                                                        else "") for i in items)
        if len(items) > 1:
            part += f" ({money(sum(i.amount for i in items))}" + (f", {tail}" if tail else "") + ")"
        elif tail:
            part += f" ({tail})"
        parts.append(part)
    line = "- to fund: " + " | ".join(parts)
    firsts = [i for i in f.to_fund if i.kind in NEED_KINDS[:2]]
    budget = f.dip if firsts else f.spendable
    if f.hold and not firsts or not round(budget):         # a hold covers only first/supply and growth
        return line
    covered = fund_cover(f)
    rest = [i for i in (firsts if f.hold else f.to_fund) if i.dept not in covered]
    # under $3,000 when only first/supply and growth are left and they may go below the reserve; else the reserve
    under = (money(RESERVE_MIN) if all(i in firsts for i in rest) and round(f.dip) > round(f.spendable)
             else "the reserve")
    owed = money(sum(i.amount for i in rest))
    if not covered and f.hold:
        return line + f": {money(budget)} covers no first/supply or growth; their {owed} would go under {under}"
    if not covered:
        return line + f": {money(budget)} covers none of them; all {owed} would go under {under}"
    return (line + f": {money(budget)} covers {', '.join(covered)}"
            + (f"; the other {owed} would go under {under}" if rest else ""))


def spend_lines(w, f: Funding) -> list[str]:
    """[to_spend_line, to_fund_line] for the mayor's briefing (each starts with "- "):
    - "- to spend: $2,698 above the $3,000 reserve[ + $1,508 idle to take back = $4,206]; operations +$233/month";
      positive only through idle money, "- to spend: $1,200: $1,508 idle to take back, less the $308 the treasury
      lacks for the $3,711 reserve (3 x $1,237 expenses); operations ..."; in a hold (Funding.hold), "- to spend: none
      (hold): the treasury is $1,723 under the $3,711 reserve (3 x $1,237 expenses); operations +$381/month close
      that in 5 months" (no "close that" unless operations are positive, nor past the game's end; no operations
      before the first month). Each then ends "; first/supply and growth may use $X (treasury[ + idle] above
      $3,000)" (Funding.dip) when that is more than the money to spend and the to-fund line lists some; a hold
      always, "nothing above $3,000" at 0.
      Operations per month are cash_flow's.
    - "- to fund: first/supply: health $2,000, education $2,500 ($4,500, upkeep +$85/month) | growth: planning $315 |
      another: police $2,500 (upkeep +$55/month): $4,206 covers health, planning; the other $5,000 would go under the
      reserve": the to_fund items by kind, each kind's sum and upkeep when it has several, then fund_cover's greedy
      cover in that order, the amount being what first/supply and growth may use (dip; spendable with none of them),
      and what is left "would go under $3,000" when only first/supply and growth are left and may go below the
      reserve, else "under the reserve". In a hold only first/supply and growth ("$400 covers no first/supply or
      growth; their $540 would go under $3,000"), and no cover clause when that is 0. "- to fund: nothing ready"
      without items.
    The harness adds only arithmetic: which to fund, how much and in what order stays the mayor's call."""
    return [_to_spend_line(w, f, cash_flow(w)), _to_fund_line(f)]


def _buildings(count: int) -> str:
    """", 3 buildings" after a to-fund item's kind when it buys more than one (keep_up_counts), else ""."""
    return f", {count} buildings" if count > 1 else ""


def keep_up_where(b: FirstBuild) -> str:
    """The buildings of a keep-up goal (DeptMoney.goal) and where, in the department's words: "3 more fire stations
    at (37,23), (12,20) and (30,28)", several services joined by "and" (by "or" for a tie: one of them)."""
    return (" and " if b.all_of or len(b.named) < 2 else " or ").join(f"{n.label} at {_and([f'({x},{y})' for x, y in spots])}" for n, spots in zip(b.named, b.sites))


def still_to_fund(f: Funding, n: int = FUNDED_SHOWN) -> str:
    """The to_fund items for a money result: "education $2,500 (first/supply), fire $2,500 (another)", the first `n`
    then "+N more"; "none" without any."""
    items = [f"{i.dept} {money(i.amount)} ({i.kind}{_buildings(i.count)})" for i in f.to_fund]
    if not items:
        return "none"
    return ", ".join(items[:n]) + (f", +{len(items) - n} more" if len(items) > n else "")


# ------------------------------------------------------------------ requests to the mayor
# The 120-month LLM runs (trinity v19 / v23): the mayor answered 59-81% of the requests sent to it, every other
# department ~100%. In v23 seed 0, 102 money asks went unanswered: the mayor saw 94 of them in its digest, gave money
# to someone else in the window every time (first/supply in a hold, 66 of them), and 80 asked for "another" building;
# it replied to few. The briefing's Open requests block and the end_turn reminder name what is still open.
REQUESTS_SHOWN = 4           # departments on the mayor's Open requests block before "+N more"
REQUEST_CHARS = 75           # the latest request of each department quoted there
PRIORITY_CHARS = 40          # a priority quoted in update_plan's nothing_to_fund flags
REMIND_MIN_LEFT = 2          # tool calls left after the end_turn for its open-requests reminder (a reply, end_turn)


class OpenRequest(NamedTuple):
    """A department's request to the mayor (or to planning: open_requests' `to`) with no answer yet
    (bench/collab.py:answer: no top-up to it, no action near the tiles it names, no message back) inside its window of
    2 x act_every months, where the collaboration score still counts one."""
    month: int
    dept: str
    text: str
    amount: float | None    # the dollars it asks for (collab's _amount), None when it names none
    last: bool              # the mayor's next turn falls outside the window: this is its last turn to answer


def open_requests(env, fleet: FleetState, pending=(), to: str = "mayor") -> list[OpenRequest]:
    """The requests to `to` (the mayor, or planning) still open, oldest first, decided as the collaboration score
    decides them (collab.answer) from the council board (messages `to` can see from the window's start), its
    successful actions in the fleet log (the mayor's allocate_budget top-ups; planning's builds, zones, dezones and
    bulldozes, which answer a request naming a tile near them), and `pending`: this turn's send_message calls
    (ToolBox.messages, posted at the end of the turn). Requests are council messages to `to` from a department of this
    city (for planning, the mayor too) that ask something (collab.is_request); later events are ordered by month, turn
    order, messages before actions, as in collab. Region run 20261005 and the 120-month runs: 176 requests went to
    planning (98 from utilities, for riverbank lots), 85 unanswered; in the late smokes it answered none."""
    w = env.world
    act = max(1, int(env.cfg.act_every))
    window = 2 * act
    since = w.month - window
    order = {r.split(".")[-1]: i for i, r in enumerate(env.roles)}
    msgs = [env._local(m) for m in env.council.visible_to(env._name(to), since_month=since)]
    events = [{"month": m["month"], "kind": "message", "from": m["sender"], "to": m["to"], "text": m["text"]}
              for m in msgs]
    events += [{"month": w.month, "kind": "message", "from": to, "to": m.get("to"), "text": m.get("text", "")}
               for m in pending]
    for e in fleet.log:
        a = e.get("action") or {}
        if to != "mayor" and e["kind"] == "action" and e["role"] == to and e.get("ok") and e["month"] >= since:
            typ = str(a.get("type", ""))
            if typ.startswith(BUILDS) or typ in ("zone", "dezone", "bulldoze", "demolish"):
                events.append({"month": e["month"], "kind": "build" if typ.startswith(BUILDS) else "act",
                               "from": to, "to": to, "text": e.get("message", ""), "tile": action_tile(a)})
            continue
        if (e["kind"] != "action" or e["role"] != "mayor" or not e.get("ok") or e["month"] < since
                or a.get("type") != "allocate_budget"):
            continue
        try:
            amount = float(a.get("amount", 0))
        except (TypeError, ValueError):
            continue
        if amount > 0:
            events.append({"month": e["month"], "kind": "money", "from": "mayor", "to": a.get("department"),
                           "amount": amount, "text": e.get("message", "")})
    key = lambda e: (e["month"], order.get(e["from"], len(order)), 0 if e["kind"] == "message" else 1)
    events.sort(key=key)
    nxt = (w.month // act + 1) * act               # `to`'s next turn (env.acts_this_month)
    rep = replies(events, window)                  # an answer to `to`'s own request: no request
    askers = set(DEPARTMENTS) - {to} if to == "mayor" else ({"mayor", *DEPARTMENTS} - {to})
    out = []
    for i, e in enumerate(events):
        if (e["kind"] != "message" or e["to"] != to or e["from"] not in askers or i in rep
                or not is_request(e["to"], e["text"]) or e["month"] + window < w.month):
            continue
        # as collab's events: the text cut to TEXT_CHARS, the amount and tiles from all of it
        req = {**e, "text": e["text"][:TEXT_CHARS], "amount": asked_amount(e["text"]), "tiles": text_tiles(e["text"])}
        later = [x for x in events[i + 1:] if key(x) > key(e) and x["month"] <= e["month"] + window]
        if answer(req, later)[0] is None:
            out.append(OpenRequest(e["month"], e["from"], e["text"], req["amount"], nxt > e["month"] + window))
    return out


def request_groups(reqs: list[OpenRequest]) -> list[tuple[str, list[OpenRequest]]]:
    """Open requests by department (repeats merged), the longest-waiting department first."""
    groups: dict[str, list[OpenRequest]] = {}
    for r in sorted(reqs, key=lambda r: r.month):
        groups.setdefault(r.dept, []).append(r)
    return list(groups.items())


def request_state(f: Funding, dept: str, covered: list[str]) -> str:
    """What the mayor can tell a department that asked, from funding_gaps (its Departments line names the building):
    "to fund $2,500 (another), not covered (hold)" ("not covered now" outside a hold) / "..., covered" (fund_cover:
    the money covers it after the items before it), or why it needs no top-up: "no top-up needed (nothing ready to
    buy)" (DeptMoney.state), "(holds $1,508 idle)", "(blocked for lack of land)", planning's "(funds cover its road)"
    / "(no road ready)"."""
    m = f.depts.get(dept)
    if m is None:
        return "not in the game"
    k = m.kind
    if k in NEED_KINDS:
        cover = "covered" if dept in covered else "not covered" + (" (hold)" if f.hold else " now")
        return f"to fund {money(m.top_up)} ({k}{_buildings(m.count)}), {cover}"
    return f"no top-up needed ({wait_why(m)})"


def buys_nothing(m: DeptMoney) -> bool:
    """A department with nothing to buy now (kind "nothing": supply has room, nothing ready, only a supplementary
    building, funds covering its builds), so a plan target above its funds is no gap but unusable (Funding.unusable,
    update_plan's nothing_to_fund). Not planning, whose spending goes past its priced road (zoning, parks), nor a
    blocked department, which keeps money for its first building until planning frees a lot. Region run 20261005: the
    mayor kept utilities at a $3,000-7,000 target with supply room ("funds $0 < target $7,000; supply has room"), and
    utilities asked against "supply has room" 28 times."""
    return m.kind == "nothing" and m.dept != "planning"


def nothing_to_fund(f: Funding, budgets: dict, priorities: dict | None = None) -> dict[str, str]:
    """update_plan's flags on targets and priorities that point a department at money it can't use now: each
    department with a target above its funds and nothing to buy (Funding.unusable), or given a priority in this
    update (`priorities`) while it has nothing to buy (buys_nothing): "target $3,000 above its $0 funds and priority
    \"expand water capacity\", but supply has room: no top-up buys anything now". Region run 20261005: utilities read
    a $3,000-7,000 target and "expand water capacity" beside "supply has room" and asked 28 times against it."""
    out = {}
    for d, m in f.depts.items():
        pri = (priorities or {}).get(d) if isinstance(priorities, dict) else None
        given = bool(isinstance(pri, str) and pri.strip()) and buys_nothing(m)
        if d not in f.unusable and not given:
            continue
        parts = ([f"target {money(budgets[d])} above its {money(math.floor(m.funds))} funds"] if d in f.unusable
                 else []) + ([f"priority \"{cut(pri, PRIORITY_CHARS)}\""] if given else [])
        out[d] = f"{' and '.join(parts)}, but {wait_why(m)}: no top-up buys anything now"
    return out


def wait_why(m: DeptMoney) -> str:
    """Why a department with no needs line needs no top-up, in the mayor's words: "holds $1,508 idle", "blocked for
    lack of land", planning's "funds cover its road" / "no road ready", else its DeptMoney.state ("supply has room",
    "nothing ready to buy", ...). request_state and Funding.unusable read it."""
    if m.kind == "idle":
        return f"holds {money(m.idle)} idle"
    if m.kind == "blocked":
        return "blocked for lack of land"
    if m.dept == "planning":
        return "funds cover its road" if m.build is not None and m.build.cost is not None else "no road ready"
    return m.state or "nothing ready to buy"


def planning_reminder(reqs: list[OpenRequest]) -> str:
    """Planning's end_turn refusal while requests to it are open (ToolBox.end_turn): who asks, since when."""
    names = [f"{d} (since m{rs[0].month})" for d, rs in request_groups(reqs)]
    n = len(reqs)
    if is_open():
        return (f"not ended: {n} request{'s' * (n != 1)} to you still open: {', '.join(names)}. Call end_turn again "
                "to end the turn whatever is still open.")
    return (f"not ended: {n} request{'s' * (n != 1)} to you still open: {', '.join(names)}. Answer each now: do what "
            "it asks at the spot it names (your Open requests block names the call), or send_message to it with one "
            "line on where it can build or why not; one send_message can answer several (to: \"police,health\"). "
            "Then call end_turn again: it ends the turn whatever is still open.")


def requests_reminder(reqs: list[OpenRequest]) -> str:
    """The end_turn refusal while requests are open (ToolBox.end_turn): the departments, what they ask, since when."""
    names = [f"{d} ({f'asks {money(a)}, ' if (a := next((r.amount for r in reversed(rs) if r.amount), None)) else ''}"
             f"since m{rs[0].month})" for d, rs in request_groups(reqs)]
    n = len(reqs)
    if is_open():
        return (f"not ended: {n} request{'s' * (n != 1)} to you still open: {', '.join(names)}. Call end_turn again "
                "to end the turn whatever is still open.")
    return (f"not ended: {n} request{'s' * (n != 1)} to you still open: {', '.join(names)}. Answer each department "
            "now: allocate_budget, or send_message to it with one line on why not or when (the reserve, a first "
            "building elsewhere first, its funds already cover it); one send_message can answer several (to: "
            "\"police,fire\"). Then call end_turn again: it ends the turn whatever is still open.")


# ------------------------------------------------------------------ departments' money asks to the mayor
# Region smoke 20261005 (trinity in both cities, 12 months, act_every 3): 46 requests to the mayor, 45 naming money.
# By the department's line on that month's Departments block: 9 for a first building or supply, 9 for planning's road,
# 15 for another building the money above the reserve didn't cover (a second police / fire station, a clinic, a
# school), 13 with nothing ready to buy (transport's "below 1,500 residents" notes, "funds for future stations", a
# library). The mayor answered 93% of them, mostly with a no (follow-through 25 / 29), and the south mayor spent all
# 20 steps of month 9 on refusals. A department's briefing says whether to ask (briefing._ask_line), and the model's
# send_message holds back once a turn a money ask the mayor can't fund now (ToolBox._ask_refusal).
class Ask(NamedTuple):
    """Whether a department should ask the mayor for money now (mayor_ask), and what to say."""
    ok: bool          # first/supply or growth with a top-up, or another building the mayor's money covers now
    amount: int       # DeptMoney.top_up when ok, else 0
    what: str         # what the top-up buys, in the department's words: "your first police station"
    why: str          # ok: why the mayor funds it; else why it won't now
    # the rest only when not ok (_not_now):
    plan: str = ""    # its plan target above its funds (target_above): a ceiling, not money to ask for
    when: str = ""    # what would make an ask valid ("a supply will be 90% used within 3 months at its growth ...")
    instead: str = "" # what to do this turn ("build another police station ($2,500): your funds cover it")


def _yours(text: str) -> str:
    """A Departments-block phrase in the department's own words: "its first school" -> "your first school"."""
    return re.sub(r"\bits\b", "your", text)


def needs_count(m: DeptMoney) -> bool:
    """A DeptMoney made alone whose coverage asks for more than one building: its count comes from funding_gaps
    (keep_up_counts), so a department reads the count the mayor's line names. The briefing and mayor_ask swap it for
    funding_gaps' own."""
    return not m.counted and not m.idle and m.want > 1


def target_above(target: float | None, funds: float) -> bool:
    """A plan target above the funds by at least the gap rule (max(GAP_MIN, GAP_SHARE x target)): funding_gaps' "<"
    marker and its unusable targets, and mayor_ask's plan note."""
    return bool(target) and target - funds >= max(GAP_MIN, GAP_SHARE * target)


def mayor_ask(env, fleet: FleetState, dept: str, m: DeptMoney | None = None) -> Ask:
    """What `dept` can ask the mayor for now, by its DeptMoney.kind: the top-up its needs line names when it is
    first/supply or growth (the mayor funds those first, from the treasury above $3,000), or another building that
    fund_cover covers (the money above the reserve after them; never in a hold); else no ask, and why: another building
    that waits ("in a hold" / "doesn't cover it"), idle money, a first building blocked for lack of land (ask
    planning), or its DeptMoney.state ("funds cover your builds", "nothing ready to buy", "supply has room", "only a
    library ready ($1,500)"; planning's "your funds cover your road" / "no road is ready"), with _not_now's plan note,
    when to ask and what to do instead. The department's own DeptMoney decides (one observation, ~0.5 ms; its kind is
    funding_gaps' but for unclaimed first buildings, which read as their state); the whole funding_gaps (~7 ms) only
    for another building, whose cover needs every top-up. `m`: the department's DeptMoney when the caller has one (the
    briefing shares it with transport's Next line, so both read one first_build)."""
    f = None
    m = m or DeptMoney(env, dept)
    if needs_count(m):                  # its count is the mayor's (keep_up_counts): one line, one ask
        f = funding_gaps(env, fleet)
        m = f.depts.get(dept, m)
    k, what = m.kind, _yours(m.what)
    if k in NEED_KINDS[:2]:
        return Ask(True, m.top_up, what, "the mayor funds first buildings, supply and planning's road first")
    if k == "another":
        f = f or funding_gaps(env, fleet)
        if dept in fund_cover(f):
            return Ask(True, m.top_up, what, "the mayor's money above its reserve covers "
                       + ("them" if m.count > 1 else "it") + " now")
        why = ("the mayor is in a hold (the treasury is at or under its reserve) and funds only first buildings, "
               "supply and planning's road" if f.hold else "the mayor's money above its reserve doesn't cover it "
               "after first buildings, supply and planning's road")
        return _not_now(env, fleet, m, f"{what} ({money(m.top_up)} short) waits: {why}", what)
    if k == "idle":
        return _not_now(env, fleet, m, f"you hold {money(m.idle)} more than you need now")
    if k == "blocked":
        return _not_now(env, fleet, m, f"{' or '.join(m.blocked)} is blocked for lack of land"
                        + advice(": ask planning for a lot, not the mayor for money"))
    if dept == "planning":
        return _not_now(env, fleet, m, "your funds cover your road" if m.build is not None
                        and m.build.cost is not None else "no road is ready")
    return _not_now(env, fleet, m, _yours(m.state) or "nothing ready to buy")


def _not_now(env, fleet: FleetState, m: DeptMoney, why: str, what: str = "") -> Ask:
    """A no-ask, with the rest the department reads on its Your budget line and in send_message's hold-back: its plan
    target as a ceiling when above its funds (target_above), when an ask would be valid (supply room: once a supply
    will be BANK_BUSY used within AHEAD_MONTHS at its growth, the tightest named; nothing ready: once a site is) and
    what to do this turn instead (its next build when its funds cover it, no supply to build, a lot from planning,
    else only what its funds pay for). Region run 20261005 (trinity in both cities, 120 months): 125 money asks held
    back, every one under a budget line saying not to ask; in 51 the plan target was above the funds, and utilities
    read "budget target: $7,000" and "priority: expand water capacity" beside "supply has room" (28 held back)."""
    w, dept, b = env.world, m.dept, m.build
    target = fleet.plan["budgets"].get(dept)
    plan = (f"your {money(target)} plan target is the mayor's ceiling for you, not money to ask for"
            if target_above(target, m.funds) else "")
    room = b is not None and b.room and m.kind == "nothing"
    unready = m.kind == "nothing" and m.state == "nothing ready to buy" and dept != "planning"
    when = ""
    if room:
        sup = w.metrics()
        tight = max((s for s in SUPPLIES if sup[f"{s}_supply"] > 0),
                    key=lambda s: demand_ahead(w, s) / sup[f"{s}_supply"], default=None)
        when = (f"a supply will be {pct(100 * BANK_BUSY)} used within {months(AHEAD_MONTHS)} at its growth"
                + (f" (now {supply_ahead(w, tight)})" if tight else ""))
    elif unready:
        when = "a site is ready for you"
    if dept == "planning":                # its Next line: parks, zones, a lot for utilities, then the road
        instead = "the builds your briefing's Next line names"
    elif m.keep_ups and not short_of(m.funds, m.goal.cost):       # every building keeping its coverage up, where
        instead = (f"build {keep_up_where(m.goal)} ({money(m.goal.cost)}): your funds cover "
                   + ("them" if m.count > 1 else "it"))
    elif m.gap is not None:                                        # a building at a ready site would add nothing
        instead = f"ask planning for a lot or a road near the uncovered homes at ({m.gap[0]},{m.gap[1]})"
    elif b is not None and b.cost is not None and b.named and not short_of(m.funds, b.cost):
        call = b.named[0].call                                     # a tier upgrade: its exact call
        instead = (f"{call} ({_yours(b.what)}, {money(b.cost)})" if call else
                   f"build {_yours(b.what)} ({money(b.cost)})") + ": your funds cover it"
    elif b is not None and b.cost is None and b.covered:          # coverage / enrolment kept up (keep_up_room)
        instead = "nothing needs building now"
    elif room:
        instead = "no supply needs building yet"
    elif unready:
        instead = "nothing is ready to build"
    elif m.kind == "blocked":
        instead = "ask planning for a lot"
    else:
        instead = "build only what your funds pay for"
    return Ask(False, 0, what, why, plan, when, instead)


def ask_refusal(a: Ask) -> str:
    """send_message's hold-back of a money ask the mayor can't fund now (ToolBox._ask_refusal): why, the plan note,
    when to ask and what to do this turn instead (_not_now). Guided only: the open style holds no ask back."""
    return (f"not sent: {a.why}, so a money ask now gets a no" + (f"; {a.plan}" if a.plan else "") + ". "
            + (f"Ask when {a.when}. " if a.when else "")
            + f"Instead this turn: {a.instead}; put what you need in your end_turn report (the mayor reads it). To post "
            "this message anyway, send it to the mayor again.")


# the mayor's actions whose results say where the money stands (ToolBox._money_note, appended by _act on success and
# on a core failure, not _explain's fix), in the model's message only
MONEY_ACTIONS = ("allocate_budget", "take_loan", "repay_loan")
def money_docs() -> dict[str, str]:
    """What ToolBox.__init__ appends to the money tools' descriptions: what a loan costs in dollars, where the core doc
    gives a rate, read when asked like action_docs' rate (a LOAN_INTEREST override is applied when the env is built,
    after import). Review: it also said what the finance score reads, which no other LLM agent is told."""
    per = money(1000 * C.LOAN_INTEREST)
    return {"take_loan": f" Interest is {per} a month per $1,000 until repaid.",
            "repay_loan": f" Each $1,000 repaid saves {per} a month in interest."}


def treasury_state(w, f: Funding) -> str:
    """The treasury against the reserve, for a money result: "treasury $2,788, $212 under the $3,000 reserve"
    ("$2,000 above the", "at the"), with what sets the reserve after it (reserve_why)."""
    gap = round(f.spare)
    where = f"{money(gap)} above" if gap > 0 else f"{money(-gap)} under" if gap < 0 else "at"
    return f"treasury {cash(w.treasury)}, {where} the {money(f.reserve)} reserve{reserve_why(w)}"


def topped_up(m: DeptMoney) -> str:
    """What a department's money buys after a top-up, for its result: "can pay for another police station (upkeep
    +$55/month)" (its next build, first_build / road_build), "still $1,000 short of its first police station: less
    buys nothing" (its needs line; less buys fewer tiles of a per-tile need, fewer of utilities' supply buildings bought
    together, and part of planning's road and zoning, so none of them says nothing: review, $600 of a $900 levee
    stretch buys 2 of its 3 tiles), "blocked for lack of land:
    this buys nothing until planning frees a lot" (DeptMoney.blocked, or its funded first buildings unclaimed), or
    with nothing named "needs nothing now: supply has room" (FirstBuild.room), "had only a library ready ($1,500)"
    (FirstBuild.extra) or "had nothing ready to buy"."""
    if (b := m.needs) is not None:
        less = ("" if m.dept == "planning" else ": less buys fewer tiles" if any(n.tiles for n in b.named)
                else ": less buys fewer of them" if b.all_of and len(b.named) > 1 or m.count > 1
                else ": less buys nothing")
        return f"still {money(m.top_up)} short of {m.what}{less}"
    if m.unclaimed or m.blocked:
        return "blocked for lack of land: this buys nothing until planning frees a lot"
    b = m.goal                                  # its next build, or the buildings that keep its coverage up
    if b is None or b.cost is None or not b.named:
        if b is not None and b.room:
            return "needs nothing now: supply has room"
        if b is not None and b.extra:
            return f"had only {a_or_an(b.extra[0])} ready ({money(b.extra[1])})"
        return "had nothing ready to buy"
    up = b.upkeep
    return f"can pay for {b.what}" + (f" (upkeep +{money(up)}/month)" if round(up) else "")


def taken_back(m: DeptMoney) -> str:
    """What a department holds after a take-back against what it keeps (DeptMoney.keep), for its result: "planning now
    $1,240, what it keeps (nothing idle)", "... $508 over what it keeps" (": still idle" when it is), or "... $1,316
    below what it keeps (2 x its $658 spending)" (keep_why), then ": it can't pay for a road into open land and
    zoning it ($640)" when what is left is short of its next build (DeptMoney.cant_pay). Open style (fmt.py): the funds
    against the keep as a number, "police now $6,932, $868 below the $7,800 it keeps": no keep reason, no build."""
    over = round(m.funds - m.keep)
    head = f"{m.dept} now {money(m.funds)}, "
    if is_open():
        return head + (f"{money(over)} over" if over > 0 else f"{money(-over)} below" if over < 0 else "at") + \
            f" the {money(m.keep)} it keeps"
    if over > 0:
        return head + f"{money(over)} over what it keeps" + (": still idle" if m.idle else "")
    if over < 0:
        cant = m.cant_pay(m.funds)
        return head + f"{money(-over)} below what it keeps ({m.keep_why})" + (f": it can't pay for {cant}" if cant
                                                                              else "")
    return head + "what it keeps (nothing idle)"


class SiteList(NamedTuple):
    sites: list[dict]       # ranked, as build_observation lists them (find_sites sorts by distance and cuts)
    short: str | None       # why the list is empty or unusable (locked), in a few words: the briefing's line
    note: str | None        # the same in full, with what to do: find_sites' "note"
    blocked: bool = False   # empty for lack of land or roads (_why_empty): planning can fix it, and money for it
                            # isn't idle (DeptMoney)


def role_obs(env, role: str, site_hints: bool = True) -> dict:
    """`role`'s observation for the fleet's look-ups and briefings: no map, council messages or memory; with the
    site hints site_lists reads unless site_hints=False."""
    return build_observation(env.world, role, council_msgs=[], memory={}, scores={}, months_total=env.cfg.months,
                             include_map=False, site_hints=site_hints)


def site_lists(w, role: str, obs: dict) -> dict[str, SiteList]:
    """Every find_sites purpose of `role` from one build_observation(..., site_hints=True): the find_sites tool
    and the briefing's Ready sites block both read this, so they can't disagree."""
    out, strips = {}, None
    for purpose, path in SITE_PURPOSES[role].items():
        found = obs
        for key in path:
            found = found.get(key) if isinstance(found, dict) else None
        if path[0] == "zoning_suggestions" and found is not None:   # MAX_SITES strips per type, the first ones as
            if strips is None:                                      # observation's 2: each type its own (ready_lines)
                strips = _zone_suggestions(w, per_type=MAX_SITES)
            found = strips.get(path[1], found)
        if purpose == "bus_stops" and w.bus_lines:        # the next line: homes the lines don't reach
            found = new_line_stops(w)
        sites = [_site(s) for s in (found or []) if _on_connected_road(w, s)]
        if purpose in BANK_TOP_UP and len(sites) < len(found or []):
            sites += _bank_top_up(w, sites, BANK_TOP_UP[purpose])
        if purpose == "congested_roads":         # avenues already last (kept): an avenue upgrade charges nothing there
            sites = [s for s in sites if s.get("load_vs_capacity", 0) >= CONGESTED_LOAD]
            sites.sort(key=lambda s: not site_tiles(w, purpose, s)[0])
        keys, blocked = SITE_UNLOCKS.get(purpose, ()), False
        if keys and not any(w.unlocked(k) for k in keys):
            key = min(keys, key=C.UNLOCK_POP.get)
            short, note = f"locked (unlocks at {C.UNLOCK_POP[key]:,} residents)", "locked: " + w.unlock_note(key)
        elif w.metrics()["population"] < SITE_MIN_POP.get(purpose, 0):
            sites, (short, note) = [], SITE_NOTES[purpose]
        elif purpose == "bus_stops" and (room := bus_room(w, sites)):    # sites kept: find_sites still shows them
            short, note = room
        elif sites and (room := link_room(w, purpose)):                  # a station that wouldn't pay: sites kept
            short, note = room
        elif sites and purpose in EDU_LEVELS and not _wanted(w, purpose):  # a school level enrolled: sites kept
            short, note = edu_room(w, purpose)
        elif sites and (room := keep_up_room(w, role, purpose, sites)):   # coverage kept up: sites kept
            short, note = room
        elif sites:
            short = note = None
        elif purpose == "road_extension" and (branches := branch_sites(w)):    # no dead end: branch off a road
            sites, short, note = [b.site(w) for b in branches], branch_text(w, branches[0]), branch_note(w)
        elif purpose in SITE_NOTES:
            short, note = SITE_NOTES[purpose]
        else:                    # models told to "inspect the map" scanned it blindly for whole turns
            short, note, blocked = _why_empty(w, role, purpose)
        out[purpose] = SiteList(sites, short, note, blocked)
    return out


def _bank_top_up(w, listed: list[dict], line: str) -> list[dict]:
    """Free riverbank lots beside highway-connected roads (where a pump or outlet can go) not in `listed`, as many as
    fit in MAX_SITES with it: the observation cuts its riverbank lists to 6 before site_lists drops the sites off
    the highway network, so off-highway roads by the bank could leave pump and sewage short, or empty ("all
    taken"), with free lots beside the connected roads. Ranked as observation._candidate_sites ranks them for
    `line` (pump: cleanest first, +0.5 at or south of an outlet, then north first; outlet: those at or north of a
    pump last, then south first; crowded spots last, marked "crowd"), with pollution as it computes it."""
    have = {(s["x"], s["y"]) for s in listed}
    ys, xs = np.nonzero(free_bank(w))
    more = [{"x": x, "y": y, "pollution": round(max(float(w.pollution[y, x]), w._river_near(x, y)), 2)}
            for x, y in zip(xs.tolist(), ys.tolist()) if (x, y) not in have]
    if line == "pump":
        outlets = w.facilities(C.OUTLET) + w.facilities(C.TREATMENT)
        more.sort(key=lambda b: (b["pollution"] + (0.5 if any(oy <= b["y"] for _, oy in outlets) else 0), b["y"]))
    else:
        pumps = w.facilities(C.PUMP)
        more.sort(key=lambda b: (any(py >= b["y"] for _, py in pumps), -b["y"]))
    k = MAX_SITES - len(listed)
    return _uncrowded_first(w, more, line, k) if k > 0 else []


def _why_empty(w, role: str, purpose: str) -> tuple[str, str, bool]:
    """(short, full, blocked): why a site list is empty and what to do: no free roadside land (roads must grow), no
    road at the river or its bank lots taken, no free roadside lot away from homes (AWAY_FROM_HOMES purposes), no free
    2x2 roadside block (TWO_BY_TWO purposes), or nothing needs the building yet (the lists only rank spots near
    uncovered homes / jobs). `blocked`: for lack of land or roads, which planning can fix (not a missing river, or
    nothing needing it). Review probe, month 60 (6,564 residents): the university list read "nothing needs it right
    now" with no free 2x2 block, and education's $12,000 keep for it read idle."""
    roads = lambda: _roads_text(_roads(w))
    if is_open():
        return _why_empty_open(w, role, purpose)
    if purpose == "road_extension":                  # and no branch either (site_lists offers those first)
        return ("no straight extension from a dead end: build_road from any road into free land",
                "no straight extension from a dead end into open land; build_road from any road tile into free "
                f"land (L-shaped roads work, bridges cross the river; {roads()})", False)
    if purpose in RIVERBANK_PURPOSES:
        river = _river_cols(w)
        if river is None:
            return "no river on this map", "this map has no river", False
        lots, contact = _riverbank_by_roads(w)
        near = ", ".join(f"({x},{y})" for x, y in _ends(contact))
        if not contact:
            return (f"no connected road reaches the river yet: ask planning for a road to it (river at x "
                    f"{_span(river[0], river[1])})",
                    f"no free riverbank tile next to a highway-connected road: a road has to reach the river first "
                    f"({_river_span(w)}); ask planning (send_message) for a road to the riverbank, and build what "
                    "doesn't need the river meanwhile", True)
        free = [(int(x), int(y)) for y, x in zip(*np.nonzero(lots & _empty_land(w)))]
        if purpose == "riverbank_2x2" and free:
            return (f"no free 2x2 riverbank block by a road: ask planning to leave one near {near}",
                    f"no free 2x2 block on the riverbank beside a highway-connected road (free bank lots are single "
                    f"tiles, e.g. ({free[0][0]},{free[0][1]})); ask planning (send_message) to leave a 2x2 riverbank "
                    f"lot beside the road near {near}", True)
        return (f"riverbank beside roads all taken: ask planning to free a lot near {near}",
                "the river bank tiles next to highway-connected roads are all taken (zoned or built): ask planning "
                f"(send_message) to leave or free (dezone) a riverbank lot beside the road (e.g. near {near}, where "
                "the road meets the river), and build what doesn't need the river meanwhile", True)
    if purpose == "train_station" and link_due(w):        # the first link waits for a road to the map edge
        return ("no free map-edge lot by a connected road: ask planning for a road to the map edge",
                "a train station goes on an empty tile at the map edge beside a highway-connected road, and none is "
                f"free: without a train station or harbour R and C demand stay capped from {C.CAP_POP_CONNECTION:,} "
                f"residents; ask planning (send_message) to extend a road to the map edge and leave the edge tile "
                f"beside it free ({roads()})", True)
    free = _free_roadside(w)
    if role == "planning":
        return ("no free roadside strip: extend the roads (find_sites road_extension)",
                f"no free strip along a highway-connected road ({roads()}; {free} free roadside tiles); "
                "extend the roads first (find_sites road_extension lists dead ends to build from)", True)
    near = (role, purpose) in NEAR_NEED
    need = _uncovered_need(w, role, purpose) if near else None
    if near and need is None:            # well covered: no lot needed, whether or not there is land
        return ("nothing needs it right now",
                f"nothing needs this right now: under {NEED_SHARE:.0%} of the {NEAR_NEED[role, purpose][0]} it would "
                f"serve are poorly covered (coverage under {NEED_COVERED}); check coverage in city_overview / "
                "my_facilities before building", False)
    if not free:
        return ("no free roadside land: ask planning for a lot",
                f"no free land next to a highway-connected road ({roads()}); ask planning (send_message) to extend a "
                "road or leave an empty lot by one", True)
    if purpose in AWAY_FROM_HOMES and not (_roadside(w) & ~_far_from_homes(w)).any():
        return ("no roadside lot away from homes (3+ within 4 tiles): ask planning for one at the town edge",
                f"all {free} free roadside tiles have 3 or more homes within 4 tiles, and a {purpose} must stay away "
                "from homes; ask planning (send_message) to leave an empty lot by a road at the edge of town, away "
                f"from the houses, or to extend a road into open land ({roads()})", True)
    if purpose in TWO_BY_TWO and not w.site_candidates(np.ones(w.struct.shape), radius=1, k=1, size=2):
        return ("no free 2x2 roadside block: ask planning for one",
                f"no free 2x2 block of empty land beside a highway-connected road ({roads()}; {free} free roadside "
                f"tiles, none in a free 2x2 block), and a {purpose} takes 2x2 tiles; ask planning (send_message) to "
                "leave a 2x2 lot by a road, or to extend a road into open land", True)
    if need is not None:
        (x, y), share = need
        what = NEAR_NEED[role, purpose][0]
        return (f"no free roadside lot near the homes that need it (around ({x},{y})): ask planning for one",
                f"no free roadside tile within {NEAR_NEED[role, purpose][2]} tiles of the uncovered {what} "
                f"({share:.0%} of them poorly covered, most around ({x},{y})); the {free} free roadside tiles are all "
                f"farther away: ask planning (send_message) to leave or free (dezone) a lot by a road near ({x},{y}), "
                f"or to extend a road there ({roads()})", True)
    return ("nothing needs it right now",
            f"nothing needs this right now: no uncovered homes or jobs near the {free} free roadside tiles "
            "(check coverage in city_overview / my_facilities before building)", False)


OPEN_BLOCKED = "blocked: "      # an open-style reason for lack of land or roads starts so (fleet/report.py reads it)


def _why_empty_open(w, role: str, purpose: str) -> tuple[str, str, bool]:
    """_why_empty in the open briefing style (fmt.py): the same cases and `blocked`, the facts without what to do
    or whom to ask; a reason for lack of land or roads starts with OPEN_BLOCKED."""
    roads = lambda: _roads_text(_roads(w))
    b = lambda short, note: (OPEN_BLOCKED + short, note, True)
    if purpose == "road_extension":
        return ("no straight extension from a dead end", "no straight extension from a dead end into open land "
                f"(build_road takes L-shaped roads; bridges cross the river; {roads()})", False)
    if purpose in RIVERBANK_PURPOSES:
        river = _river_cols(w)
        if river is None:
            return "no river on this map", "this map has no river", False
        lots, contact = _riverbank_by_roads(w)
        near = ", ".join(f"({x},{y})" for x, y in _ends(contact))
        if not contact:
            return b(f"no connected road reaches the river yet (river at x {_span(river[0], river[1])})",
                     f"no free riverbank tile next to a highway-connected road: no such road reaches the river "
                     f"({_river_span(w)})")
        free = [(int(x), int(y)) for y, x in zip(*np.nonzero(lots & _empty_land(w)))]
        if purpose == "riverbank_2x2" and free:
            return b(f"no free 2x2 riverbank block by a road (roads meet the river near {near})",
                     f"no free 2x2 block on the riverbank beside a highway-connected road (free bank lots are single "
                     f"tiles, e.g. ({free[0][0]},{free[0][1]}); roads meet the river near {near})")
        return b(f"riverbank beside roads all taken (zoned or built; roads meet the river near {near})",
                 f"the river bank tiles next to highway-connected roads are all taken (zoned or built); roads meet the "
                 f"river near {near}")
    if purpose == "train_station" and link_due(w):
        return b("no free map-edge lot by a connected road",
                 "a train station goes on an empty tile at the map edge beside a highway-connected road, and none is "
                 f"free; without a train station or harbour R and C demand stay capped from {C.CAP_POP_CONNECTION:,} "
                 f"residents ({roads()})")
    free = _free_roadside(w)
    if role == "planning":
        return b("no free roadside strip", f"no free strip along a highway-connected road ({roads()}; {free} free "
                 "roadside tiles)")
    near = (role, purpose) in NEAR_NEED
    need = _uncovered_need(w, role, purpose) if near else None
    if near and need is None:
        return ("nothing needs it right now",
                f"nothing needs this right now: under {NEED_SHARE:.0%} of the {NEAR_NEED[role, purpose][0]} it would "
                f"serve are poorly covered (coverage under {NEED_COVERED})", False)
    if not free:
        return b("no free roadside land", f"no free land next to a highway-connected road ({roads()})")
    if purpose in AWAY_FROM_HOMES and not (_roadside(w) & ~_far_from_homes(w)).any():
        return b("no roadside lot away from homes (3+ within 4 tiles)",
                 f"all {free} free roadside tiles have 3 or more homes within 4 tiles, and a {purpose} must stay away "
                 f"from homes ({roads()})")
    if purpose in TWO_BY_TWO and not w.site_candidates(np.ones(w.struct.shape), radius=1, k=1, size=2):
        return b("no free 2x2 roadside block",
                 f"no free 2x2 block of empty land beside a highway-connected road ({roads()}; {free} free roadside "
                 f"tiles, none in a free 2x2 block), and a {purpose} takes 2x2 tiles")
    if need is not None:
        (x, y), share = need
        what = NEAR_NEED[role, purpose][0]
        return b(f"no free roadside lot near the homes that need it (around ({x},{y}))",
                 f"no free roadside tile within {NEAR_NEED[role, purpose][2]} tiles of the uncovered {what} "
                 f"({share:.0%} of them poorly covered, most around ({x},{y})); the {free} free roadside tiles are all "
                 f"farther away ({roads()})")
    return ("nothing needs it right now",
            f"nothing needs this right now: no uncovered homes or jobs near the {free} free roadside tiles", False)


def _served(w, what: str) -> np.ndarray:
    """What a NEAR_NEED building serves per tile, as World._compute_services counts its demand."""
    if what == "people":
        return w._people_grid()
    if what == "buildings":
        return (np.isin(w.struct, C.ZONES) & (w.level > 0)).astype(float)
    res = w.residents_grid()
    if what == "residents":
        return res
    n = w.cohorts
    return res * float(n[0 if what == "children" else 1].sum()) / max(1.0, float(n.sum()))


def _uncovered_need(w, role: str, purpose: str) -> tuple[tuple[int, int], float] | None:
    """For a NEAR_NEED list with real need (at least NEED_SHARE of what it serves under NEED_COVERED coverage): the
    tile a building would serve the most of that from (within the list's radius) and the share; else None. Smoke v14
    seed 0, months 3-12: planning zoned the lots near homes after the mayor funded fire, health and education, and all
    three read "nothing needs it right now" at coverage 0 (16 such turns in v9-v14); fire held $2,500 to month 12."""
    if (role, purpose) not in NEAR_NEED:
        return None
    what, grid, radius = NEAR_NEED[role, purpose]
    served = _served(w, what)
    uncovered = served * (getattr(w, grid) < NEED_COVERED)
    total, left = float(served.sum()), float(uncovered.sum())
    if total <= 0 or left < NEED_SHARE * total:
        return None
    y, x = np.unravel_index(int(np.argmax(box_sum(uncovered, radius))), uncovered.shape)
    return (int(x), int(y)), left / total


def _tiles_text(tiles) -> str:
    """"(20,23)", or a straight run's ends "(20,22)-(20,23)"."""
    (xa, ya), (xb, yb) = min(tiles), max(tiles)
    return f"({xa},{ya})" + (f"-({xb},{yb})" if (xa, ya) != (xb, yb) else "")


class Branch(NamedTuple):
    """A road to branch off the highway-connected network into open land (branch_sites): off road tile `road`, going
    `way`, straight through `clear` (the zoned tiles beside the road to free first, nearest first; none from a free
    lot) and `run` (empty land or river, ending on land)."""
    road: tuple[int, int]
    way: tuple[int, int]
    clear: tuple[tuple[int, int], ...]
    run: tuple[tuple[int, int], ...]
    opens: int              # empty land tiles without road access within 2 tiles of it (observation's opens_land)
    built: int              # developed tiles among `clear`
    levels: int             # their levels summed

    @property
    def start(self) -> tuple[int, int]:
        return (self.clear or self.run)[0]

    @property
    def end(self) -> tuple[int, int]:
        return self.run[-1]

    @property
    def tiles(self) -> tuple[tuple[int, int], ...]:
        return self.clear + self.run

    def first(self, w) -> str:
        """What to run before the road, "" from a free lot: "dezone (20,23) (empty lot, free)", or with a building
        "bulldoze (20,23) (a level-1 home, $5)" (bulldoze takes the lots with it)."""
        if not self.clear:
            return ""
        where, n = _tiles_text(self.clear), len(self.clear)
        if not self.built:
            return f"dezone {where} (empty lot{'s' * (n > 1)}, free)"
        what = Counter(f"level-{int(w.level[y, x])} {BUILT_NAMES[C.LETTER_BY_ZONE[int(w.struct[y, x])]]}"
                       if w.level[y, x] > 0 else "empty lot" for x, y in sorted(self.clear))   # as `where` reads
        plural = lambda k: k[:-1] + "ies" if k.endswith("y") else k + "s"
        items = [a_or_an(k) if c == 1 else f"{c} {plural(k)}" for k, c in what.items()]
        return f"bulldoze {where} ({_and(items)}, {money(n * C.CATALOG['bulldoze']['cost'])})"

    def steps(self, w) -> str:
        """"dezone (20,23) (empty lot, free), then build_road (20,23)-(20,10)"."""
        road = f"build_road ({self.start[0]},{self.start[1]})-({self.end[0]},{self.end[1]})"
        first = self.first(w)
        return f"{first}, then {road}" if first else road

    def site(self, w) -> dict:
        """The find_sites entry: the build_road's ends (x1, y1 the tile beside the road), its tiles and bridge tiles,
        the land it opens, the road tile it branches off and what to run first; cost: all of it, as charged."""
        (x1, y1), (x2, y2) = self.start, self.end
        bridges = sum(bool(w.terrain[y, x] == C.WATER) for x, y in self.tiles)
        cost = ((len(self.tiles) - bridges) * C.CATALOG["road"]["cost"] + bridges * C.CATALOG["bridge"]["cost"]
                + (len(self.clear) * C.CATALOG["bulldoze"]["cost"] if self.built else 0))
        out = {"x": x1, "y": y1, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "tiles": len(self.tiles),
               "bridge_tiles": bridges, "opens_land": self.opens, "branch_from": list(self.road), "cost": cost}
        return {**out, "first": self.first(w)} if self.clear else out


def branch_sites(w, clear: int = BRANCH_CLEAR, k: int = MAX_SITES) -> list[Branch]:
    """Branch roads into open land, for where no dead end extends (observation's road_suggestions empty): from each
    highway-connected road tile (not the highway entry), each way its neighbour isn't road, through up to `clear` zoned
    tiles and then at least BRANCH_MIN tiles of empty land or river (trimmed to end on land; starting on a lot, so not
    down the river from a bridge), BRANCH_LEN tiles in all at most, opening at least BRANCH_OPENS land tiles (as
    observation counts opens_land: empty land without road access within 2 tiles of it). Ranked: none to bulldoze (a
    free lot or empty lots to dezone) by land opened, then fewer tiles to clear, then nearer the town hall; then those
    through buildings, the fewest levels first. Up to `k`, those running the same way more than BRANCH_APART apart.
    Smoke v17 seed 0, months 3-9: one road across the map (y=24 x 0-47), every tile beside it zoned or built, and
    planning, told "no straight extension from a dead end", built into buildings and past the map edge for 3 turns."""
    s, level = w.struct.tolist(), w.level.tolist()
    wet = (w.terrain == C.WATER).tolist()
    band = (w.struct == C.NONE) & (w.terrain == C.LAND) & ~w.access
    area = np.pad(band.cumsum(0).cumsum(1), ((1, 0), (1, 0))).tolist()     # summed-area table of `band`
    zones, W, H = set(C.ZONES), w.w, w.h
    found = []
    ys, xs = np.nonzero(_connected_roads(w) & (w.struct != C.HIGHWAY))
    for x, y in zip(xs.tolist(), ys.tolist()):
        for dx, dy in DIRECTIONS:
            cx, cy, cleared, run, land = x + dx, y + dy, [], [], 0
            while len(cleared) < clear and 0 <= cx < W and 0 <= cy < H and s[cy][cx] in zones:
                cleared.append((cx, cy))
                cx, cy = cx + dx, cy + dy
            while len(cleared) + len(run) < BRANCH_LEN and 0 <= cx < W and 0 <= cy < H and s[cy][cx] == C.NONE:
                run.append((cx, cy))
                land = land if wet[cy][cx] else len(run)
                cx, cy = cx + dx, cy + dy
            run = run[:land]
            if len(run) < BRANCH_MIN or not cleared and wet[run[0][1]][run[0][0]]:    # a lot, not the river
                continue
            (xa, ya), (xb, yb) = (cleared or run)[0], run[-1]
            x0, x1 = max(0, min(xa, xb) - 2), min(W - 1, max(xa, xb) + 2)
            y0, y1 = max(0, min(ya, yb) - 2), min(H - 1, max(ya, yb) + 2)
            opens = area[y1 + 1][x1 + 1] - area[y0][x1 + 1] - area[y1 + 1][x0] + area[y0][x0]
            if opens < BRANCH_OPENS:
                continue
            built = [level[by][bx] for bx, by in cleared if level[by][bx] > 0]
            found.append(Branch((x, y), (dx, dy), tuple(cleared), tuple(run), int(opens), len(built), sum(built)))
    tx, ty = w.townhall
    found.sort(key=lambda b: (b.built > 0, b.levels, -b.opens, len(b.clear),
                              abs(b.start[0] - tx) + abs(b.start[1] - ty), b.start[1], b.start[0]))
    out = []
    for b in found:
        if not any(o.way == b.way and abs(o.start[0] - b.start[0]) + abs(o.start[1] - b.start[1]) <= BRANCH_APART
                   for o in out):
            out.append(b)
            if len(out) >= k:
                break
    return out


def branch_text(w, b: Branch) -> str:
    """The Ready sites line's reason for road_extension with branch roads (open style: the road and what is in its
    way, no steps)."""
    if is_open():
        way = f" through {b.first(w).split(' ', 1)[1]}" if b.clear else ""
        return (f"no dead end; branch road ({b.start[0]},{b.start[1]})-({b.end[0]},{b.end[1]}){way}, opens {b.opens} "
                "land tiles")
    return f"no dead end; branch: {b.steps(w)}, opens {b.opens} land tiles"


def branch_note(w) -> str:
    """find_sites' note on road_extension with branch roads."""
    if is_open():
        return ("no straight extension from a dead end into open land, so these are branch roads: each leaves a road "
                "tile (branch_from) at x1,y1 and runs straight to x2,y2; 'first' names the zoned lots in its way "
                f"(dezoning is free; bulldozing a building {money(C.CATALOG['bulldoze']['cost'])} a tile; "
                f"{_roads_text(_roads(w))})")
    return ("no straight extension from a dead end into open land, so these are branch roads, best first: each leaves "
            "a road tile (branch_from) at x1,y1 and runs straight to x2,y2; run its 'first' action if it has one "
            f"(dezone empty lots: free; bulldoze a building: {money(C.CATALOG['bulldoze']['cost'])} a tile), then "
            f"build_road x1,y1 to x2,y2 ({_roads_text(_roads(w))})")


def ways_out(w) -> list[tuple]:
    """Where a new road can still reach open land without clearing a lot, best first: the tiles of each straight
    extension from a dead end (observation's road_suggestions), else of each branch from a free lot (branch_sites with
    nothing to clear); [] when planning has boxed itself in (a zone that leaves none warns)."""
    sign = lambda v: (v > 0) - (v < 0)
    ext = [tuple((r["x1"] + i * sign(r["x2"] - r["x1"]), r["y1"] + i * sign(r["y2"] - r["y1"]))
                 for i in range(1, r["tiles"] + 1)) for r in _road_extensions(w)]
    return ext or [b.run for b in branch_sites(w, clear=0)]


def _brief(v) -> str:
    return f"{round(v):,}" if abs(v) >= 10 else f"{round(float(v), 2):g}"


def _where(s: dict) -> str:
    return f"({s['x1']},{s['y1']})-({s['x2']},{s['y2']})" if "x2" in s else f"({s['x']},{s['y']})"


def _extra(purpose: str, s: dict) -> str:
    for k, fmt in SITE_FIELDS:
        if k in s and k not in SITE_FIELDS_HIDDEN.get(purpose, ()):
            return "" if purpose in SITE_FIELDS_NONZERO and not s[k] else ", " + fmt.format(_brief(s[k]))
    return ""


def scarce_bank(w) -> set[tuple[int, int]]:
    """The free riverbank lots beside highway-connected roads (where a pump or outlet can go) when there are at most
    SCARCE_BANK of them, else none: the Ready sites line of another purpose whose site is one of them says to keep
    it for pump / sewage."""
    ys, xs = np.nonzero(free_bank(w))
    return {(int(x), int(y)) for y, x in zip(ys, xs)} if len(xs) <= SCARCE_BANK else set()


def _price(w, name: str) -> float | None:
    """What `name` (a tier name or an OFF_TIER_SPECS key) costs: what its action charges. None while it is locked."""
    if name in OFF_TIER_SPECS:
        cost, key = OFF_TIER_SPECS[name]
        return cost() if key is None or w.unlocked(key) else None
    line, tier = C.TIER_BY_NAME[name]
    spec = C.TIER_LINES[line][tier - 1]
    return None if spec.get("unlock") and not w.unlocked(spec["unlock"]) else spec["cost"]


def upkeep_of(w, dept: str, name: str) -> float:
    """The monthly upkeep of what `name` (a tier name or an OFF_TIER_SPECS key) builds, as `dept` would pay it: its
    tier spec's "upkeep" (OFF_TIER_UPKEEP off the tier lines; per tile for PER_TILE names), read when asked so balance
    overrides (TIERS.*) reach it, times the department's funding level as World.dept_upkeep applies it (50% funding:
    half). 0 for anything else. Smoke v13-v19: the needs lines gave the price only; in the 120-month run service
    upkeep was a median 34% of income."""
    if name in OFF_TIER_UPKEEP:
        base = OFF_TIER_UPKEEP[name]()
    elif name in C.TIER_BY_NAME:
        line, tier = C.TIER_BY_NAME[name]
        base = C.TIER_LINES[line][tier - 1]["upkeep"]
    else:
        return 0.0
    return float(base) * w.funding.get(dept, 1.0)


def building_name(name: str) -> str:
    """A SITE_BUILDS name for people: a line's base tier by its struct's name (tower -> water tower, outlet -> sewage
    outlet), a higher tier by its own (water works: the struct is the line's), the others as they are (bus line)."""
    if name in C.TIER_BY_NAME:
        line, tier = C.TIER_BY_NAME[name]
        name = C.STRUCT_NAME[C.TIER_LINES[line][0]["struct"]] if tier == 1 else name
    return name.replace("_", " ")


def build_purpose(role: str, action: str, args: dict) -> str | None:
    """`role`'s find_sites purpose for what a build action asks for: its tier name (the "kind", else the building the
    action is named after; "tier" picks another of its line) in SITE_BUILDS, else another tier of its line there,
    else the action's building when that is a purpose itself (build_casino); None when none fits."""
    builds, own = SITE_BUILDS.get(role, {}), action.removeprefix("build_")
    kind = _kind_name(args.get("kind"))
    if action == "build_school":
        kind = SCHOOL_KINDS.get(kind or "elementary", kind)
    name = kind if kind in C.TIER_BY_NAME else own
    line = C.TIER_BY_NAME.get(name, (None,))[0]
    try:
        tier = int(args["tier"]) if line and args.get("tier") is not None else None
    except (TypeError, ValueError):
        tier = None
    if tier and 1 <= tier <= len(C.TIER_LINES[line]):
        name = C.TIER_LINES[line][tier - 1]["name"]
    exact = [p for p, names in builds.items() if name in names]
    same = [p for p, names in builds.items() if line and any(C.TIER_BY_NAME.get(n, (None,))[0] == line for n in names)]
    return next(iter(exact + same + [own] * (own in SITE_PURPOSES.get(role, {}))), None)


class Price(NamedTuple):
    cost: float             # as the action charges it, unrounded (one tile for PER_TILE names)
    building: str           # what it buys, for people (building_name: "water tower", "bus line", "levee tile")
    per_tile: bool = False
    first: bool = True      # a line's base tier or an off-tier spec: a first building (not a clinic's hospital, nor
                            # a NOT_FIRST one: the university)
    size: tuple[int, int] = (1, 1)      # its footprint (the tier's, as C.footprint gives it): a university 2x2
    name: str = ""          # the SITE_BUILDS name pricing it (a tier name or an OFF_TIER_SPECS key)
    zoning: float = 0.0     # planning's road (road_price): zoning the land it opens, included in `cost`, shown apart
    upkeep: float = 0.0     # a month, at the department's funding level (upkeep_of; per tile like `cost`); site_price
                            # fills it, planning's road has none

    def text(self) -> str:
        if self.zoning:                         # "$50 road + $75 zoning"
            return f"{money(self.cost - self.zoning)} road + {money(self.zoning)} zoning"
        return money(self.cost) + ("/tile" if self.per_tile else "")


def short_of(funds: float, cost: float) -> bool:
    """`funds` can't pay `cost`, as actions._spend decides (it lets float dust through)."""
    return funds + 1e-9 < cost


def site_price(w, role: str, purpose: str) -> Price | None:
    """What a site on `role`'s find_sites `purpose` list costs: the cheapest unlocked thing the list is for
    (SITE_BUILDS), as its action charges it (per tile for PER_TILE names). None for purposes without a price or while
    all of theirs are locked."""
    priced = [(c, n) for n in SITE_BUILDS.get(role, {}).get(purpose, ()) if (c := _price(w, n)) is not None]
    best = min(priced, key=lambda t: t[0], default=None)
    if best is None:
        return None
    tier = C.TIER_BY_NAME.get(best[1])
    return Price(best[0], building_name(best[1]), per_tile=best[1] in PER_TILE,
                 first=(tier is None or tier[1] == 1) and best[1] not in NOT_FIRST,
                 size=tuple(C.TIER_LINES[tier[0]][tier[1] - 1]["footprint"]) if tier else (1, 1), name=best[1],
                 upkeep=upkeep_of(w, role, best[1]))


def site_tiles(w, purpose: str, site: dict) -> tuple[int, int]:
    """(tiles, charged units) of one site of a per-tile purpose built whole: a congested stretch's roads below avenue
    class, as upgrade_road charges them (BRIDGE_UPGRADE units on a bridge; none on a stretch of avenues), else the
    site's "tiles" (a levee run: empty bank land, one unit each), one when it doesn't say."""
    if purpose == "congested_roads":
        xs = range(min(site["x1"], site["x2"]), max(site["x1"], site["x2"]) + 1)
        ys = range(min(site["y1"], site["y2"]), max(site["y1"], site["y2"]) + 1)
        road = [(x, y) for y in ys for x in xs
                if w.struct[y, x] == C.ROAD and w.road_class[y, x] < C.ROAD_CLASSES["avenue"]]
        return len(road), sum(BRIDGE_UPGRADE if w.terrain[y, x] == C.WATER else 1 for x, y in road)
    n = max(1, int(site.get("tiles", 1)))
    return n, n


def best_stretch(w, purpose: str, sites: list[dict]) -> tuple[int, int]:
    """site_tiles of the best (first) site, (0, 0) when it has nothing to charge: site_lists ranks congested
    stretches that are avenues already last (seed 0's most congested roads were by month 60), so a first one with
    nothing to charge means none has, and first_build leaves the purpose out (review: it priced "an avenue tile
    ($30)" and the Next line's upgrade_road failed "already avenue")."""
    return site_tiles(w, purpose, sites[0])


def ready_prices(w, role: str, lists: dict[str, SiteList]) -> dict[str, Price]:
    """site_price of each of `role`'s priced purposes with ready sites (listed and unlocked): the Ready sites lines
    end with it, and funds short of every one buy no ready site."""
    return {p: pr for p, lst in lists.items() if lst.sites and not lst.short and (pr := site_price(w, role, p))}


def cheapest(prices: dict[str, Price], supplementary: bool = False) -> tuple[tuple[str, ...], Price] | None:
    """(purposes, price) of the lowest price, SUPPLEMENTARY purposes skipped unless `supplementary` (the mayor's
    needs line and the budget line name them); None when there is none. Every purpose at that price, in order:
    health's clinic and landfill both cost $2,000, and naming the first alone would pick the clinic over the
    playbook's landfill-first."""
    kept = {p: pr for p, pr in prices.items() if supplementary or p not in SUPPLEMENTARY}
    low = min(kept.values(), key=lambda pr: pr.cost, default=None)
    return None if low is None else (tuple(p for p, pr in kept.items() if pr.cost == low.cost), low)


class Need(NamedTuple):
    """One building the mayor's needs line names (first_build): "its first police station (5 sites, $55/month)" while
    the city has none (buildings_of), "another police station (coverage 74%, $55/month)" once it has one, "3 avenue
    tiles (1 site, $4/month)" for what isn't counted; planning's road has no upkeep: "a road into open land and
    zoning it (opens 71 tiles)"."""
    what: str                   # the building ("police station"); uncounted, whole ("a levee tile", "3 avenue tiles")
    sites: int                  # ready sites of the first purpose pricing it (up to MAX_SITES)
    have: int | None            # buildings_of; None when not counted
    status: str = ""            # for another one: its coverage or supply (build_status), "" for none; a utilities
                                # supply building, first or another, its supply's cover (SupplyBuild.status)
    first: bool = False         # a primary first building (Price.first) the city lacks: the mayor's priority line
    upkeep: float = 0.0         # what it adds a month (Price.upkeep; per-tile ones for all their tiles); 0 for the road
    count: int = 1              # how many: more than one only to keep coverage up (DeptMoney.goal), `upkeep` for all
    call: str = ""              # the exact call when it is no site build (a tier upgrade: Upgrade.call)
    note: str = ""              # for the department's line only: what the alternative would do (Upgrade.alt)
    tiles: bool = False         # a per-tile build (avenue or levee tiles): less money buys fewer of them

    @property
    def label(self) -> str:
        """What it is, without the details: "its first police station", "another clinic", "3 avenue tiles", "3 more
        fire stations" (count)."""
        if self.have is None:
            return self.what
        if self.count > 1:
            return f"{self.count} more {self.what}s"
        return f"its first {self.what}" if not self.have else f"another {self.what}"

    def text(self) -> str:
        """label, then its ready sites (a first one; also another or an uncounted one without a status) or status,
        and its upkeep when that rounds to $1 or more, in parentheses."""
        sites = f"{self.sites} site{'s' * (self.sites != 1)}"
        up = f", {money(self.upkeep)}/month" if round(self.upkeep) else ""
        return f"{self.label} ({sites if self.have == 0 else self.status or sites}{up})"


LABEL_HEADS = ("its first ", "another ")


def labels_and(labels: list[str]) -> str:
    """Need labels all bought together, a run of the same head said once: "its first coal plant, water pump and sewage
    outlet", "another coal plant and its first sewage outlet"."""
    out, prev = [], None
    for lab in labels:
        head = next((h for h in LABEL_HEADS if lab.startswith(h)), None)
        out.append(lab[len(head):] if head and head == prev else lab)
        prev = head
    return _and(out)


class FirstBuild(NamedTuple):
    cost: float | None          # the cheapest ready site (cheapest), a per-tile purpose's best stretch whole; None
                                # when none is ready
    named: tuple[Need, ...]     # what it buys, one per building at that price
    dearest: float              # the dearest ready or blocked first building (Price.first, SUPPLEMENTARY skipped), at
                                # least `cost`: what it keeps
    blocked: tuple[str, ...]    # when none is ready, the cheapest blocked first building(s) ("a police station")
    funded: dict | None = None  # its funded_firsts buildings as funded_sites gives them (purpose -> price and ready
                                # sites): unclaimed_firsts claims them across departments
    keep_for: str = ""          # what `dearest` buys, as its Need label ("its first university", "another fire
                                # station", "5 avenue tiles"): DeptMoney.keep_why names it
    room: bool = False          # with nothing else ready or blocked: ready utilities sites left out because their
                                # supply has room (_wanted): "supply has room" (DeptMoney.state)
    extra: tuple[str, float] | None = None  # with nothing else ready or blocked: the cheapest ready SUPPLEMENTARY
                                            # building and its price ("only a library ready ($1,500)")
    all_of: bool = False        # `named` are bought together, `cost` their sum: utilities' one building per short
                                # supply (short_supplies); else each is one way to spend `cost` (a tie)
    waits: tuple[str, ...] = () # utilities: each short supply's building with no ready site, and what it waits for
                                # ("sewage outlet waits for a road to the river"): the mayor's needs line adds them
    purposes: tuple[str, ...] = ()  # the find_sites purpose of each `named` building, in order; with nothing named,
                                    # `extra`'s (transport's Next line renders the build from them: _transport_next)
    sites: tuple = ()           # for each `named` building of a KEEP_UP purpose, its ready sites ((x, y), best first,
                                # up to MAX_SITES), else (): keep_up picks among them (DeptMoney.keep_ups)
    keep_up: bool = False       # more than one building, to bring coverage back (DeptMoney.goal): `named` are bought
                                # together (all_of when several), each with its count and coverage
    covered: str = ""           # with nothing named: what is covered already, its purposes left out (education's
                                # "enrolment covered: elementary 99%, high-school 92%"; police, fire and health's
                                # keep_up_room shorts: "fire coverage 97% (goal 90%)", "ready sites add under 1%
                                # (health coverage 86%)"): DeptMoney.state
    gap: tuple[int, int] | None = None  # with `covered` saying no ready site adds coverage: where the uncovered homes
                                        # are (KeepUp.gap), for DeptMoney.gap
    stuck: str = ""             # and what it says, with where the coverage is lost (stuck_note): "ready sites add under
                                # 1% (fire coverage 80%): the uncovered homes near (7,9) are out of their road reach"
                                # (the department's keep-up line)

    @property
    def what(self) -> str:
        """What it buys, without the details: "its first police station", "another clinic or another landfill" (a
        tie), every building bought together "its first coal plant and water tower" (all_of)."""
        labels = [n.label for n in self.named]
        return labels_and(labels) if self.all_of else " or ".join(labels)

    @property
    def upkeep(self) -> float:
        """What it adds a month: the buildings' upkeep summed when bought together (all_of), else the largest."""
        ups = [n.upkeep for n in self.named]
        return sum(ups) if self.all_of else max(ups, default=0.0)

    def text(self) -> str:
        """The mayor's needs line after "needs $X for ": each Need's text, " or " between them; several bought
        together, the labels and the upkeep summed: "its first coal plant and water tower ($110/month)". Their
        supplies' state is left to the City block's supply / demand line: with each one's (e.g. "power 25% covered")
        the worst mix of test_briefings_stay_bounded_at_the_limits took the mayor's briefing to 7,699-7,707 of its
        7,700."""
        if not self.all_of or len(self.named) < 2:
            return " or ".join(n.text() for n in self.named)
        if self.keep_up:                        # each with its coverage: "2 more clinics (...) and another landfill"
            return " and ".join(n.text() for n in self.named)
        return self.what + (f" ({money(self.upkeep)}/month)" if round(self.upkeep) else "")


def supply_of(name: str) -> str | None:
    """"power", "water" or "sewage" for a tier name on a utilities line (a line's name is its tier 1), else None."""
    line = C.TIER_BY_NAME.get(name, (None,))[0]
    return next((s for s, lines in (("power", POWER_LINES), ("water", WATER_LINES), ("sewage", SEWAGE_LINES))
                 if line in lines), None)


def buildings_of(w, name: str) -> int | None:
    """How many buildings the city has of what the SITE_BUILDS `name` builds: its tier line's at that tier or above
    (a hospital counts as a clinic, a clinic not as a hospital), for a utilities building every one making its supply
    (a pump makes a water tower another one), bus lines by count; None for tiles and off-tier stations."""
    if name == "bus_line":
        return len(w.bus_lines)
    if name not in C.TIER_BY_NAME:
        return None
    line, tier = C.TIER_BY_NAME[name]
    supply = supply_of(name)
    lines, tier = ([ln for ln in C.TIER_LINES if supply_of(ln) == supply], 1) if supply else ([line], tier)
    structs = {t["struct"] for ln in lines for t in C.TIER_LINES[ln][tier - 1:]}
    return sum(w.facility_tier(x, y) >= tier for st in structs for x, y in w.facilities(st))


def bus_seats(w) -> tuple[int, int]:
    """(riders, seats) over the city's bus lines, as traffic sets them each month (a line not yet run: 0 riders)."""
    return (round(sum(ln.get("riders", 0) for ln in w.bus_lines)),
            round(sum(ln.get("capacity", C.LINE_CAPACITY) for ln in w.bus_lines)))


def bus_room(w, sites: list[dict]) -> tuple[str, str] | None:
    """(short, note) when the bus_stops list offers no line to build: fewer than the 2 stops create_bus_line needs, or
    BUS_LINES_MAX lines already, or buses with room (a line under BUS_FULL of its seats: the scripted transport's
    rule) while transit reaches TRANSIT_GOAL or the next line through `sites` would add under BUS_GAIN of the trips
    (transit_estimate); None when a line is worth building (none yet, every line near full, or transit under the goal
    and the line adds riders). Smoke / 120-month LLM runs: the mayor read "needs $1,500 for another bus line (transit
    4.6%)" with buses at ~210 of 1,500 seats, a second line on the same stops. None without stops (SITE_NOTES says
    why)."""
    if not sites:
        return None
    if len(sites) < 2:
        return ("one stop only: a line needs 2 (more homes and jobs on the roads)",
                "only one stop is suggested and a bus line needs 2 or more; wait for more homes and job centres")
    lines = w.bus_lines
    if len(lines) >= BUS_LINES_MAX:
        return (f"{len(lines)} bus lines already", f"the city has {len(lines)} bus lines; more add upkeep, not riders")
    if not lines or all(ln.get("riders", 0) >= BUS_FULL * ln.get("capacity", C.LINE_CAPACITY) for ln in lines):
        return None
    transit, more = transit_estimate(w, [[s["x"], s["y"]] for s in sites[:C.MAX_STOPS]])
    gain = more - transit
    if transit < TRANSIT_GOAL and gain >= BUS_GAIN:
        return None
    riders, seats = bus_seats(w)
    why = (f"transit {pct(100 * transit)} reaches the {pct(100 * TRANSIT_GOAL)} your score counts"
           if transit >= TRANSIT_GOAL else
           f"a new line here would add {pct(100 * max(0.0, gain))} of trips (transit {pct(100 * transit)})")
    return (f"buses have room ({riders:,} of {seats:,} seats); {why}",
            f"the bus lines carry {riders:,} riders of {seats:,} seats; {why}. A trip rides the best line covering both "
            f"its ends, so another line pays once buses run {pct(100 * BUS_FULL)} full, or while transit is under "
            f"{pct(100 * TRANSIT_GOAL)} and the next line (today's job centres and the homes the lines miss) would "
            f"add {pct(100 * BUS_GAIN)} of trips or more")


def linked(w) -> bool:
    """The city has a train station or harbour (World.demand_caps' regional link)."""
    return bool(w.facilities(C.TRAIN) or w.facilities(C.HARBOUR))


def capped_unlinked(w) -> bool:
    """R and C demand are capped for want of a regional link: no train station or harbour from CAP_POP_CONNECTION
    residents (World.demand_caps). transport's first link is then a first building (first_build)."""
    return not linked(w) and w.metrics()["population"] >= C.CAP_POP_CONNECTION


# an unlinked city's first link is transport's next build (first_build) and what it keeps from this share of
# CAP_POP_CONNECTION residents (5,100 of 6,000), so it is bought before R and C demand are capped. civitas-v2 review,
# eval_v2 seed 20 at month 96: 5,932 residents, no link, freight ~80%, a harbour ready at $10k and transport holding
# $5,194 after asking for $10k; its needs line named avenue tiles, so the mayor read "idle $4,194 ... no top-up needed"
LINK_AHEAD = 0.85


def link_due(w) -> bool:
    """The city has no train station or harbour and LINK_AHEAD of CAP_POP_CONNECTION residents or more (capped_unlinked
    included): transport's first link is a first building, after a bus line its list offers until the cap is reached
    (first_build), and a blocked one asks planning for a road to the map edge (blocked_firsts)."""
    return not linked(w) and w.metrics()["population"] >= LINK_AHEAD * C.CAP_POP_CONNECTION


def link_room(w, purpose: str) -> tuple[str, str] | None:
    """(short, note) when a train station or harbour (LINK_PURPOSES) wouldn't pay now: the city has a link and, for a
    train station, one already, and freight runs under FREIGHT_FULL of its capacity (exports + imports, as the scripted
    transport reads it); None when it would pay, and for other purposes."""
    if purpose not in LINK_PURPOSES or not linked(w) or (purpose == "train_station" and not w.facilities(C.TRAIN)):
        return None
    m = w.metrics()
    used = (m.get("exports", 0.0) + m.get("imports", 0.0)) / max(1.0, m.get("freight_capacity", 0.0))
    if used >= FREIGHT_FULL:
        return None
    what = "train station" if purpose == "train_station" else "harbour"
    trains, harbours = len(w.facilities(C.TRAIN)), len(w.facilities(C.HARBOUR))
    return (f"freight runs {pct(100 * used)} of capacity: another {what} pays from {pct(100 * FREIGHT_FULL)}",
            f"the city's {trains} train station{'s' * (trains != 1)} and {harbours} harbour{'s' * (harbours != 1)} link "
            f"it to the region already (without a link R and C demand stay capped from {C.CAP_POP_CONNECTION:,} "
            f"residents); freight runs {pct(100 * used)} of its capacity, and another {what} adds only freight room, "
            f"which pays once freight passes {pct(100 * FREIGHT_FULL)}")


# transit_estimate's results per world: (World.version, {(stops, the lines' stops, fare): (now, with)}), dropped when
# the version moves (refresh() after every action, tick()). site_lists runs bus_room from every DeptMoney, so one
# mayor turn asked for the same estimate dozens of times
_ESTIMATES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def transit_estimate(w, stops: list[list[int]] | None = None) -> tuple[float, float]:
    """(now, with) the transit share of trips as traffic computes it, for the city's bus and metro lines as they are
    now and with one more bus line through `stops` (the same with none): traffic's own pieces (TrafficMixin
    _trip_origins, _gravity, _metro_cover, _hub_share, _split_trips) on World.hubs from the last tick, writing
    nothing. A trip rides a bus when one line's stops reach both its home and its job centre, after the metro took
    its share, up to the lines' seats. `now` is this month's share for lines built since the tick (the transit_share
    metric is last month's); with the same hubs and homes it is the metric exactly. (0, 0) without trips."""
    lines = tuple(tuple(map(tuple, ln["stops"])) for ln in w.bus_lines)
    key = (tuple(map(tuple, stops or ())), lines, w.fare)
    version, cache = _ESTIMATES.get(w, (None, {}))
    if version != w.version:
        cache = {}
        _ESTIMATES[w] = (w.version, cache)
    if key not in cache:
        cache[key] = _transit_estimate(w, stops)
    return cache[key]


def _transit_estimate(w, stops: list[list[int]] | None) -> tuple[float, float]:
    hubs = w.hubs
    if not hubs:
        return 0.0, 0.0
    omass = w._trip_origins(w.residents_grid(), w._anchors())
    if omass.sum() <= 0:
        return 0.0, 0.0
    flows = omass[None, :] * w._gravity(hubs)[3]
    m_share = w._metro_cover(hubs)[0]
    base = w._bus_base()
    trips = float(omass.sum())

    def share(covs: list[np.ndarray]) -> float:
        best = np.zeros((len(hubs), w.h * w.w))
        for cov in covs:
            best = np.maximum(best, w._hub_share(hubs, cov, base))
        _, m_riders, _, _, riders = w._split_trips(flows, np.minimum(best, 1.0), m_share, len(covs))
        return (riders + m_riders) / trips

    covs = [w._cover(ln["stops"], C.STOP_RADIUS) for ln in w.bus_lines]
    now = share(covs)
    return now, (share(covs + [w._cover(stops, C.STOP_RADIUS)]) if stops else now)


def bus_cover(w) -> np.ndarray:
    """The tiles within C.STOP_RADIUS of a bus stop, from the lines' stops as they are now (World.transit_cov is
    recomputed only by the next tick, so it misses a line created this turn)."""
    cover = np.zeros((w.h, w.w), bool)
    for line in w.bus_lines:
        cover |= w._cover(line["stops"], C.STOP_RADIUS)
    return cover


def new_line_stops(w) -> list[list[int]]:
    """The next bus line's stops once the city has one: the biggest job centres and the biggest home areas outside the
    lines' reach (observation._suggest_stops with avoid; its default suggestions, which the scripted transport reads,
    keep naming the first line's stops)."""
    return _suggest_stops(w, avoid=bus_cover(w))


def bus_stops(lst: SiteList) -> list[list[int]]:
    """Every suggested stop of a bus_stops list, up to C.MAX_STOPS: one line through all of them, never a choice of
    sites (the LLM runs built lines of 1-2 stops from "6 sites, best (40,5)")."""
    return [[int(s["x"]), int(s["y"])] for s in lst.sites[:C.MAX_STOPS]]


def bus_line_call(stops: list[list[int]]) -> str:
    """The create_bus_line call through `stops`: "create_bus_line stops [[12,24],[30,24]]"."""
    return "create_bus_line stops " + json.dumps(stops, separators=(",", ":"))


def first_bus_line(w, lists: dict[str, SiteList]) -> Price | None:
    """The first bus line's price while the city has none and its stops are ready (bus_stops listed with 2+ stops):
    transport's priority line (briefing) and the upgrade_road guard (ToolBox._bus_line_refusal). None otherwise."""
    lst = lists.get("bus_stops")
    if w.bus_lines or lst is None or lst.short or len(lst.sites) < 2:
        return None
    return site_price(w, "transport", "bus_stops")


def build_status(w, dept: str, purpose: str, name: str) -> str:
    """How the kind `name` builds is doing, for another one on the mayor's needs line: the share of its supply used
    ("water 117% used": demand 197 of 169; "water supply 169, demand 197" took 16 more characters a line), else its
    COVERAGE_OF coverage ("coverage 74%"); "" for neither."""
    m = w.metrics()
    if supply := supply_of(name):
        have, need = m[f"{supply}_supply"], m[f"{supply}_demand"]
        return f"{supply} {pct(100 * need / have)} used" if have > 0 else f"no {supply} supply"
    key = COVERAGE_OF.get((dept, purpose))
    if key and (fall := coverage_trend(w, key)):
        return f"coverage {fall}"
    return f"{COVERAGE_LABEL.get(key, 'coverage')} {pct(100 * m[key])}" if key else ""


# ------------------------------------------------------------------ keeping up with coverage
# Late smoke 20261007 seed 2 (the LLM fleet from month 96, act_every 3): population +30% in 12 months while fire
# coverage went 94 -> 86 -> 81 -> 78%, police 94 -> 79, health 83 -> 67, schools 93 -> 78, with $35-39k above the
# reserve all along. Each needs line named one "another <building>", the mayor funded that one every turn, and a
# quarter later coverage was lower. A department short of KEEP_UP_GOAL now names the buildings it takes to get back
# (keep_up), as many as the money above the reserve covers after every other top-up (funding_gaps)
KEEP_UP_GOAL = 0.9           # coverage the count aims at (the score counts coverage linearly; most playbooks say 85%)
KEEP_UP_MAX = 4              # buildings named at most for one service (the successful-action cap is 8 a turn)
KEEP_UP_GAIN = 0.01          # a building adding less coverage than this isn't counted (crowded, or nothing near)
DOWNTOWN_SHOWN = 0.02        # the stuck note names the coverage lost downtown from this much (2 points)
TREND_STEP = 3               # months between the points of the falling note (a quarter: the fleet's cadence)
TREND_POINTS = 3             # points it shows at most: "94 -> 86 -> 81%"
# (dept, purpose) -> the service whose coverage its building adds to (service_coverage); COVERAGE_OF has the metric
KEEP_UP = {("police", "station"): "police", ("fire", "station"): "fire", ("health", "clinic"): "health",
           ("health", "landfill"): "garbage", ("education", "school"): "school",
           ("education", "high_school"): "high"}
# keep_up's results per world, dropped when what coverage reads moves (_cover_key): funding_gaps runs for every guard
# and money note of a mayor turn, and its money moves (each a refresh: a new World.version) change none of it. Keyed
# on the version, keep_up took 4.2 s of a 9.3 s fleet_scripted episode (6.3 s without it)
_KEEP_UPS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _cover_key(w) -> tuple:
    """What service_coverage reads that can change between two calls: the month (residents, cohorts, congestion and
    landfill fill move only in a tick), the roads, the buildings and their levels and density, the funding levels,
    the policies (city-wide and by district) and each facility's tier and modules."""
    return (w.month, w.road_key, hash(w.struct.tobytes()), hash(w.level.tobytes()), hash(w.density.tobytes()),
            tuple(sorted(w.funding.items())), tuple(sorted(w.policies)), hash(w.district_id.tobytes()),
            tuple(sorted((d["id"], tuple(sorted(d["policies"]))) for d in w.districts.values())),
            tuple(sorted((k, v.get("tier"), tuple(v.get("modules", ()))) for k, v in w.facility_state.items())))


def coverage_trend(w, key: str) -> str:
    """"falling: 94 -> 86 -> 81%" when the coverage metric `key` fell (in whole percent) at each of the last points
    TREND_STEP months apart (the newest is now: World.metrics), at most TREND_POINTS of them; "" when it didn't fall
    over the last step (or there is no such history). Transit is left out: its line is transport's own."""
    if key == "transit_share" or len(w.history) <= TREND_STEP:
        return ""
    vals = [round(100 * w.metrics().get(key, 0.0))]
    for k in range(1, TREND_POINTS):
        i = len(w.history) - 1 - k * TREND_STEP
        if i < 0 or key not in w.history[i]:
            break
        v = round(100 * w.history[i][key])
        if v <= vals[-1]:
            break
        vals.append(v)
    return "" if len(vals) < 2 else "falling: " + " -> ".join(map(str, reversed(vals))) + "%"


def _road_cover(w, parts: list[list[tuple[int, int]]], radius: int, caps: np.ndarray, demand: np.ndarray,
                full: float) -> np.ndarray:
    """ServicesMixin._cover_group (with its _road_field) for buildings at `parts`, without the BFS cache: coverage
    before funding and traffic."""
    dist, lab = road_bfs(w.road_labels >= 0, parts, max_d=radius + 1)
    td, tl = spread_to_tiles(dist, lab, C.ACCESS_RADIUS)
    for k, tiles in enumerate(parts):                  # the facility's own tiles
        if any(lab[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() >= 0 for x, y in tiles):
            for x, y in tiles:
                td[y, x], tl[y, x] = 0, k
    reach = td <= radius
    load = np.bincount(tl[reach], weights=demand[reach], minlength=len(parts))
    ratio = np.where(load > caps, caps / np.maximum(load, 1e-9), 1.0)
    f = np.clip((radius - td) / max(1e-9, (1.0 - full) * radius), 0.0, 1.0) * reach if full < 1 else reach * 1.0
    return f * ratio[np.maximum(tl, 0)]


class _Sources:
    """ServicesMixin.coverage for the buildings of one struct, with buildings of a tier-1 `spec` added at top-left
    tiles (grid(extra)); the buildings there and their capacities read once. Writes nothing. grid's `swap` puts one
    of them a tier up ((x, y) -> (its next spec, its tiles then)), `drop` leaves some out and `add` adds buildings of
    other specs ((spec, tiles)): a tier upgrade's estimate (tier_up)."""

    def __init__(self, w, struct: int, demand: np.ndarray, dept: str, spec: dict | None = None, *,
                 full: float = C.COVERAGE_FULL, skip=frozenset(), traffic: bool = False, cap_key: str = "capacity",
                 min_tier: int = 1):
        self.w, self.demand, self.spec, self.full, self.traffic, self.cap_key = w, demand, spec, full, traffic, cap_key
        self.struct, self.min_tier = struct, min_tier
        self.sources = [s for s in w.facilities(struct) if w.building_network(*s) >= 0 and s not in skip
                        and w.facility_tier(*s) >= min_tier]
        self.eff = w.funding_eff(dept)
        specs = [w.facility_spec(*s) for s in self.sources]
        self.parts = [w.parts(*s) for s in self.sources]
        self.caps = [sp[cap_key] * self.eff * w.module_factor(x, y, "expansion")
                     for (x, y), sp in zip(self.sources, specs)]
        self.radii = [int(sp["radius"]) for sp in specs]

    def takes(self, spec: dict) -> bool:
        """Whether a building of `spec` would be one of these (its struct, at min_tier or above)."""
        return spec["struct"] == self.struct and spec["tier"] >= self.min_tier

    def grid(self, extra=(), swap: dict | None = None, drop=frozenset(), add=()) -> np.ndarray:
        w = self.w
        keep = [k for k, s in enumerate(self.sources) if s not in drop]
        parts, caps, radii = [self.parts[k] for k in keep], [self.caps[k] for k in keep], [self.radii[k] for k in keep]
        for (x, y), (spec, tiles) in (swap or {}).items():
            if (x, y) in self.sources and (x, y) not in drop:
                k = keep.index(self.sources.index((x, y)))
                parts[k] = list(tiles)
                caps[k] = spec[self.cap_key] * self.eff * w.module_factor(x, y, "expansion")
                radii[k] = int(spec["radius"])
        adds = [(self.spec, [(x + i, y + j) for j in range(self.spec["footprint"][1])
                             for i in range(self.spec["footprint"][0])]) for x, y in extra] + list(add)
        for spec, tiles in adds:
            parts.append(list(tiles))
            caps.append(spec[self.cap_key] * self.eff)
            radii.append(int(spec["radius"]))
        if not parts:
            return np.zeros((w.h, w.w))
        cov = None
        for r in sorted(set(radii)):
            idx = [k for k, rk in enumerate(radii) if rk == r]
            g = _road_cover(w, [parts[k] for k in idx], r, np.array([caps[k] for k in idx]), self.demand, self.full)
            cov = g if cov is None else np.maximum(cov, g)
        cov = cov * min(1.0, self.eff)
        if self.traffic:
            cov = cov * w.congestion_penalty
        return np.clip(cov, 0.0, 1.0)


class _Coverage:
    """One service's coverage metric as ServicesMixin._compute_services computes it from the world as it is now, with
    tier-1 buildings of `spec` added (value(extra)); what doesn't change with them (residents, demand, tier-2 and
    hospital cover, incinerators) read once, so keep_up's tries cost one road BFS per radius each."""

    def __init__(self, w, service: str, spec: dict | None = None):
        self.res = res = w.residents_grid()
        self.rsum = float(res.sum())
        self.service = service
        developed = np.isin(w.struct, C.ZONES) & (w.level > 0)
        self.dev_hi = developed & (w.level >= C.DOWNTOWN_LEVEL)
        self.fixed_src = None
        if service in ("police", "fire"):
            struct, demand = (C.POLICE, w._people_grid()) if service == "police" else (C.FIRE, developed.astype(float))
            self.own = _Sources(w, struct, demand, service, spec, traffic=True)
            self.fixed_src = _Sources(w, struct, demand, service, traffic=True, min_tier=2)     # tier-2 cover
        elif service == "health":
            self.own = _Sources(w, C.CLINIC, res, "health", spec, traffic=True)
            self.fixed_src = _Sources(w, C.HOSPITAL, res, "health", traffic=True)
        elif service in ("school", "high"):
            n = w.cohorts
            share = float(n[0 if service == "school" else 1].sum()) / max(1.0, float(n.sum()))
            self.own = _Sources(w, C.SCHOOL if service == "school" else C.HIGHSCHOOL, res * share, "education", spec,
                                cap_key="seats")
        else:                                           # garbage: landfills with room left, and incinerators
            prod = w._garbage_production()
            full = {s for s in w.facilities(C.LANDFILL) if (cap := w.facility_spec(*s).get("fill"))
                    and w.facility_state.get(s, {}).get("fill", 0.0) >= cap}       # _full_landfills, read-only
            self.own = _Sources(w, C.LANDFILL, prod, "health", spec, full=1.0, skip=full, cap_key="collect")
            self.fixed_src = _Sources(w, C.INCINERATOR, prod, "health", full=1.0, cap_key="collect")
        self.fixed = self.fixed_src.grid() if self.fixed_src is not None else None

    def value(self, extra=(), up: tuple | None = None) -> tuple[float, np.ndarray]:
        """The metric, and the per-tile cover it weighs residents by (a school's: covered or not). `up`: one
        building a tier up instead, ((x, y), its next spec, its tiles then; World.next_tier), as upgrade_building
        leaves it: the same struct keeps its place among its sources with the new capacity and radius, a new struct
        (a clinic's hospital) leaves them; the new spec joins the tier-2 / hospital cover when it counts there."""
        if self.rsum <= 0:
            return 0.0, np.zeros(self.res.shape)
        if up is None:
            g, fixed = self.own.grid(extra), self.fixed
        else:
            xy, spec, tiles = up
            moved = {xy: (spec, tiles)}
            g = (self.own.grid(extra, swap=moved) if self.own.takes(spec)
                 else self.own.grid(extra, drop={xy}))
            src = self.fixed_src
            fixed = (self.fixed if src is None or not src.takes(spec) else src.grid(swap=moved)
                     if xy in src.sources else src.grid(add=[(spec, tiles)]))
        if self.service in ("school", "high"):
            g = (g > 0.05).astype(float)
        elif self.service == "garbage":
            g = np.maximum(g, fixed)
        else:                   # downtown tiles need tier-2 cover (T1_DOWNTOWN); a hospital covers like a clinic
            if self.service == "health":
                g = np.maximum(g, fixed)
            g = np.where(self.dev_hi, np.maximum(C.T1_DOWNTOWN * g, fixed), g)
        return float((g * self.res).sum() / self.rsum), g


def service_coverage(w, service: str, extra=(), spec: dict | None = None) -> float:
    """The coverage metric of `service` (a KEEP_UP value) as ServicesMixin._compute_services computes it from the
    world as it is now, with tier-1 buildings of `spec` added at `extra` (top-left tiles); writes nothing. With no
    extra and nothing changed since the last tick it is the metric. Tier-1 buildings add nothing to the tier-2 cover
    downtown tiles also need (T1_DOWNTOWN)."""
    return _Coverage(w, service, spec).value(extra)[0]


class KeepUp(NamedTuple):
    """What it takes to bring one service's coverage back to KEEP_UP_GOAL (keep_up)."""
    key: str                                # the COVERAGE_OF metric
    now: float                              # the metric now (last month's)
    levels: tuple[float, ...]               # the estimated coverage with 0, 1, 2 ... of `sites` built
    sites: tuple[tuple[int, int], ...]      # where, in the order picked
    gap: tuple[int, int] | None = None      # with none picked (no ready site adds KEEP_UP_GAIN): the tile at the
                                            # middle of the most uncovered residents outside downtown, which no ready
                                            # site reaches
    downtown: float = 0.0                   # with none picked: the coverage lost on level 4-5 tiles, which tier-1
                                            # buildings cover only to T1_DOWNTOWN (a tier-2 one covers them)

    @property
    def n(self) -> int:
        return len(self.sites)


def keep_up(w, dept: str, purpose: str, sites) -> KeepUp | None:
    """How many more of the building `dept`'s `purpose` builds bring its coverage metric to KEEP_UP_GOAL, picked
    greedily among its ready `sites` ((x, y), best first; at most MAX_SITES): each next one where it adds the most
    coverage (service_coverage: the sim's own coverage with the building placed), stopping at the goal, at
    KEEP_UP_MAX, or when the best adds under KEEP_UP_GAIN. None for a purpose not in KEEP_UP, with the metric at or
    above the goal, or with no site. Cached while what it reads stays (_cover_key). About 1-2 ms a building tried."""
    service = KEEP_UP.get((dept, purpose))
    key = COVERAGE_OF.get((dept, purpose))
    sites = tuple(tuple(s) for s in sites)[:MAX_SITES]
    if service is None or not sites or w.metrics()[key] >= KEEP_UP_GOAL:
        return None
    cache = _keep_up_cache(w)
    k = (dept, purpose, sites)
    if k not in cache:
        line = C.TIER_BY_NAME[SITE_BUILDS[dept][purpose][0]][0]
        spec = C.TIER_LINES[line][0]
        model = _Coverage(w, service, spec)
        base, cover = model.value()
        # lazy greedy: a site's gain only shrinks as others are picked (they share the uncovered homes), so a gain
        # computed before the last pick bounds it; only the top one is recomputed (about 2 tries a pick, not 6)
        levels, chosen = [base], []
        value = {s: model.value([s])[0] for s in sites}             # with the buildings picked when `fresh`
        fresh = set(sites)
        while value and len(chosen) < KEEP_UP_MAX and levels[-1] < KEEP_UP_GOAL:
            s = max(value, key=lambda t: (value[t], -sites.index(t)))
            if s not in fresh:
                value[s], fresh = model.value(chosen + [s])[0], fresh | {s}
                continue
            if value[s] - levels[-1] < KEEP_UP_GAIN:
                break
            chosen.append(s)
            levels.append(value.pop(s))
            fresh = set()
        gap, downtown = None, 0.0
        if not chosen and levels[0] < KEEP_UP_GOAL:     # where the uncovered residents are, out of the sites' reach
            lost = model.res * (1.0 - cover)
            hi = model.dev_hi if service in ("police", "fire", "health") else np.zeros_like(model.dev_hi)
            downtown = float((lost * hi).sum()) / max(1.0, model.rsum)
            y, x = np.unravel_index(int(np.argmax(box_sum(lost * ~hi, 2))), cover.shape)
            gap = (int(x), int(y))
        cache[k] = KeepUp(key, w.metrics()[key], tuple(levels), tuple(chosen), gap, downtown)
    return cache[k]


def _keep_up_cache(w) -> dict:
    """keep_up's per-world cache (_KEEP_UPS), emptied when what coverage reads moves (_cover_key)."""
    state, cache = _KEEP_UPS.get(w, (None, {}))
    if state != (now := _cover_key(w)):
        cache = {}
        _KEEP_UPS[w] = (now, cache)
    return cache


# civitas-v2 review, eval_v2 seeds at month 96: with no tier upgrade to name, police and fire still read "build another
# police / fire station ($2,500): your funds cover it" at 89-99% coverage, where the best ready site added 0.0-1.5
# points (11 cases; seed 4 fire 96.9%: +0.00; seeds 5 and 20 police 99.4% / 97.1%: +0.00), and health "another clinic
# or another landfill" with garbage at 98%. Another building of a KEEP_UP purpose of these departments is now a ready
# site only while its coverage is under KEEP_UP_GOAL and one of its ready sites adds KEEP_UP_GAIN or more
# (keep_up_room): its list reads covered or "adds under 1%" instead, so no needs line, ask, to-fund item or keep names
# it. Education goes by enrolment instead (_wanted, edu_room)
KEEP_UP_GATED = ("police", "fire", "health")


def at_goal(w, dept: str, purpose: str) -> bool:
    """Another building of a KEEP_UP purpose of police, fire or health (KEEP_UP_GATED; the city has one) while its
    coverage metric (COVERAGE_OF, last month's) is at KEEP_UP_GOAL or more."""
    if dept not in KEEP_UP_GATED or (dept, purpose) not in KEEP_UP:
        return False
    return (w.metrics()[COVERAGE_OF[dept, purpose]] >= KEEP_UP_GOAL
            and bool(buildings_of(w, SITE_BUILDS[dept][purpose][0])))


def keep_up_room(w, dept: str, purpose: str, sites: list[dict]) -> tuple[str, str] | None:
    """(short, note) for the list of another building of a KEEP_UP_GATED purpose that wouldn't keep coverage up: its
    coverage is at KEEP_UP_GOAL or more (at_goal: "fire coverage 97% (goal 90%)"), or under it with no ready site adding
    KEEP_UP_GAIN (keep_up picks none: "ready sites add under 1% (health coverage 86%)", the note saying where the
    uncovered homes are and what is lost downtown); None otherwise (a first one, or a site that adds coverage). The
    sites are kept: find_sites still shows them."""
    if dept not in KEEP_UP_GATED or (dept, purpose) not in KEEP_UP or not sites:
        return None
    if not buildings_of(w, SITE_BUILDS[dept][purpose][0]):
        return None
    service, key, goal = KEEP_UP[dept, purpose], COVERAGE_OF[dept, purpose], pct(100 * KEEP_UP_GOAL)
    now = pct(100 * w.metrics()[key])
    what = building_name(SITE_BUILDS[dept][purpose][0])
    if at_goal(w, dept, purpose):
        return (f"{service} coverage {now} (goal {goal})",
                f"{service} coverage {now}, at or above the {goal} goal: find_sites lists the sites for another {what}")
    ku = keep_up(w, dept, purpose, [(s["x"], s["y"]) for s in sites])
    if ku is None or ku.n:
        return None
    if ku.gap is None:                  # at the goal already by the sim's coverage now (buildings since the last tick)
        est = pct(100 * ku.levels[0])
        return (f"{service} coverage ~{est} now (goal {goal})",
                f"{service} coverage {now} last month, ~{est} with the buildings since: find_sites lists the sites "
                f"for another {what}")
    gain = f"{100 * KEEP_UP_GAIN:g}%"
    return (f"ready sites add under {gain} ({service} coverage {now})",
            f"another {what} at any ready site adds under {gain} to {service} coverage ({now}): {stuck_note(ku)}")


def stuck_note(ku: KeepUp) -> str:
    """Where the coverage keep_up found no ready site for is missing (a KeepUp with no site picked): "the uncovered
    homes near (38,28) are out of their road reach", then, from DOWNTOWN_SHOWN, "; 7 points are lost downtown (level
    4-5), where tier 1 covers 60% at most: upgrade_facility to tier 2" (the advice left out in the open style)."""
    x, y = ku.gap
    hi = round(100 * ku.downtown)
    return (f"the uncovered homes near ({x},{y}) are out of their road reach"
            + (f"; {hi} points are lost downtown (level 4-5), where tier 1 covers {pct(100 * C.T1_DOWNTOWN)} at most"
               + advice(": upgrade_facility to tier 2") if ku.downtown >= DOWNTOWN_SHOWN else ""))


# ------------------------------------------------------------------ tier upgrades
# v1 leaderboard (4 models x 8 seeds, fleet_smoke_late, the models from month 96): police and fire read "build another
# police / fire station ($2,500)" every turn and never an upgrade; tier-1 stations cover level 4-5 tiles only to
# T1_DOWNTOWN, the model cities grew high-density tiles 140 -> ~250, and each model built 23-26 tier-1 police stations
# and 16-26 fire stations with almost no upgrade (the scripted fleet: 6 police and 6 fire upgrades). Police, fire and
# health name the tier upgrade that adds the most coverage (tier_up) instead of another tier-1 building when it adds
# more than one at the best ready site; education names a level's bigger school while that level's enrolment is short
# of seats rather than of schools in reach (school_upgrade). An upgrade is "another" building on the mayor's lines
# (Need.first False): it comes out of the money above the reserve after first buildings, supply and planning's road
# (fund_cover), never their allowance, and the guards treat it as they treat another station
TIER_UP = {"police": ("station", "police"), "fire": ("station", "fire"), "health": ("clinic", "health")}
TIER_UP_LINES = {"police": ("police_station",), "fire": ("fire_station",), "health": ("clinic",),
                 "education": ("elementary_school", "high_school")}
# the builds an upgrade may take the place of (a first building or a levee keeps its turn; so does a landfill while
# garbage coverage is under KEEP_UP_GOAL: health's clinic and landfill both cost $2,000, so a tie names both)
TIER_UP_REPLACES = {"police": {"station", "jail"}, "fire": {"station"}, "health": {"clinic", "hospital"}}
TIER_UP_TRIED = 3            # upgrade offers estimated at most (observation lists them by downtown need)
UPGRADE_OFFERS = {"police": ("police", "upgrades"), "fire": ("fire", "upgrades"),
                  "health": ("health", "upgrades", "care"), "education": ("education", "upgrades")}
# education's enrolment (World.enrollment, what scoring._enrolled weighs by children and teens): a level is short
# under ENROL_GOAL. v1: "build your first university ($12k)" on 7 of 8 seeds at month 96, while high-school enrolment
# was 78-85% (seed 7: 28%); a university counts in no enrolment the score reads, costs $150 a month, and each model
# built 7, then 16-22 elementary schools at ~99% elementary enrolment (the needs line's cheapest "another"). The
# school list's purpose -> its level; a university only once both school levels are covered (_wanted)
ENROL_GOAL = 0.9
EDU_LEVELS = {"school": "elementary", "high_school": "high", "university": "university"}
EDU_WHO = {"elementary": "children", "high": "teens", "university": "students"}
EDU_LABEL = {"elementary": "elementary", "high": "high-school", "university": "university"}


class Upgrade(NamedTuple):
    """A tier upgrade a department's next build names (tier_up, school_upgrade)."""
    x: int                  # the building (its top-left tile); a fresh building's site
    y: int
    frm: str                # its tier name now ("police_station"); "" for a fresh tier-2 building (health's hospital
                            # at a ready 2x2 site, when no clinic has room to grow into one)
    to: str                 # the next tier's ("police_hq")
    cost: float             # what upgrade_facility charges (catalog.upgrade_cost)
    upkeep: float           # what it adds a month, at the department's funding level
    status: str             # why, for the needs line: "coverage 81%, ~88% with it; 24 downtown tiles lack tier-2 cover"
    purpose: str            # the find_sites purpose of its line's tier-1 building ("station", "clinic", "school")
    alt: str = ""           # what the tier-1 building would do instead ("another police station adds under 1%")

    @property
    def call(self) -> str:
        if not self.frm:
            return f"build_{self.to} " + json.dumps({"x": self.x, "y": self.y}, separators=(",", ":"))
        return "upgrade_facility " + json.dumps({"x": self.x, "y": self.y, "module": "tier"}, separators=(",", ":"))

    @property
    def what(self) -> str:
        """"an upgrade of its police station at (12,20) to a police HQ" (its -> your in the department's words); a
        fresh one "a hospital at (14,17)"."""
        if not self.frm:
            return f"{a_or_an(tier_name(self.to))} at ({self.x},{self.y})"
        return f"an upgrade of its {tier_name(self.frm)} at ({self.x},{self.y}) to {a_or_an(tier_name(self.to))}"


def _replaceable(w, dept: str, purposes) -> bool:
    """Whether a tier upgrade may take the place of `dept`'s next build of these purposes (TIER_UP_REPLACES)."""
    covered = w.metrics()["garbage_coverage"] >= KEEP_UP_GOAL
    return all(p in TIER_UP_REPLACES[dept] or (p == "landfill" and covered) for p in purposes)


def tier_name(name: str) -> str:
    """A tier name for people: building_name, HQ in capitals ("police HQ", "large high school")."""
    return re.sub(r"\bhq\b", "HQ", building_name(name))


def upgrade_bound(w, dept: str) -> float:
    """The dearest unlocked tier upgrade of `dept`'s TIER_UP_LINES buildings (room to grow not checked): DeptMoney's
    bound, so funds that cover every site price but not an upgrade still read short of it."""
    best = 0.0
    for line in TIER_UP_LINES.get(dept, ()):
        tiers = C.TIER_LINES[line]
        for st in {t["struct"] for t in tiers}:
            for x, y in w.facilities(st):
                t = w.facility_tier(x, y)
                if t < len(tiers) and (not tiers[t].get("unlock") or w.unlocked(tiers[t]["unlock"])):
                    best = max(best, C.upgrade_cost(tiers[t - 1], tiers[t]))
    return best


def _offers(obs: dict, dept: str) -> list[dict]:
    """The observation's upgrade offers for `dept`'s TIER_UP_LINES buildings (unlocked, room to grow), best first."""
    found = obs
    for key in UPGRADE_OFFERS[dept]:
        found = found.get(key) if isinstance(found, dict) else None
    lines = TIER_UP_LINES[dept]
    return [o for o in found or [] if C.TIER_BY_NAME.get(o["from"], (None,))[0] in lines]


def _upgrade(w, dept: str, o: dict, cur: dict, nxt: dict, status: str, purpose: str, alt: str = "") -> Upgrade:
    up = (nxt["upkeep"] - cur["upkeep"]) * w.funding.get(dept, 1.0)
    return Upgrade(int(o["x"]), int(o["y"]), cur["name"], nxt["name"], float(C.upgrade_cost(cur, nxt)), up, status,
                   purpose, alt)


def tier_up(w, dept: str, sites, offers: list[dict], fresh=()) -> Upgrade | None:
    """Police, fire or health's tier upgrade worth naming: while level 4-5 tiles lack tier-2 cover (World.stats
    downtown_uncovered), the offer (up to TIER_UP_TRIED) whose upgrade brings the coverage metric highest (the sim's
    coverage with that building a tier up: _Coverage.value(up=...)), when it adds KEEP_UP_GAIN or more and more than
    one more tier-1 building at the best of its ready `sites` ((x, y); none ready: nothing); else None. Health's
    `fresh` are its ready hospital sites (2x2 top-left tiles, up to TIER_UP_TRIED): a hospital built there competes
    with the upgrades (a clinic grows into one only with a free 2x2 block around it). Cached like keep_up. Late
    states, seeds 4 and 7 at month 96: "ready sites add under 1% ... 18 points are lost downtown". Nothing while the
    coverage is at KEEP_UP_GOAL or more (civitas-v2 review, eval_v2 seed 18: a $7,800 police HQ for 97 -> 98%)."""
    purpose, service = TIER_UP[dept]
    key = COVERAGE_OF[dept, purpose]
    down = (w.stats.get("downtown_uncovered") or {}).get(service, 0)
    offers = [o for o in offers if C.TIER_BY_NAME.get(o["from"], (None,))[0] in TIER_UP_LINES[dept]][:TIER_UP_TRIED]
    fresh = tuple(tuple(s) for s in fresh)[:TIER_UP_TRIED] if dept == "health" else ()
    if not down or not (offers or fresh) or w.metrics()[key] >= KEEP_UP_GOAL:     # kept up: as keep_up_room
        return None
    sites = tuple(tuple(s) for s in sites)[:MAX_SITES]
    cache = _keep_up_cache(w)
    k = ("tier", dept, sites, tuple((o["x"], o["y"]) for o in offers), fresh)
    if k not in cache:
        line = TIER_UP_LINES[dept][0]
        model = _Coverage(w, service, C.TIER_LINES[line][0])
        base = model.value()[0]
        station = max((model.value([s])[0] for s in sites), default=base)
        best = None
        for o in offers:
            try:
                cur, nxt, _, tiles = w.next_tier(o["x"], o["y"])
            except ValueError:
                continue
            level = model.value(up=((o["x"], o["y"]), nxt, tiles))[0]
            if best is None or level > best[0]:
                best = (level, o, cur, nxt)
        hosp = C.TIER_LINES["clinic"][1]
        for x, y in fresh:                          # xy None: no building of its own moves, the hospital is added
            fw, fh = hosp["footprint"]
            level = model.value(up=(None, hosp, [(x + i, y + j) for j in range(fh) for i in range(fw)]))[0]
            if best is None or level > best[0]:
                best = (level, {"x": x, "y": y}, None, hosp)
        cache[k] = None
        if best is not None and best[0] - base >= KEEP_UP_GAIN and best[0] > station:
            level, o, cur, nxt = best
            now = w.metrics()[key]
            alt = building_name(SITE_BUILDS[dept][purpose][0])
            gain = round(100 * (station - base))
            status = (f"coverage {coverage_trend(w, key) or pct(100 * now)}, ~{pct(100 * level)} with it; {down} "
                      + ("downtown tile lacks" if down == 1 else "downtown tiles lack") + " tier-2 cover")
            alt = (f"another {alt} at your best ready site adds " + (f"~{gain}%" if gain >= 1 else "under 1%")
                   if sites else f"no ready site for another {alt}")
            if cur is None:                         # a fresh hospital, priced as build_hospital charges it
                pr = site_price(w, dept, "hospital")
                cache[k] = Upgrade(o["x"], o["y"], "", nxt["name"], pr.cost, pr.upkeep, status, "hospital", alt)
            else:
                cache[k] = _upgrade(w, dept, o, cur, nxt, status, purpose, alt)
    return cache[k]


def enrolment(w, level: str) -> tuple[float, float, float, float]:
    """(enrolled share, students, seats, share of their homes in reach) of a school level as the sim's
    _update_enrollment computes it: World.enrollment (last tick), the cohorts, the seats and the coverage."""
    n, st = w.cohorts, w.stats
    if level == "university":
        students, cov = float(n[2, 2]) * C.STUDENT_AGE_SHARE, 1.0 if st.get("seats_university", 0) else 0.0
    else:
        students = float(n[0 if level == "elementary" else 1].sum())
        cov = st.get("school_coverage" if level == "elementary" else "high_coverage", 0.0)
    return float(w.enrollment.get(level, 0.0)), students, float(st.get(f"seats_{level}", 0)), float(cov)


def enrol_short(w, level: str) -> bool:
    """A school level with students enrols under ENROL_GOAL of them."""
    share, students, _, _ = enrolment(w, level)
    return students >= 1 and share < ENROL_GOAL


def enrol_text(w, level: str) -> str:
    """"high-school enrolment 86%: 700 seats for 817 teens, 92% of their homes in reach"."""
    share, students, seats, cov = enrolment(w, level)
    reach = f", {pct(100 * cov)} of their homes in reach" if level != "university" else ""
    return (f"{EDU_LABEL[level]} enrolment {pct(100 * share)}: {round(seats):,} seats for {round(students):,} "
            f"{EDU_WHO[level]}{reach}")


def upgrade_line(w, dept: str, obs: dict) -> list[str]:
    """The Ready sites line of `dept`'s tier upgrades on offer (TIER_UP_LINES; unlocked, with room to grow: the
    observation's, which my_facilities lists too), [] with none: "- upgrade: 3 on offer, best police station (12,20) to
    police HQ, need 318; $7,800" (need: what the observation ranks them by; downtown people, buildings or residents
    without tier-2 cover, or a school's students over its seats)."""
    if dept not in UPGRADE_OFFERS:
        return []
    offers = _offers(obs, dept)
    if not offers:
        return []
    o = offers[0]
    cost = o["cost"]
    need = f", need {_brief(o['need'])}" if o.get("need") else ""
    return [f"- upgrade: {len(offers)} on offer, best {tier_name(o['from'])} ({o['x']},{o['y']}) to "
            f"{tier_name(o['to'])}{need}; {money(cost)} (upgrade_facility, module \"tier\")"]


def edu_room(w, purpose: str) -> tuple[str, str]:
    """(short, note) for a school level's list while education doesn't want it (_wanted): its level enrols ENROL_GOAL
    already, or, for a university, a school level is short (or its own enrolment isn't). v1: each model built 16-22
    elementary schools at ~99% elementary enrolment and 7 universities beside 78-85% high-school enrolment."""
    goal = pct(100 * ENROL_GOAL)
    if purpose != "university":
        text = enrol_text(w, EDU_LEVELS[purpose])
        return (f"{text.split(':')[0]} (goal {goal})", f"{text}: at {goal} or more another school of this level "
                "adds upkeep, not enrolment (the score counts elementary and high-school enrolment)")
    short = [lv for lv in ("elementary", "high") if enrol_short(w, lv)]
    if short:
        why = " and ".join(enrol_text(w, lv).split(":")[0] for lv in short)
        upkeep = money(upkeep_of(w, "education", "university"))
        if is_open():
            return (f"{why} (a university counts in no enrolment the score reads)",
                    f"{why}, under {goal}: elementary and high-school enrolment count in your score, a university's "
                    f"doesn't (it trains graduates for skilled jobs and costs {upkeep}/month)")
        return (f"{why}: schools first (a university counts in no enrolment the score reads)",
                f"{why}, under {goal}: elementary and high-school enrolment count in your score, a university's doesn't "
                f"(it trains graduates for skilled jobs and costs {upkeep}/month); build it once both school levels "
                f"enrol {goal}")
    text = enrol_text(w, "university")
    return (f"{text.split(':')[0]} (goal {goal})", f"{text}: another university adds upkeep, not graduates")


def enrolled_text(w) -> str:
    """Education's state with every school level enrolled (no EDU_LEVELS purpose wanted): "enrolment covered:
    elementary 99%, high-school 92%, university 100%" (university once it has seats)."""
    levels = ["elementary", "high"] + ["university"] * bool(w.stats.get("seats_university", 0))
    return "enrolment covered: " + ", ".join(f"{EDU_LABEL[lv]} {pct(100 * enrolment(w, lv)[0])}" for lv in levels)


def seat_bound(w, level: str) -> bool:
    """More seats would enrol more: the level's seats (at education's funding) are fewer than the students whose homes
    a school reaches (the sim's min(students x coverage, seats x funding) / students)."""
    _, students, seats, cov = enrolment(w, level)
    return seats * w.funding_eff("education") < students * cov - 0.5


def school_upgrade(w, obs: dict, purpose: str) -> Upgrade | None:
    """Education's upgrade for a short school level ("school" or "high_school") whose enrolment is short of seats, not of
    schools in reach (seat_bound): the most overloaded school of the level the observation offers a bigger tier for
    (large elementary from 1,000 residents, large high school from 5,000); None otherwise. Seed 3 at month 96: high
    school enrolment 86% with 92% of the teens' homes in reach and 700 seats for 817 teens."""
    level = EDU_LEVELS[purpose]
    if not enrol_short(w, level) or not seat_bound(w, level):
        return None
    line = "elementary_school" if purpose == "school" else "high_school"
    for o in _offers(obs, "education"):
        if C.TIER_BY_NAME[o["from"]][0] != line:
            continue
        try:
            cur, nxt, _, _ = w.next_tier(o["x"], o["y"])
        except ValueError:
            continue
        return _upgrade(w, "education", o, cur, nxt, enrol_text(w, level), purpose)
    return None


def blocked_firsts(w, dept: str, lists: dict[str, SiteList]) -> dict[str, Price]:
    """`dept`'s unlocked first buildings (Price.first, SUPPLEMENTARY skipped but transport's first link while R and C
    demand are capped without one: capped_unlinked) whose lists are blocked for lack of land or roads
    (SiteList.blocked), priced: purpose -> site_price. A utilities building or a school level only while the city
    needs it (_wanted). Seed 4 at month 96: 6,052 residents, no train station or harbour, and no edge lot by a road
    ("nothing needs it right now" before v2). Neither another building whose coverage is kept up (at_goal) nor a jail
    while the cells have room (_cells_busy), as first_build names them."""
    return {p: pr for p, lst in lists.items() if lst.blocked and _wanted(w, p)
            and (p not in SUPPLEMENTARY or p in LINK_PURPOSES and link_due(w))     # the first link, near the cap
            and not at_goal(w, dept, p) and not not_next(w, dept, p)
            and (pr := site_price(w, dept, p)) and pr.first}


def _wanted(w, purpose: str) -> bool:
    """False for a utilities purpose (SUPPLY_OF) while its supply has room: demand under BANK_BUSY of it now and in
    AHEAD_MONTHS at its growth (_busy), and for a
    pump or outlet the city has one (a fresh map, supply 0, wants them all); True for other purposes. Smoke v14 seed 1,
    months 6 and 9: pump and outlet built, water 370-490 and sewage 250 against demand of 99-116, the bank lots beside
    the roads taken, and planning read "blocked for lack of land: utilities (pump, sewage) ... dezone a roadside lot for
    them" (its playbook frees a bank lot when asked); their own lists still say the bank is taken. Smoke v15: all 8
    leave-free lines named a water tower with water never short (e.g. 125 of 250): the near_homes list ranks every
    roadside lot by the homes around it. A riverbank 2x2 building by the supply of the one pricing it (water works:
    water): first_build counts only wanted ones, and with the tower and turbine left out it would name that $12k.
    Education's school levels go by enrolment (EDU_LEVELS): an elementary or high school while its level enrols under
    ENROL_GOAL (enrol_short), a university only once both school levels reach it and while its own enrolment is short;
    v1 named the first university ahead of 78-85% high-school enrolment on 7 of 8 seeds."""
    if purpose in EDU_LEVELS:
        if purpose == "university":
            return not (enrol_short(w, "elementary") or enrol_short(w, "high")) and enrol_short(w, "university")
        return enrol_short(w, EDU_LEVELS[purpose])
    if purpose == "riverbank_2x2":
        pr = site_price(w, "utilities", purpose)
        return pr is None or _busy(w, supply_of(pr.name))
    if purpose not in SUPPLY_OF:
        return True
    supply, bank = SUPPLY_OF[purpose]
    return bank in _bank_needed(w) or _busy(w, supply)


def _busy(w, supply: str) -> bool:
    """Demand is at least BANK_BUSY of the `supply` ("power", "water", "sewage") now or within AHEAD_MONTHS at its
    growth (demand_ahead)."""
    return demand_ahead(w, supply) >= BANK_BUSY * w.metrics()[f"{supply}_supply"]


def demand_ahead(w, supply: str) -> float:
    """The `supply`'s demand AHEAD_MONTHS from now at its growth over the last AHEAD_MONTHS months of history (fewer
    early on, scaled to AHEAD_MONTHS; no growth when it fell or before two months are recorded): demand now plus that
    growth. Region run 20261005 (trinity in both cities, 120 months, act_every 3): 28 money asks were held back on
    "supply has room", 16 of them right after utilities built; 5 of those supplies reached 99%+ used before its next
    turn, and all 5 had demand plus the last quarter's growth at 90% or more of the supply (13 of the 28 did).
    Growth that persists only: each month's demand is taken without the season / event multiplier it was computed
    under (metrics' demand_factor: summer water x1.15, winter power x1.2, a heatwave's x1.2-1.3), and the growth put
    back at today's, so spring-to-summer or a heatwave's first month reads as no growth (a review found utilities
    building for a temporary spike)."""
    m = w.metrics()
    now = m[f"{supply}_demand"]
    h, key = w.history, f"{supply}_demand"
    span = min(AHEAD_MONTHS, len(h) - 1)
    if span <= 0:
        return now
    base = lambda x: x[key] / (x.get("demand_factor", {}).get(supply) or 1.0)
    grew = (base(h[-1]) - base(h[-1 - span])) * (m.get("demand_factor", {}).get(supply) or 1.0)
    return now + max(0.0, grew * AHEAD_MONTHS / span)


def supply_ahead(w, supply: str) -> str:
    """The share of the `supply` used now and in AHEAD_MONTHS at its growth (demand_ahead), for utilities' no-ask line:
    "water 84% used, 88% in 3 months"; "" without supply."""
    have = w.metrics()[f"{supply}_supply"]
    if have <= 0:
        return ""
    now, ahead = w.metrics()[f"{supply}_demand"], demand_ahead(w, supply)
    return f"{supply} {pct(100 * now / have)} used, {pct(100 * ahead / have)} in {months(AHEAD_MONTHS)}"


def supply_need(w, supply: str) -> float:
    """What a short `supply` wants added: max(SUPPLY_MARGIN x demand, demand + SUPPLY_GROWTH) less what the city makes
    itself. Service imports are left out: they cover at most IMPORT_SHARE_MAX of demand, are bought by the unit and
    stop once the city's own output covers it (a fresh map's power 1.5 of 6 is all imports)."""
    m = w.metrics()
    demand = m[f"{supply}_demand"]
    return max(SUPPLY_MARGIN * demand, demand + SUPPLY_GROWTH[supply]) - own_supply(m, supply)


def supply_output(name: str, supply: str, site: dict) -> float:
    """What a building of the tier `name` adds to `supply` at `site`: its tier's output, a wind turbine's times the
    site's wind strength (the windy list's "wind", as World.plant_output_factor applies it)."""
    line, tier = C.TIER_BY_NAME[name]
    out = float(C.TIER_LINES[line][tier - 1].get(supply) or 0)
    return out * float(site.get("wind", 1.0)) if line == "wind" else out


def pick_supply(need: float, options: list[tuple]) -> tuple | None:
    """Of `options` (purpose, Price, output, site), the cheapest whose output covers `need`, else the one with the
    most output (the cheaper on a tie); None without options. Only one-tile first buildings (Price.first, size 1x1)
    while one is ready; riverbank_2x2's water works, advanced treatment or nuclear plant only when none is. Review,
    scripted seed 0 month 90 (11,788 residents, water at 1,235 of 1,235): past 10,000 residents water wants 700 or
    more, which only the $12,000 water works (2,000) covers, so the line named it over six ready $1,500 pumps (main
    named a pump), the mayor read "needs $12k for another water works" at the top of the to-fund line, and it would
    have taken water to 2.5x demand, far past the playbook's 10-60% margin."""
    small = [o for o in options if o[1].first and tuple(o[1].size) == (1, 1)] or options
    covering = [o for o in small if o[2] >= need]
    if covering:
        return min(covering, key=lambda o: o[1].cost)
    return max(small, key=lambda o: (o[2], -o[1].cost), default=None)


def footprint_tiles(site: dict, size: tuple[int, int]) -> set[tuple[int, int]]:
    """The tiles a building of footprint `size` (w, h) covers with its top-left tile at `site`."""
    return {(site["x"] + dx, site["y"] + dy) for dy in range(size[1]) for dx in range(size[0])}


class SupplyBuild(NamedTuple):
    """One short supply (short_supplies) and the building utilities' needs line and short-now line name for it."""
    supply: str                 # "power", "water" or "sewage"
    have: float                 # the supply as the metrics give it (imports in, as the City block and Key metrics)
    demand: float
    need: float                 # what it wants added (supply_need)
    purpose: str | None = None  # the ready purpose picked (pick_supply); None when nothing is ready for it
    price: Price | None = None
    output: float = 0.0         # what that building adds (supply_output)
    site: dict | None = None    # where: that purpose's first site no earlier pick took
    sites: int = 0              # that purpose's ready sites (up to MAX_SITES)
    blocked: Price | None = None    # with nothing ready: its cheapest first building blocked for lack of land or
                                    # roads (blocked_firsts), which utilities keeps money for
    imported: float = 0.0       # the part of `have` bought in (service imports): supply_need leaves it out
    blocked_fix: str = ""       # what `blocked` needs beyond a roadside lot (land_fix: "a road to the river")
    ahead: float | None = None  # demand in AHEAD_MONTHS at its growth (demand_ahead); None: as `demand`

    @property
    def status(self) -> str:
        """The supply against demand, imports in (as the City block shows them): short of demand "power 25% covered",
        else the share used, as build_status says it ("power 98% used"; imports filling a gap read "100% used": at
        capacity, where "100% covered" read as nothing to do), and when it is wanted only for its growth (under
        BANK_BUSY used now, at it within AHEAD_MONTHS: _busy) that share then: "water 84% used, 92% in 3 months"."""
        if self.demand <= 0:
            return f"no {self.supply} demand yet"
        if self.have < self.demand:
            return f"{self.supply} {pct(100 * self.have / self.demand)} covered"
        used = f"{self.supply} {pct(100 * self.demand / self.have)} used"
        if self.ahead is not None and self.demand < BANK_BUSY * self.have <= self.ahead:
            return used + f", {pct(100 * self.ahead / self.have)} in {months(AHEAD_MONTHS)}"
        return used


def short_supplies(w, lists: dict[str, SiteList]) -> list[SupplyBuild]:
    """Each supply with a utilities purpose it wants (_wanted: demand at BANK_BUSY of its supply now or within
    AHEAD_MONTHS, or a pump or outlet
    the city lacks), in SUPPLIES order, with the building to name for it: of its ready purposes (ready_prices), each at
    its first site no earlier pick took, pick_supply's for supply_need. Supplies whose ready purposes all need the
    riverbank pick first (an outlet can go nowhere else: with one free bank lot it gets the lot and water a tower),
    then in SUPPLIES order (a coal plant and a tower ranked on the same lot take two). A supply with nothing ready
    carries its cheapest blocked first building (blocked_firsts). Utilities' needs line (first_build) and short-now
    line (briefing) read it, and bank_keep the lots it picks. Smoke v20 seed 1: the needs line named only the cheapest
    wanted building; at month 3 utilities, holding $4,400, built nothing."""
    ready, blocked = ready_prices(w, "utilities", lists), blocked_firsts(w, "utilities", lists)
    short, cands, stuck = [], {s: [] for s in SUPPLIES}, {s: [] for s in SUPPLIES}
    for p in SITE_PURPOSES["utilities"]:
        pr = site_price(w, "utilities", p)
        if pr is None or (s := supply_of(pr.name)) is None or not _wanted(w, p):
            continue
        short += [s] * (s not in short)
        if p in ready:
            cands[s].append(p)
        elif p in blocked:
            stuck[s].append((blocked[p].cost, p))
    bank_only = lambda s: bool(cands[s]) and all(p in RIVERBANK_PURPOSES for p in cands[s])
    need = {s: supply_need(w, s) for s in short}
    picks, taken = {}, set()
    for s in sorted(short, key=lambda s: (not bank_only(s), SUPPLIES.index(s))):
        opts = []
        for p in cands[s]:
            site = next((x for x in lists[p].sites if not footprint_tiles(x, ready[p].size) & taken), None)
            if site is not None:
                opts.append((p, ready[p], supply_output(ready[p].name, s, site), site))
        picks[s] = pick_supply(need[s], opts)
        if picks[s]:                     # a 2x2 building's whole footprint: an outlet can't go under a water works
            taken |= footprint_tiles(picks[s][3], picks[s][1].size)
    m = w.metrics()
    out = []
    for s in (s for s in SUPPLIES if s in short):
        base = SupplyBuild(s, m[f"{s}_supply"], m[f"{s}_demand"], need[s], imported=m["service_imports"].get(s, 0.0),
                           ahead=demand_ahead(w, s))
        if (pick := picks[s]) is not None:
            p, pr, output, site = pick
            out.append(base._replace(purpose=p, price=pr, output=output, site=site,
                                     sites=len(lists[p].sites[:MAX_SITES])))
        elif stuck[s]:
            p = min(stuck[s])[1]
            out.append(base._replace(blocked=blocked[p], blocked_fix=land_fix(w, "utilities", p)))
        else:
            out.append(base)
    return out


def bank_keep(w, builds: list[SupplyBuild]) -> dict[tuple[int, int], str]:
    """The scarce free riverbank lots (scarce_bank) -> what utilities' Ready sites say each is kept for: the pump or
    sewage purpose short_supplies picked it for, else "pump/sewage" (as other roles' lines say). Smoke v20 seed 1,
    month 3: its one lot read "(riverbank: keep for pump/sewage)" on the wind line and "same site as wind" on the pump
    and sewage lines, and utilities spent its turn inspecting sewage sites."""
    picked = {(b.site["x"], b.site["y"]): b.purpose for b in builds if b.purpose in BANK_TOP_UP}
    return {xy: picked.get(xy, "pump/sewage") for xy in sorted(scarce_bank(w))}


def dept_lists(env) -> dict[str, dict[str, SiteList]]:
    """Each department in the game with a priced purpose -> its site lists, one observation each: planning's blocked
    (land_blocked) and leave-free (funded_firsts) lines read them."""
    w = env.world
    return {d: site_lists(w, d, role_obs(env, d)) for d in DEPARTMENTS
            if d in env.roles and any(site_price(w, d, p) for p in SITE_PURPOSES[d])}


def land_blocked(env, lists: dict[str, dict[str, SiteList]] | None = None) -> dict[str, list[str]]:
    """Each department with first buildings blocked for lack of land or roads -> the cheapest of them (blocked_firsts;
    every one on a tie, as the mayor's line names them), from `lists` (dept_lists by default): planning's Ready sites
    block names them. Review probes, months 24, 60 and 120: naming every one, the line always fell back to the
    department names alone."""
    w = env.world
    out = {}
    for d, ls in (dept_lists(env) if lists is None else lists).items():
        if found := blocked_firsts(w, d, ls):
            out[d] = list(cheapest(found, supplementary=True)[0])      # transport's first link is one
    return out


def land_fix(w, dept: str, purpose: str) -> str:
    """What a purpose blocked for lack of land needs beyond a free roadside lot, "" for none: a road to the river
    while no connected road reaches it, else a free riverbank lot (RIVERBANK_PURPOSES); a free 2x2 lot (TWO_BY_TWO);
    "near homes" for NEAR_NEED while free roadside land is left (then that is why it is blocked: _why_empty); "away
    from homes" for AWAY_FROM_HOMES (a lot freed among the homes doesn't do). Planning's blocked line tags them: smoke
    v14 seed 1 read "utilities (pump, sewage): extend a road into open land or dezone a roadside lot for them"; v15
    seed 1, months 6-9: health's landfill had "no roadside lot away from homes", planning left none at the town edge
    and garbage coverage stayed 0 all run."""
    if purpose in AWAY_FROM_HOMES:
        return "away from homes"
    if purpose == "train_station":
        return "a road to the map edge"
    if purpose in RIVERBANK_PURPOSES:
        if not _riverbank_by_roads(w)[1]:
            return "a road to the river"
        return "a free 2x2 riverbank lot" if purpose == "riverbank_2x2" else "a free riverbank lot"
    if (dept, purpose) in NEAR_NEED and _free_roadside(w):
        return "near homes"
    return "a free 2x2 lot" if purpose in TWO_BY_TWO else ""


def edge_site(w, roads: list[dict]) -> dict | None:
    """The first of planning's road_extension sites (a dead end's extension or a branch road) whose far end (x2, y2)
    has under 3 homes within 4 tiles (observation._far_from_homes): where to leave a lot for a building that must
    stay away from homes (AWAY_FROM_HOMES); None without one."""
    near = _far_from_homes(w)
    return next((s for s in roads if not near[s["y2"], s["x2"]]), None)


def _away_once_dezoned(w, tiles: set[tuple[int, int]]) -> bool:
    """Whether a lot would be away from homes as the AWAY_FROM_HOMES lists want it (observation._far_from_homes: under
    3 homes within 4 tiles of its site, the top-left tile) once its `tiles` are dezoned."""
    x, y = min(tiles)
    homes = int(np.isin(w.struct[max(0, y - 4):y + 5, max(0, x - 4):x + 5], C.HOMES).sum())
    return homes - sum(int(w.struct[ty, tx]) in C.HOMES for tx, ty in tiles) < 3


def funded_firsts(w, dept: str, lists: dict[str, SiteList]) -> dict[str, Price]:
    """`dept`'s cheapest ready primary first building(s) on a lot (Price.first; SUPPLEMENTARY, PER_TILE and road-tile
    sites such as bus stops skipped; a utilities building only while _wanted), all on a tie, when its funds cover
    them (a top-up this month counts): purpose -> price; else {}. Planning's leave-free line names them (claim_spots)
    and a zone that leaves one no site warns. Smoke v14 seed 0, months 3-6: the mayor funded fire, health and
    education for their ready sites near homes, planning zoned those lots on its turn, and none of them built."""
    prices = {p: pr for p, pr in ready_prices(w, dept, lists).items()
              if pr.first and not pr.per_tile and _wanted(w, p) and not not_next(w, dept, p)
              and _on_lot(w, lists[p].sites[0])}
    low = cheapest(prices)
    return {} if low is None or short_of(w.dept_funds[dept], low[1].cost) else {p: prices[p] for p in low[0]}


def _funded_of(w, dept: str, lists: dict[str, SiteList]) -> dict[str, tuple[Price, list]]:
    """purpose -> price and ready sites (x, y) of each of `dept`'s funded_firsts buildings."""
    return {p: (pr, [(s["x"], s["y"]) for s in lists[p].sites]) for p, pr in funded_firsts(w, dept, lists).items()}


def may_fund(w, dept: str) -> bool:
    """Whether `dept`'s funds cover its cheapest priced first building on a lot (SUPPLEMENTARY and PER_TILE purposes
    skipped): only then can funded_firsts name one, so only then is its observation worth building."""
    low = min((pr.cost for p in SITE_PURPOSES[dept] if p not in SUPPLEMENTARY
               and (pr := site_price(w, dept, p)) and pr.first and not pr.per_tile), default=None)
    return low is not None and not short_of(w.dept_funds[dept], low)


def _on_lot(w, site: dict) -> bool:
    return bool(w.struct[site["y"], site["x"]] == C.NONE and w.terrain[site["y"], site["x"]] == C.LAND)


class Spot(NamedTuple):
    dept: str
    purpose: str
    building: str               # Price.building
    xy: tuple[int, int]         # the site (a multi-tile building's top-left tile)
    size: tuple[int, int] = (1, 1)

    @property
    def tiles(self) -> set[tuple[int, int]]:
        return {(self.xy[0] + i, self.xy[1] + j) for i in range(self.size[0]) for j in range(self.size[1])}


def funded_sites(w, depts: dict[str, dict[str, SiteList]]) -> dict[tuple[str, str], tuple[Price, list]]:
    """(department, purpose) -> price and ready sites (x, y) of each funded_firsts building, in turn order."""
    return {(d, p): found for d, lists in depts.items() for p, found in _funded_of(w, d, lists).items()}


def claim_spots(funded: dict[tuple[str, str], tuple[Price, list]], keep: dict | None = None) -> dict[tuple, Spot]:
    """Each funded building's site (funded_sites), in turn order: its first one whose footprint no earlier claim
    covers; none when every one is claimed. With `keep` (an earlier claim), the buildings whose claimed site is still
    listed go first, each trying that site before the rest of its list, so only a building whose site went moves.
    Smoke v15 seed 1, month 6: "leave free (funded): water tower (22,23), police station (34,25), fire station (34,25),
    clinic (34,25), elementary school (34,25)", one lot for four buildings. Review: taking the first unclaimed site in
    list order, a kept police station moved onto a lot listed ahead of its own and pushed fire off the one it kept."""
    taken, out = set(), {}
    order = [k for k in funded if keep and k in keep and keep[k].xy in funded[k][1]]
    for k in order + [k for k in funded if k not in order]:
        pr, sites = funded[k]
        if k in order:
            sites = [keep[k].xy] + [xy for xy in sites if xy != keep[k].xy]
        for xy in sites:
            spot = Spot(*k, pr.building, xy, pr.size)
            if not spot.tiles & taken:
                taken |= spot.tiles
                out[k] = spot
                break
    return {k: out[k] for k in funded if k in out}


def funded_spots(w, depts: dict[str, dict[str, SiteList]]) -> list[Spot]:
    """Each funded building's claimed site (claim_spots): planning's leave-free line names them and its zone strips
    spare them (spare_funded)."""
    return list(claim_spots(funded_sites(w, depts)).values())


def unclaimed_spots(funded: dict[tuple[str, str], tuple[Price, list]], claims: dict[tuple, Spot]) -> list[Spot]:
    """Each funded building (funded_sites) that claim_spots (`claims`) seats nowhere, every site of it under an earlier
    department's claim, at its best listed site: planning's leave-free line says it has no lot of its own. Smoke v18
    seed 1, month 6: the mayor funded education's first school, its only site (38,25) was police's too, and planning
    read only "leave free (funded): police station (38,25)"; education never built (33.4)."""
    return [Spot(*k, pr.building, sites[0], pr.size) for k, (pr, sites) in funded.items() if k not in claims and sites]


def unclaimed_firsts(depts: dict[str, DeptMoney]) -> dict[str, tuple[str, ...]]:
    """Each department all of whose funded first buildings (FirstBuild.funded) claim_spots seats nowhere, in turn order
    across departments (unclaimed_spots) -> their names ("an elementary school"): the mayor's line calls it blocked,
    as it does a first building blocked for lack of land (one of them seated can go up: not blocked). Only those whose
    funds may cover one (may_fund) build their observation (DeptMoney.build). The money is kept already: a ready first
    building counts in DeptMoney.keep."""
    funded = {(d, p): found for d, m in depts.items() if m.priced and may_fund(m.env.world, d)
              and m.build and m.build.funded for p, found in m.build.funded.items()}
    claims, out = claim_spots(funded), {}
    for d in dict.fromkeys(d for d, _ in funded):
        mine = [k for k in funded if k[0] == d]
        if not any(k in claims for k in mine):
            out[d] = tuple(dict.fromkeys(a_or_an(funded[k][0].building) for k in mine))
    return out


def spare_funded(lists: dict[str, SiteList], spots: list[Spot]) -> dict[str, SiteList]:
    """Planning's lists with its zone strips sparing the funded sites (funded_spots, whole footprints): strips covering
    none first, and those covering some marked "leave": "(20,22) for the fire station" ("(10,18), (20,22) free" for
    several), as a bank lot is kept for pump / sewage. Planning zones what it is suggested: next to "leave free
    (funded): fire station (20,22)", its best zone_R strip was (20,20)-(21,22)."""
    keep = {}
    for sp in spots:
        keep.setdefault(sp.xy, (set(), []))
        keep[sp.xy][0].update(sp.tiles)
        keep[sp.xy][1].append(sp.building)
    if not keep:
        return lists
    out = dict(lists)
    for p, lst in lists.items():
        if not p.startswith("zone_"):
            continue
        marked = []
        for s in lst.sites:
            hit = [xy for xy, (tiles, _) in keep.items()
                   if any(s["x1"] <= x <= s["x2"] and s["y1"] <= y <= s["y2"] for x, y in tiles)]
            where = ", ".join(f"({x},{y})" for x, y in hit)
            marked.append({**s, "leave": f"{where} for the {', '.join(keep[hit[0]][1])}" if len(hit) == 1 else
                           f"{where} free"} if hit else s)          # several: the leave-free line names them
        out[p] = lst._replace(sites=sorted(marked, key=lambda s: "leave" in s))
    return out


def _cells_busy(w) -> bool:
    """Inmates past JAIL_BUSY of the cells (jails and police stations' holding cells), as the police playbook and the
    scripted police build a jail: only then does a first jail come before another station on the needs line (jail
    sites are listed whatever the inmates)."""
    st = w.stats
    return st.get("inmates", 0.0) > JAIL_BUSY * max(1.0, st.get("jail_capacity", 0.0))


def not_next(w, dept: str, purpose: str, sites=()) -> bool:
    """A ready purpose first_build doesn't name by its price (the briefing's "can't pay" line skips it too): police's
    jail while the cells have room (_cells_busy: with every station kept up, keep_up_room, the cheapest ready site
    would read "its first jail"), health's hospital at its ready `sites` (site dicts) while health coverage is at
    KEEP_UP_GOAL or none of them adds KEEP_UP_GAIN (hospital_gain), as keep_up_room rules for a clinic."""
    if dept == "police" and purpose == "jail":
        return not _cells_busy(w)
    if dept == "health" and purpose == "hospital":
        return (w.metrics()["health_coverage"] >= KEEP_UP_GOAL
                or hospital_gain(w, [(s["x"], s["y"]) for s in sites]) < KEEP_UP_GAIN)
    return False


def hospital_gain(w, sites) -> float:
    """The most health coverage a hospital built at one of `sites` ((x, y) top-left tiles, the first TIER_UP_TRIED)
    adds, as tier_up estimates a fresh one (the sim's coverage with it placed); 0 without sites. Cached like keep_up."""
    sites = tuple(tuple(s) for s in sites)[:TIER_UP_TRIED]
    cache = _keep_up_cache(w)
    k = ("hospital", sites)
    if k not in cache:
        model = _Coverage(w, "health", C.TIER_LINES["clinic"][0])
        base, hosp = model.value()[0], C.TIER_LINES["clinic"][1]
        fw, fh = hosp["footprint"]
        cache[k] = max((model.value(up=(None, hosp, [(x + i, y + j) for j in range(fh) for i in range(fw)]))[0]
                        for x, y in sites), default=base) - base
    return cache[k]


def _label(w, pr: Price, n: int = 1) -> tuple[str, int | None]:
    """(what, buildings_of) for a Need of `pr`: the building, counted ("police station", 1); a per-tile one whole and
    uncounted ("3 avenue tiles", None; "a levee tile"); one that is never a first building (Price.first False) while
    the city has none uncounted too ("a university", None), so no line reads "its first" for it."""
    have = None if pr.per_tile else buildings_of(w, pr.name)
    if have == 0 and not pr.first:
        have = None
    return (pr.building if have is not None else f"{n} {pr.building}s" if n != 1 else a_or_an(pr.building)), have


def first_build(env, dept: str) -> FirstBuild | None:
    """`dept`'s next build from one observation, the cheapest ready first building the city lacks (Price.first,
    buildings_of 0; SUPPLEMENTARY and per-tile purposes skipped), else its cheapest ready site: the mayor's needs line
    names it. Review, seed 0, months 42-69: a $30 avenue stretch was transport's cheapest site, so its first $1,500 bus
    line (0 bus lines) got no needs line, no to-fund entry and "funds cover its next build"; education's first high
    school and university hid behind another elementary school. With it the dearest ready or blocked first building
    (and what it is, keep_for), which DeptMoney keeps: review replay (seed 0, month 9), utilities held $3,000 for a
    $2,500 turbine with power at capacity, read idle above its $1,200 outlet, and lost it; smoke v13 seed 0, month 6,
    police, fire, health and education held their first building's price with every roadside lot zoned, read idle
    above the $1,000 floor, and lost it. A tier 2+ purpose (hospital, riverbank_2x2's water works) is an upgrade, not
    a first building. A per-tile purpose counts its best site whole ("$90 for 3 avenue tiles", best_stretch), also
    when picking the cheapest: one $30 tile undersold an upgrade. A utilities building counts only while its supply is
    short (_wanted): smoke v14-v15 seed 1 read "needs $2,200 for a water tower" with water spare, and towers were built
    to 4.7-4.9x demand. Every building at that price is named (Need: first or another, buildings_of), with the ready
    sites of the first purpose pricing it; at a tie with a first one, the others go (health's clinic built: its first
    landfill alone). Each Need carries what the building adds in upkeep a month (Price.upkeep; a per-tile one for its
    whole stretch, bridges twice). Its funded_firsts ride along (FirstBuild.funded) for unclaimed_firsts. Utilities
    names one building per short supply instead, bought together (_supply_build). Education's school levels count only
    while their enrolment is short (_wanted: a university after both school levels). civitas-v2: in place of another
    station, clinic or school (never of a first building, a levee, or a landfill while garbage is short), police, fire
    and health name the tier upgrade (or health's fresh hospital) that adds the most coverage when it adds more than one
    more tier-1 building (tier_up), education a short level's bigger school while it lacks seats (school_upgrade): one
    Need with its exact call (Need.call), priced as the action charges it, "another" on the mayor's lines. Transport's
    first link to the region is a first building once demand is capped without one (capped_unlinked), before another
    bus line or an avenue; with nothing else ready its station, harbour or metro station is a priced build ("another"),
    no longer a cost-less extra. With nothing else ready or blocked, a FirstBuild without a cost says why when something
    was ready: utilities sites whose supply has room (`room`), every school level enrolled (`covered`), a supplementary
    building (`extra`); else None. civitas-v2 review: another police / fire station, clinic or landfill is ready only
    while its coverage is under KEEP_UP_GOAL and a ready site adds KEEP_UP_GAIN (keep_up_room; tier_up keeps the same
    goal), else police, fire and health read their coverage as `covered` (and `gap` when no site reaches the uncovered
    homes); a jail only while the cells are busy, a hospital only as tier_up's; a university is never a first building
    (NOT_FIRST: "a university", another); transport's first link is a first building from LINK_AHEAD of the cap
    (link_due), after a bus line its list offers until the cap is reached."""
    w = env.world
    obs = role_obs(env, dept)
    lists = site_lists(w, dept, obs)
    blocked = blocked_firsts(w, dept, lists)
    ready = ready_prices(w, dept, lists)
    prices = {p: pr for p, pr in ready.items() if _wanted(w, p) and not not_next(w, dept, p, lists[p].sites)}
    tiles = {}
    for p, pr in list(prices.items()):
        if pr.per_tile:                  # the whole stretch; a bridge tile charged BRIDGE_UPGRADE times costs twice
            tiles[p], units = best_stretch(w, p, lists[p].sites)     # the upkeep (World.road_upgrade_upkeep)
            if not tiles[p]:             # every listed stretch is an avenue already: nothing to buy there
                del prices[p], tiles[p]
                continue
            bridges = (units - tiles[p]) / (BRIDGE_UPGRADE - 1)
            prices[p] = pr._replace(cost=pr.cost * units, upkeep=pr.upkeep * (tiles[p] + bridges))
    labels = {p: _label(w, pr, tiles.get(p, 1)) for p, pr in prices.items()}
    capped = dept == "transport" and capped_unlinked(w)
    unlinked = dept == "transport" and link_due(w)  # from LINK_AHEAD of the cap: capped included
    for p in LINK_PURPOSES if unlinked else ():      # the first link to the region: "its first train station"
        if p in prices:
            labels[p] = (prices[p].building, 0)
    # never below the needs line: with only an upgrade ready (health's $8,000 hospital) money saved for it isn't idle
    label = lambda what, have: what if have is None else f"its first {what}" if not have else f"another {what}"
    firsts = [(pr.cost, label(*labels[p])) for p, pr in prices.items() if p not in SUPPLEMENTARY and pr.first] + [
        (pr.cost, label(*_label(w, pr))) for pr in blocked.values()]
    if dept == "utilities" and (supply := _supply_build(w, lists, firsts, label)) is not None:
        return supply._replace(funded=_funded_of(w, dept, lists))
    lacking = {p: pr for p, pr in prices.items()                                         # cheapest skips SUPPLEMENTARY
               if pr.first and labels[p][1] == 0}
    # transport's first link to the region (a train station or harbour) once the city is capped without one (R and C
    # demand at CAP_LEVEL from CAP_POP_CONNECTION residents: World.demand_caps): a first building, before another bus
    # line or an avenue (after the first bus line, a lacking first building); from LINK_AHEAD of the cap (link_due) a
    # first building after a bus line its list offers, before an avenue; below that it is a station like any
    link = cheapest({p: prices[p] for p in LINK_PURPOSES if p in prices}, supplementary=True) if unlinked else None
    if link:                                         # kept for also while a bus line comes first: not idle
        firsts.append((link[1].cost, label(*labels[link[0][0]])))
    # transport's next bus line, while its list offers one (bus_room: transit under TRANSIT_GOAL and the line adds
    # riders, or full buses), before any avenue stretch: the 120-month LLM runs named "$30 for an avenue tile" on 131
    # of the mayor's transport lines and another bus line on 9, transport got 0.8-3.2% of the mayor's top-ups, and
    # transit fell to 1-3% (briefing._transport_next shows transport this same build)
    line = (("bus_stops",), prices["bus_stops"]) if dept == "transport" and "bus_stops" in prices else None
    # transport's station, harbour or metro station with nothing else ready: a priced build ("another" on the mayor's
    # lines), no longer a cost-less extra. v1 (4 models x 8 seeds, months 96-107): the Next line named "a train station
    # ($8,000)" or "a harbour ($10k)" in about half the turns that had one, with transport's funds short, while the
    # mayor had $17-27k above the reserve: kind "nothing", so no to-fund item and no ask, and none was ever funded.
    # site_lists lists them only when needed (link_room: no link yet, a first train station, or freight running full)
    stations = {p: pr for p, pr in prices.items() if p in STATION_PURPOSES} if dept == "transport" else {}
    first = cheapest(lacking) or (link if capped else None)
    purposes, low = (first or line or link or cheapest(prices) or cheapest(stations, supplementary=True)
                     or ((), None))
    up = None
    if first or low and any(labels[p][1] == 0 for p in purposes):
        pass                                        # a first building keeps its turn
    elif dept in TIER_UP and (low is None or _replaceable(w, dept, purposes)):
        p = TIER_UP[dept][0]                         # another station / clinic: a tier upgrade if it adds more (its
        sites = [(s["x"], s["y"]) for s in lists[p].sites] if p in lists else []   # kept also when they add little:
                                                                                    # keep_up_room)
        fresh = [(s["x"], s["y"]) for s in lists["hospital"].sites] if "hospital" in ready else []
        up = tier_up(w, dept, sites, _offers(obs, dept), fresh)
    elif dept == "education":                        # a short level's bigger school while it lacks seats, not reach
        levels = ([p for p in purposes if p in ("school", "high_school")] if low is not None
                  else [p for p in ("school", "high_school") if _wanted(w, p)])
        up = next((u for p in levels if (u := school_upgrade(w, obs, p)) is not None), None)
    if up is not None:
        need = Need(up.what, 1, None, up.status, False, up.upkeep, call=up.call, note=up.alt)
        dearest = max(firsts + [(up.cost, need.label)], key=lambda t: t[0])
        return FirstBuild(up.cost, (need,), dearest[0], (), _funded_of(w, dept, lists), keep_for=dearest[1],
                          purposes=(up.purpose,))
    if low is None and not blocked:
        extra = cheapest(prices, supplementary=True)
        room = dept == "utilities" and len(prices) < len(ready)
        covered = (enrolled_text(w) if dept == "education" and not any(_wanted(w, p) for p in EDU_LEVELS if p in ready
                                                                        or p != "university") else "")
        gap, stuck = None, ""
        if dept in KEEP_UP_GATED:                    # every kind kept up (keep_up_room): its coverage, or where it lacks
            kept = [(p, room_) for d, p in KEEP_UP if d == dept and p in lists
                    and (room_ := keep_up_room(w, dept, p, lists[p].sites)) is not None]
            covered = "; ".join(r[0] for _, r in kept)
            gaps = [(r[0], ku) for p, r in kept if not at_goal(w, dept, p)
                    and (ku := keep_up(w, dept, p, [(s["x"], s["y"]) for s in lists[p].sites])) is not None
                    and ku.gap is not None]
            if gaps:
                gap, stuck = gaps[0][1].gap, f"{gaps[0][0]}: {stuck_note(gaps[0][1])}"
        if not (room or extra or covered):
            return None
        return FirstBuild(None, (), 0.0, (), None, room=room,
                          extra=(extra[1].building, extra[1].cost) if extra else None,
                          purposes=extra[0][:1] if extra else (), covered=covered, gap=gap, stuck=stuck)
    named, which = {}, {}
    for p in purposes:
        pr, (what, have) = prices[p], labels[p]
        which.setdefault(what, p)
        named.setdefault(what, Need(what, len(lists[p].sites[:MAX_SITES]), have,
                                    build_status(w, dept, p, pr.name) if have else "", have == 0 and pr.first,
                                    pr.upkeep, tiles=pr.per_tile))
    if any(nd.have == 0 for nd in named.values()):
        named = {k: nd for k, nd in named.items() if not nd.have}
    if low:
        firsts.append((low.cost, next(iter(named.values())).label))
    dearest = max(firsts, key=lambda t: t[0])                   # the first of the dearest, as max() reads them
    shut = () if low else tuple(dict.fromkeys(a_or_an(blocked[p].building) for p in cheapest(blocked, True)[0]))
    spots = tuple(tuple((int(s["x"]), int(s["y"])) for s in lists[which[k]].sites[:MAX_SITES])
                  if (dept, which[k]) in KEEP_UP else () for k in named)
    return FirstBuild(low.cost if low else None, tuple(named.values()), dearest[0], shut, _funded_of(w, dept, lists),
                      keep_for=dearest[1], purposes=tuple(which[k] for k in named),
                      sites=spots if any(spots) else ())


def _supply_build(w, lists: dict[str, SiteList], firsts: list[tuple[float, str]], label) -> FirstBuild | None:
    """Utilities' first_build: one building per short supply (short_supplies) bought together (FirstBuild.all_of, the
    cost their sum), each Need with its supply's cover as its status ("power 25% covered"), in SUPPLIES order; None
    when no short supply has a ready one (first_build's own path then says blocked, room or nothing). It keeps those
    buildings and each blocked supply's cheapest first building together (the pump and outlet before the road reaches
    the river), or the dearest single first building (`firsts`) when that is more. Smoke v20 seed 1, month 0: "needs
    $2,200 for its first water tower" with power at 25% and sewage at 0%; the mayor gave $2,200, and at month 6
    utilities' $4,400, above the $4,000 coal plant, read "funds cover its builds" with power at 17%."""
    builds = short_supplies(w, lists)
    if not any(b.purpose for b in builds):
        return None
    named, held = [], []
    for b in builds:
        if b.purpose:
            have = buildings_of(w, b.price.name)
            named.append(Need(b.price.building, b.sites, have, b.status, have == 0 and b.price.first, b.price.upkeep))
            held.append((b.price.cost, named[-1].label))
        elif b.blocked is not None:
            held.append((b.blocked.cost, label(*_label(w, b.blocked))))
    together = (sum(c for c, _ in held), labels_and([lab for _, lab in held]))
    dearest = max([together] + firsts, key=lambda t: t[0])      # all of them on a tie
    waits = tuple(f"{b.blocked.building} waits for {b.blocked_fix or 'a free roadside lot'}" for b in builds
                  if not b.purpose and b.blocked is not None)
    return FirstBuild(sum(b.price.cost for b in builds if b.purpose), tuple(named), dearest[0], (),
                      keep_for=dearest[1], all_of=True, waits=waits)


def road_cost(site: dict) -> float:
    """What a road_extension site costs, as the actions charge it: a dead end's extension (observation) is build_road
    over its `tiles` from the dead end, `bridge_tiles` of them on the river; a branch road (Branch.site) carries its
    `cost`, bulldozing included (dezoning is free)."""
    if "cost" in site:
        return site["cost"]
    bridges = site.get("bridge_tiles", 0)
    return (site["tiles"] - bridges) * C.CATALOG["road"]["cost"] + bridges * C.CATALOG["bridge"]["cost"]


def zoning_cost(site: dict) -> float:
    """Zoning the land a road_extension site opens (its opens_land) at low density, as zone charges it: road_price adds
    it to the road. Smoke v18 seed 0, month 9: the mayor read "needs $50 for a road into open land (opens 15 tiles)",
    moved exactly $50, and planning had nothing left to zone the land the road opened. A little over what zoning all
    of it charges: opens_land counts the road's own tiles beyond its dead end's reach."""
    return site.get("opens_land", 0) * C.CATALOG["zone"]["cost"]


def road_max(w) -> float:
    """The dearest a road_extension site can be, zoning included: BRANCH_LEN bridge tiles (an extension runs up to 15,
    as a branch), BRANCH_CLEAR tiles to bulldoze and OPENS_MAX tiles to zone."""
    return (BRANCH_LEN * C.CATALOG["bridge"]["cost"] + BRANCH_CLEAR * C.CATALOG["bulldoze"]["cost"]
            + OPENS_MAX * C.CATALOG["zone"]["cost"])


def road_price(lists: dict[str, SiteList]) -> Price | None:
    """Planning's best road_extension site (a dead end's extension, else a branch road) priced: the road (road_cost)
    and zoning the land it opens (zoning_cost), for its Ready sites line ("$50 road + $75 zoning") and the mayor's
    needs line (road_build); None without one. Never in ready_prices or site_price: a road is no building, so it stays
    out of the department budget line, the leave-free line (funded_firsts, claim_spots), the blocked line and
    cheapest()."""
    lst = lists.get("road_extension")
    if not lst or not lst.sites:
        return None
    best = lst.sites[0]
    return Price(road_cost(best) + zoning_cost(best), ROAD_NAME, name="road", zoning=zoning_cost(best))


def road_build(env) -> FirstBuild | None:
    """Planning's needs line from one observation (DeptMoney.build): its best road_extension site, as "a road into open
    land and zoning it (opens 71 tiles)" (Need, neither a first nor another building), costing road_price; None
    without one. The live 120-month run, seed 0, months 9-12: planning at $0 read "road_extension: 1 site, best
    (24,20)-(24,5), opens 71 land tiles", failed on its $150 twice and asked for it, and the mayor's line for it showed
    no need."""
    w = env.world
    lists = {"road_extension": site_lists(w, "planning", role_obs(env, "planning"))["road_extension"]}
    if (pr := road_price(lists)) is None:
        return None
    lst = lists["road_extension"]
    need = Need(a_or_an(ROAD_NAME), len(lst.sites[:MAX_SITES]), None, f"opens {lst.sites[0]['opens_land']} tiles")
    return FirstBuild(pr.cost, (need,), pr.cost, ())


def _overlaps(a: dict, b: dict) -> bool:
    return not (a["x2"] < b["x1"] or a["x1"] > b["x2"] or a["y2"] < b["y1"] or a["y1"] > b["y2"])


ZONE_RULE = "one zone type per strip"


def zone_line(purpose: str, sites: list[dict], strips: dict[str, dict], site: Callable) -> tuple[str, dict | None]:
    """A zone purpose's Ready sites text after its count, and the strip it shows: its best strip no earlier zone line
    shows (`strips`: purpose -> strip, which it adds to; overlapping counts as the same), as "best ..." when that is
    its first, else "next free ..." with ZONE_RULE; with none left, which lines took them. v19 (120 months, seeds
    0-1): "zone_C: best as zone_R; next ...", "zone_I: same strip as zone_R"; planning zoned R on a strip, then I on
    the same rectangle, 19 times ("all 40 tiles are zoned R")."""
    own = next((s for s in sites if not any(_overlaps(s, t) for t in strips.values())), None)
    if own is None:
        took = _and([p + "'s" for p, t in strips.items() if any(_overlaps(s, t) for s in sites)])
        return f"all {took} strips ({ZONE_RULE})" + advice(": extend the roads"), None
    strips[purpose] = own
    if own is sites[0]:
        return f"best {site(purpose, own)}", own
    text = f"next free {site(purpose, own)}"
    return (text + f" ({ZONE_RULE})" if len(purpose) + len(text) + len(ZONE_RULE) <= 88 else text), own


def strip_cost(purpose: str, strip: dict) -> float:
    """A zone strip (a zone_<letter> site) at low density, industry and mixed use at the high-density price, as zone
    charges them: planning's Ready sites zone lines."""
    return strip["tiles"] * zone_price(zone_density(purpose[-1], "low"))


def ready_lines(role: str, lists: dict[str, SiteList], keep_bank: set | dict = frozenset(),
                prices: dict[str, Price] | None = None, funds: float | None = None) -> list[str]:
    """The Ready sites lines: per purpose the count and best site, or the short reason. A best site already shown
    on an earlier line is named once: the line points back to it and shows the next distinct site instead (several
    zone types rank the same strip first); a purpose reading another's list says so. A site in `keep_bank`
    (scarce_bank) shown for a purpose that doesn't need the river is marked to keep for pump / sewage. Utilities'
    `keep_bank` says what each lot is kept for (bank_keep: {(x, y): "sewage"}): every line showing it says so
    ("(riverbank: kept for sewage)", never "same site as"), and a purpose it isn't kept for shows its next site with
    "(not (33,23): riverbank kept for sewage)" (one that needs the river keeps a lot kept for "pump/sewage"). A line
    with sites ends with its price (`prices`: ready_prices, and planning's road_price), and with `funds` (rounded
    down) when they are short of it; so does a reason with sites (planning's branch roads). Smoke v20 seed 1, month 3:
    "wind: best (33,23) (riverbank: keep for pump/sewage)", "pump: same site as wind", "sewage: same site as wind";
    utilities spent the turn inspecting sewage sites. Zone purposes (planning's strips) each show a strip of their own
    instead (zone_line: one zone type per strip)."""
    lines, shown, paths = [], {}, {}                # shown: where -> purpose; paths: hint path -> purpose
    strips: dict[str, dict] = {}                     # zone purpose -> the strip its line shows (zone_line)
    kept = keep_bank if isinstance(keep_bank, dict) else None

    def kept_for(s: dict) -> str | None:
        return kept.get((s["x"], s["y"])) if kept and "x2" not in s else None

    def mine(purpose: str, s: dict) -> bool:
        """`s` may be shown as `purpose`'s best: not a lot kept for another purpose."""
        k = kept_for(s)
        return k is None or k == purpose or (k == "pump/sewage" and purpose in RIVER_PURPOSES)

    def tag(purpose: str, s: dict) -> str:
        if kept is None:
            bank = purpose not in RIVER_PURPOSES and "x2" not in s and (s["x"], s["y"]) in keep_bank
            return advice(" (riverbank: keep for pump/sewage)", " (scarce riverbank lot)") if bank else ""
        k = kept_for(s)
        return "" if k is None or (k == "pump/sewage" and purpose in RIVER_PURPOSES) else f" (riverbank: kept for {k})"

    def site(purpose: str, s: dict) -> str:
        return (_where(s) + _extra(purpose, s) + tag(purpose, s)
                + (f" (leave {s['leave']})" if "leave" in s and not is_open() else ""))

    def price(purpose: str) -> str:
        pr = (prices or {}).get(purpose)
        return "" if pr is None else f"; {pr.text()}" + (f" - you have {money(math.floor(funds))}"
                                                         if funds is not None and short_of(funds, pr.cost) else "")

    for purpose, lst in lists.items():
        head = f"- {purpose}: "
        path = SITE_PURPOSES[role][purpose]
        if lst.short:
            locked = lst.short.startswith("locked")
            lines.append(head + (lst.short if locked else f"none - {lst.short}")
                         + (price(purpose) if lst.sites and not locked else ""))
            continue
        if purpose == "bus_stops":                   # one line through every stop, not a choice of sites
            stops = bus_stops(lst)
            lines.append(head + f"{len(stops)} stops for one line: "
                         + advice(bus_line_call(stops), ", ".join(f"({x},{y})" for x, y in stops)) + price(purpose))
            continue
        sites = lst.sites[:MAX_SITES]
        head += f"{len(sites)} site{'s' * (len(sites) != 1)}, "
        if purpose.startswith("zone_") and "x2" in sites[0]:     # one zone type per strip (zone_line)
            text, own = zone_line(purpose, sites, strips, site)
            cost = None
            if own is not None:
                shown[_where(own)] = purpose
                cost = strip_cost(purpose, own)                                      # planning: its strip's price
            lines.append(head + text + (price(purpose) if cost is None or role != "planning" or prices is None else
                                        f"; {money(cost)}" + (f" - you have {money(math.floor(funds))}"
                                                              if funds is not None and short_of(funds, cost) else "")))
            continue
        if path in paths:
            lines.append(head + f"same list as {paths[path]}" + price(purpose))
            continue
        paths[path] = purpose
        best, first = sites[0], shown.get(_where(sites[0]))
        if kept_for(best) is not None:                 # a kept bank lot: said on every line, never "same site as"
            own = next((s for s in sites if mine(purpose, s)), None)
            if mine(purpose, best) or own is None:
                lines.append(head + f"best {site(purpose, best)}" + price(purpose))
            else:
                if kept_for(own) is None:
                    shown.setdefault(_where(own), purpose)
                lines.append(head + f"best {site(purpose, own)} (not {_where(best)}: riverbank kept for "
                             f"{kept_for(best)})" + price(purpose))
            continue
        if first is None:
            shown[_where(best)] = purpose
            lines.append(head + f"best {site(purpose, best)}" + price(purpose))
            continue
        nxt = next((t for t in sites[1:] if _where(t) not in shown), None)
        if nxt is None:
            lines.append(head + f"same {'strip' if 'x2' in best else 'site'} as {first}" + price(purpose))
        else:
            shown[_where(nxt)] = purpose
            lines.append(head + f"best as {first}; next {site(purpose, nxt)}" + price(purpose))
    return lines


# ------------------------------------------------------------------ toolbox
# what core/actions.py appends to a failure's reason as advice (shared with the plain LLM agents): the open briefing
# style shows the reason alone (core_fact); executed and the history log keep the core message
CORE_ADVICE = ("; ask the mayor for funds", "; bulldoze it or route around")


def core_fact(message: str) -> str:
    """A core action message as the open style shows it: without CORE_ADVICE ("costs $7,800 but the police budget only
    has $6,632"); unchanged in the guided style."""
    if not is_open():
        return message
    for a in CORE_ADVICE:
        message = message.replace(a, "")
    return message


class Took(NamedTuple):
    """A successful action of this turn: its call, action and arguments, and the tiles it took (a build_road its
    whole line, road already or not; anything else the tiles it filled)."""
    call: int
    action: str
    args: dict
    tiles: frozenset


class ToolBox:
    def __init__(self, env, role: str, branch_dezone: bool = False, max_steps: int | None = None,
                 next_line: bool = False, style: str = "guided"):
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if style not in STYLES:
            raise ValueError(f"briefing style must be one of {', '.join(STYLES)}, got {style!r}")
        self.env, self.role = env, role
        self.style = style                                # the briefing style its results render in (fmt.py): open
                                                          # results state facts, never a call or a fix to make
        self.world = env.world
        self.branch_dezone = branch_dezone                # planning's branch road dezones its empty lots (_branch_clear)
        self.next_line = next_line                        # a successful action's result ends with the role's refreshed
                                                          # Next line (_next_refresh; ToolAgent only)
        self.max_steps = max_steps                        # the turn's tool-call limit (ToolAgent only): the mayor's
        self.reminded = False                             # end_turn reminder of open requests, once a turn (end_turn)
        self.held_ask = False                             # a department's money ask held back, once a turn
        self.fleet = FleetState.of(env)
        self.ended = False
        self.report = ""
        self.executed: list[dict] = []
        self.messages: list[dict] = []
        self.n_calls = 0                                  # tool calls this turn; "call N" counts from 1
        self.took: list[Took] = []                        # this turn's successful actions, oldest first
        self.looked: Counter = Counter()                  # look-ups run this turn by tool (repeats not counted)
        self._mapped: int | None = None                   # inspects run before map_overview first ran this turn
        self._seen: dict[tuple, tuple[int, dict]] = {}    # look-up key -> (call, result) since the last change
        self._failed: dict[tuple, tuple[int, str]] = {}   # action key -> (call, error) since the last success
        self._invalid: dict[tuple, tuple[int, str]] = {}  # bad call key -> (call, error) since the last success
        self._funded: dict | None = None                  # _funded_sites() since the last successful action
        self._depts: dict | None = None                   # dept_lists() since the last successful action (_bank_hint)
        self._refused: set[tuple] = set()                 # EARNED_CONFIRM calls refused this turn (_money_key), each
                                                          # until confirm: true overrides it once
        self.start_treasury = self.world.treasury         # at the turn's start: the loan guard (_loan_refusal)
        self.repaid = 0.0                                 # debt the mayor repaid this turn: the loan guard too
        self.borrowed = 0.0                               # debt the mayor took on this turn: the reserve guard
        self.tools: dict[str, Tool] = {}
        lookups = self._lookup_tools()
        self.lookups = {t.name for t in lookups}
        for t in lookups + self._turn_tools():
            self.tools[t.name] = t
        docs, extra = action_docs(), money_docs()
        for a in ROLES[role]["actions"]:
            if a != "noop":
                doc = (docs[a][0], docs[a][1] + extra.get(a, ""))
                self.tools[a] = _action_tool(a, doc, partial(self._act, a), confirm=guarded(role, a))
        from .region import region_tools                  # region mode: region_overview, send_message, deals
        for t in region_tools(self):
            self.tools[t.name] = t
            if t.name not in ALL_ACTIONS and t.name != "send_message":
                self.lookups.add(t.name)

    # ------------------------------------------------------------ interface
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: str | dict | None = None) -> dict:
        """Run one tool call (`_call`) in the toolbox's briefing style and report it to the game's live feed
        (bench/live.py), if it has one."""
        with briefing_style(self.style):
            res = self._call(name, arguments)
        live.call(self.env, self.role, name, arguments, res, name in ALL_ACTIONS)
        return res

    def _call(self, name: str, arguments: str | dict | None = None) -> dict:
        """Run one tool call. An identical repeat (same tool, same arguments after coercion) of a look-up returns
        the earlier result marked "repeat", and one of an action that failed is refused ("refused": "repeat")
        without running, as is one of any call with bad arguments or an unknown tool (same arguments as sent),
        until something changes: a successful action (or plan update) for look-ups, a successful action for failed
        actions and bad calls; a failed action only refreshes `history`, which lists it. A bulldoze that would cut
        roads off the highway or transport's demolish of their only station / harbour / airport link ("refused":
        "disconnect") is refused before it runs (not executed, logged or remembered as a failure) unless it passes
        confirm: true; so is the mayor's allocate_budget identical to one that succeeded this turn ("refused":
        "duplicate"), taking a department below what it keeps ("refused": "take_back") or topping up another building
        under the reserve ("refused": "reserve"), and its self-inflicted take_loan or one while only other buildings
        wait ("refused": "loan"), but there confirm: true counts only on a repeat of a call refused this turn
        (EARNED_CONFIRM; the result then says "override"). Once INSPECTS_FREE inspects
        have run this turn (repeats don't count), each further inspect result notes that map_overview shows the whole
        map (only the count when its radius note says so), until map_overview runs; after its first run, once
        INSPECTS_AFTER_MAP more have run, each further one notes that its map already shows the layout. The notes
        aren't cached, so a repeat comes without them."""
        self.n_calls += 1
        try:
            tool, args = self._resolve(name, arguments)
        except ToolError as e:
            bad = (name, self._sent_key(arguments))
            if bad in self._invalid:
                return self._refusal(*self._invalid[bad])
            self._invalid[bad] = (self.n_calls, str(e)[:REPEAT_ERROR_CHARS])
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in self._seen:
            n, res = self._seen[key]
            return {**copy.deepcopy(res), "repeat": f"same as call {n} (nothing changed since)"}
        if key in self._failed:
            return self._refusal(*self._failed[key])
        executed = len(self.executed)
        res = self._run(tool, args)
        if name in self.lookups:
            self._seen[key] = (self.n_calls, copy.deepcopy(res))     # callers may change the result they get
            self.looked[name] += 1
            if name == "map_overview" and self._mapped is None:
                self._mapped = self.looked["inspect"]
            elif name == "inspect":
                self._inspect_note(res)
        elif name in ALL_ACTIONS:
            if res.get("ok"):
                self._seen.clear()
                self._failed.clear()
                self._invalid.clear()
            elif "refused" not in res:               # a guard refusal ran nothing: an identical repeat is refused again
                self._failed[key] = (self.n_calls, str(res.get("error"))[:REPEAT_ERROR_CHARS])
                if len(self.executed) > executed:         # logged as a failure: history shows it
                    self._seen = {k: v for k, v in self._seen.items() if k[0] != "history"}
        elif name == "update_plan" and res.get("ok"):
            self._seen.clear()
        return res

    def _inspect_note(self, res: dict) -> None:
        """The note on an inspect result (see call): before map_overview, from the (INSPECTS_FREE + 1)th, that it
        shows the whole map; after it, from the (INSPECTS_AFTER_MAP + 1)th since, that its map already shows the
        layout. Either after the radius note when there is one (that one names map_overview)."""
        n = self.looked["inspect"]
        if self._mapped is None:
            if n <= INSPECTS_FREE:
                return
            count = f"inspect #{n} this turn"
            res["note"] = f"{res['note']}; {count}" if res.get("note") else \
                f"{count}: map_overview shows the whole map in one call"
            return
        if n - self._mapped <= INSPECTS_AFTER_MAP:
            return
        note = (f"inspect #{n - self._mapped} since map_overview: its map already shows the layout; inspect only a "
                "tile you are about to act on")
        res["note"] = f"{res['note']}; {note}" if res.get("note") else note

    @staticmethod
    def _refusal(n: int, err: str) -> dict:
        """The refusal of an identical repeat of the call that failed at call `n` with `err`."""
        if "call end_turn" not in err:
            err += "; change the arguments or do something else"
        elif not err.endswith("call end_turn"):
            err += "; call end_turn"
        return {"ok": False, "refused": "repeat", "error": f"this exact call failed at call {n}: {err}"}

    @staticmethod
    def _sent_key(arguments) -> str:
        """Arguments as sent, for the repeat check of a call that didn't resolve: JSON with sorted keys (none or
        blank as {}, but "null" stays null), else the stripped text."""
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            return "{}"
        try:
            return json.dumps(json.loads(arguments) if isinstance(arguments, str) else arguments, sort_keys=True,
                              default=str)
        except (TypeError, ValueError):
            return str(arguments).strip()

    def _resolve(self, name: str, arguments) -> tuple[Tool, dict]:
        """The tool and its checked, coerced arguments; ToolError for unknown tools and bad arguments."""
        tool = self.tools.get(name)
        if tool is None:
            if name == "update_plan":
                raise ToolError("update_plan is for the mayor only; send the mayor a message instead")
            if name == "noop":
                raise ToolError("there is no noop tool; to do nothing, call end_turn")
            if name in ALL_ACTIONS:
                raise ToolError(f"the {self.role} role cannot use '{name}'")
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.tools)}")
        return tool, self._check(tool, self._parse(arguments))

    def _run(self, tool: Tool, args: dict) -> dict:
        try:
            res = tool.fn(**args)
            res = _fit(_py(res), newest_last=tool.newest_last, hint=tool.trim_hint)
            if res.get("ok") is False and "error" not in res:
                res["error"] = res.get("message", "failed")
            return res
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:                           # never let a tool crash the agent loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _parse(arguments) -> dict:
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            return {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as e:
                raise ToolError(f"arguments are not valid JSON ({e}); send an object such as {{\"x\": 3, \"y\": 4}}")
        if not isinstance(arguments, dict):
            raise ToolError(f"arguments must be a JSON object of named parameters, got {type(arguments).__name__}")
        return arguments

    @staticmethod
    def _check(tool: Tool, args: dict) -> dict:
        out = {}
        for k, v in args.items():
            if v is None:                                # optional arguments sent as null count as absent
                continue
            if k not in tool.properties:
                if tool.strict:
                    valid = ", ".join(tool.properties) or "none"
                    raise ToolError(f"{tool.name} has no argument {k!r}; valid arguments: {valid}")
                out[k] = v
                continue
            if k in tool.lists and isinstance(v, list):
                v = ",".join(str(x) for x in v)
            out[k] = _coerce(k, v, tool.properties[k])
        missing = [k for k in tool.required if k not in out]
        if missing:
            raise ToolError(f"{tool.name} needs argument(s) {', '.join(missing)}")
        return out

    # ------------------------------------------------------------- actions
    def _act(self, name: str, /, **args) -> dict:
        """Run one action. Its guard first (guarded()): for the mayor's EARNED_CONFIRM actions it always runs, and a
        refusal is overridden only when confirm: true repeats a call (_money_key) refused this turn, which the
        override uses up; the result then carries "override": the refusal's kind. A refusal whose call sent confirm
        says it didn't count (EARNED_NOTE). The mayor's money actions (MONEY_ACTIONS) end with _money_note on
        success and on a core failure, the others with warnings or _explain's fix (the model only: executed and the
        history log keep the core message)."""
        w = self.world
        cap = self.env.cfg.max_actions_per_turn
        if sum(r["ok"] for r in self.executed) >= cap:
            raise ToolError(f"action limit reached for this turn ({cap} successful actions); call end_turn")
        t = args.pop("type", None)
        if t is not None and t != name:
            raise ToolError(f"the tool name is the action ({name}); don't pass type={t!r} - did you mean kind?")
        confirm, override = bool(args.pop("confirm", False)), None
        if self.role == "mayor" and name in EARNED_CONFIRM:
            kind, refusal = (self._allocation_refusal(args) if name == "allocate_budget"
                             else ("loan", self._loan_refusal(args)))
            if refusal:
                key = _money_key(name, args)
                hard = kind == "reserve" and self.borrowed > 0    # a loan this turn: no override (_reserve_refusal)
                if hard or not (confirm and key in self._refused):
                    self._refused.add(key)
                    if confirm and not hard:              # "... with confirm: true (confirm: true counts only ...)."
                        refusal = (refusal[:-1] + EARNED_NOTE + "." if refusal.endswith(".")
                                   else refusal + EARNED_NOTE)
                    return {"ok": False, "refused": kind, "error": refusal}
                self._refused.discard(key)
                override = kind
        elif not confirm and guarded(self.role, name):
            kind, refusal = (("bus_line", self._bus_line_refusal(args)) if name == "upgrade_road"
                             else ("disconnect", self._disconnect_refusal(name, args)))
            if refusal:
                return {"ok": False, "refused": kind, "error": refusal}
        cleared = self._branch_clear(args) if name == "build_road" and self.branch_dezone else ""
        bank, funded = (free_bank(w), self._funded_sites()) if name == "zone" else (None, None)
        ways = self._ways_out() if name == "zone" else None
        before, debt = w.struct.copy(), w.debt
        res = apply_action(w, self.role, {**args, "type": name}).to_dict()
        if res["ok"]:
            self._funded = self._depts = None
            self.took.append(Took(self.n_calls, name, dict(args), self._taken(name, args, before)))
            if name == "repay_loan":
                self.repaid += debt - w.debt
            elif name == "take_loan":
                self.borrowed += w.debt - debt
        self.executed.append(res)
        self.fleet.record(w.month, self.role, "action", action=res["action"], ok=res["ok"], message=res["message"],
                          cost=res["cost"])
        dept = ROLES[self.role]["budget"]
        if self.role == "mayor" and name in MONEY_ACTIONS:     # advice goes to the model only; the log keeps the
            try:                                               # core text; it never fails an action that ran
                note = self._money_note(name, args, res)
            except Exception:
                note = ""
        elif res["ok"]:
            note = self._bank_warning(bank) if bank is not None else ""
            note += self._funded_warning(funded, bank=bool(note)) if funded is not None else ""
            prior = note or (res["message"] if "WARNING" in res["message"] else "")   # the game's no-access warning
            note += self._branch_warning(ways, before, prior) if ways is not None else ""
            note += self._zone_left() if name == "zone" and self.role == "planning" else ""   # when nearly spent
        else:
            done, fix = self._done_note(name, args), self._explain(name, args, res["message"])
            note = done + (f"; {fix.removeprefix(' - ')}" if done and fix else fix)
        message = cleared + core_fact(res["message"]) + note
        if res["ok"] and self.next_line:
            message += self._next_refresh(len(message))
        out = {"ok": res["ok"], "message": message, "cost": res["cost"],
               "budget_left": round(w.dept_funds[dept] if dept else w.treasury, 2)}
        if override:
            out["override"] = override
        return out

    def _branch_clear(self, args: dict) -> str:
        """For planning's build_road that is a listed branch road (branch_sites: the same two ends) whose way is blocked
        only by empty zoned lots (no building): dezone them first within the same call (free; executed and the
        history log get it as its own action), when the road is affordable and the turn has room for both actions.
        Returns the text put before the road's message, "" when it did nothing. ToolAgent only (branch_dezone): the
        scripted fleet plays its own dezones. v19 (120 months, seeds 0-1): 10 of planning's 29 roads into zoned lots
        were the briefing's own branch road ("dezone (19,5)-(20,5) (empty lots, free), then build_road ...") sent
        without the dezone."""
        try:
            w = self.world
            if self.role != "planning":
                return ""
            line = _line(w, args)
            if not any(w.struct[y, x] in C.ZONES for x, y in line):
                return ""
            ends = {(args["x1"], args["y1"]), (args["x2"], args["y2"])}
            b = next((b for b in branch_sites(w) if b.clear and not b.built and {b.start, b.end} == ends), None)
            cap = self.env.cfg.max_actions_per_turn
            if b is None or sum(r["ok"] for r in self.executed) + 2 > cap:
                return ""
            cost = sum(C.CATALOG["bridge" if w.terrain[y, x] == C.WATER else "road"]["cost"]
                       for x, y in line if w.struct[y, x] not in (C.ROAD, C.HIGHWAY))
            if short_of(w.dept_funds["planning"], cost):
                return ""
            xs, ys = zip(*b.clear)
            rect = {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
            res = apply_action(w, self.role, {**rect, "type": "dezone"}).to_dict()
            self.executed.append(res)
            self.fleet.record(w.month, self.role, "action", action=res["action"], ok=res["ok"],
                              message=res["message"], cost=res["cost"])
            if not res["ok"]:
                return ""
            self._funded = None
            return f"{res['message']} at {_tiles_text(b.clear)} first (the branch road's empty lots, free); "
        except Exception:
            return ""

    def _taken(self, name: str, args: dict, before: np.ndarray) -> frozenset:
        """The tiles a successful action took (Took.tiles): a build_road's whole line, else the tiles it filled
        (`before`: the structures before it ran)."""
        w = self.world
        if name == "build_road":
            return frozenset(_line(w, args))
        ys, xs = np.nonzero((w.struct != before) & (w.struct != C.NONE))
        return frozenset(zip(xs.tolist(), ys.tolist()))

    def _disconnect_refusal(self, name: str, args: dict) -> str | None:
        """Why a bulldoze or demolish (guarded()) with these arguments must not run: it would cut highway-connected
        road tiles off the region (seed 1, v7: planning cleared the 2-tile link to the highway and the city emptied;
        a demolished train station, harbour or airport unlinks the roads it was the only link of). None when it
        wouldn't, or when the arguments are bad (the action reports those)."""
        w = self.world
        try:
            tiles = _rect(w, args) if name in ROAD_REMOVERS else [_point(w, args)]
        except Exception:
            return None
        if name in ROAD_REMOVERS:
            lost, gone = cut_off(w, tiles), [(x, y) for x, y in tiles if w.struct[y, x] == C.ROAD]
            why = advice("; build a new link first or pick tiles that aren't on the only route")
        else:
            (x, y), = tiles
            link = LINKS.get(int(w.struct[y, x]))
            if link is None:                              # not a link: nothing to relabel
                return None
            lost, gone = cut_off(w, (), drop={w._anchor_of(x, y)}), w.parts(x, y)
            why = f": the {link} is their only link" + advice("; ask planning to connect them to the highway by road "
                                                               "first, or build another link")
        if not lost.any():
            return None
        ys, xs = np.nonzero(lost)
        i = min(range(len(xs)), key=lambda i: min(abs(xs[i] - x) + abs(ys[i] - y) for x, y in gone))
        return (f"this would cut {len(xs)} road tile{'s' * (len(xs) > 1)} (e.g. around ({xs[i]},{ys[i]})) off the "
                f"{CUT_FROM[name]}{why}; to do it anyway repeat with confirm: true")

    def _bus_line_refusal(self, args: dict) -> str | None:
        """Why transport's upgrade_road (guarded()) must not run: the city has no bus line, its first one is ready
        (first_bus_line) and the upgrade, which transport's funds cover, would leave them short of it. The 120-month
        LLM runs: transit 0.00 (no bus line in 120 months) and 0.28 (a first line at month 72); seed 0, month 78:
        $2,000 went $690 on avenues, then create_bus_line failed at $1,310. The scripted transport builds the line
        first. None when it may run or the arguments are bad (the action reports those)."""
        w = self.world
        if w.bus_lines or w.metrics()["population"] < SITE_MIN_POP["bus_stops"]:
            return None
        cost = upgrade_cost(w, args)
        funds = w.dept_funds["transport"]
        if cost is None or short_of(funds, cost):
            return None
        lists = site_lists(w, "transport", role_obs(self.env, "transport"))
        bus = first_bus_line(w, lists)
        if bus is None or not short_of(funds - cost, bus.cost):
            return None
        return (f"this upgrade ({money(cost)}) would leave transport {money(math.floor(funds - cost))}, short of the "
                f"{money(bus.cost)} first bus line, and the city has none yet (transit is a fifth of your score): "
                + advice(f"build the line first, {bus_line_call(bus_stops(lists['bus_stops']))}, asking the mayor for "
                         "what you lack; ") + "to upgrade anyway repeat with confirm: true")

    def _allocations(self) -> list[tuple[str, float]]:
        """This turn's successful allocate_budget calls as (department, amount), oldest first (take-backs negative)."""
        return [a for r in self.executed if r["ok"] and r["action"].get("type") == "allocate_budget"
                and (a := _allocation(r["action"])) is not None]

    def _duplicate_refusal(self, args: dict) -> str | None:
        """Why an allocate_budget (guarded()) must not run: one with the same department and amount, as the handler
        reads them, succeeded this turn. Allocations add up and a success clears the repeat cache (v10 seed 0:
        utilities +$5,500 and health +$2,000 twice; v12 seed 0 month 3: a batch of three sent three times; the
        treasury drained and loans followed); so do take-backs (v19 seed 1, month 6: the mayor repeated its take-back
        pairs). None for another amount or department, a zero amount, a retry of one that failed, and bad arguments
        (the action reports those)."""
        key = _allocation(args)
        if key is None or not key[1] or key not in self._allocations():
            return None
        dept, amount = key
        now = money(self.world.dept_funds[dept])
        if amount < 0:
            return (f"you already took {money(-amount)} back from {dept} this turn (it now has {now}); take-backs add "
                    "up - repeat with confirm: true only to take it back again")
        return (f"you already moved {money(amount)} to {dept} this turn (it now has {now}); allocations add up - "
                "repeat with confirm: true only to add it again")

    def _allocation_refusal(self, args: dict) -> tuple[str, str | None]:
        """The guard an allocate_budget (guarded()) meets, as (refused kind, why or None): a take-back (a negative
        amount) the keep guard (_take_back_refusal) first, then the duplicate guard, then a top-up (a positive
        amount) the reserve guard (_reserve_refusal)."""
        key = _allocation(args)
        if key is not None and key[1] < 0 and (why := self._take_back_refusal(*key)):
            return "take_back", why
        if why := self._duplicate_refusal(args):
            return "duplicate", why
        if key is not None and key[1] > 0 and (why := self._reserve_refusal(*key)):
            return "reserve", why
        return "duplicate", None

    def _reserve_refusal(self, dept: str, amount: float) -> str | None:
        """Why a top-up (allocate_budget with a positive `amount`; guarded()) must not run: it buys another building
        (DeptMoney.kind "another", as the to-fund line lists it) or nothing now (kind "nothing" or "idle":
        RESERVE_REFUSED) and leaves the treasury at least $1 under the reserve (reserve()), which the playbook keeps for
        first buildings, supply and planning's road. It leads with what the top-up buys, then the treasury after it,
        the reserve and why, and the first/supply and growth top-ups still unfunded (Funding.to_fund), else what the
        reserve is for, and the idle money elsewhere to take back first. Smoke v20 seed 0, month 6: the to-fund line
        read "$4,305 covers planning, police; the other $6,500 would go under the reserve" and the mayor funded another
        fire station and another clinic anyway, leaving $303 of a $3,216 reserve; month 9: another police station
        took a $4,150 treasury to $1,650 (plan section 6 deferred this guard until v20 showed it). Review: $3,000 to
        transport with nothing ready emptied the reserve unrefused while $400 for another station was refused, and
        with nothing waiting the text read "first/supply and growth come first and none is waiting", no reason to
        hold back. None for first/supply, growth and blocked (the money waits for planning's lot), fire while
        flood_ahead (levees are first/supply, and the playbook's flood rule funds fire at once, as the scripted mayor
        does, though fire's needs line names a cheaper station and no levee site may be listed: fleet_scripted acting
        monthly met the guard only there, seeds 3 and 4, months 43, 65 and 66), a top-up that leaves the treasury at
        or above the reserve, a department not in the game, and one the action refuses (more than the treasury holds:
        its money note names the idle money). After a loan this turn (`borrowed`) the refusal says so and no confirm
        overrides it (_act): smoke v22 seed 1, month 9, the mayor borrowed $4,500 for a first high school, then
        confirmed another landfill down to $1,107 of a $3,000 reserve. The scripted mayor tops up only from the
        treasury above its own reserve, bar that flood rule."""
        w = self.world
        if dept not in DEPARTMENTS or dept not in self.env.roles or amount > w.treasury:
            return None
        after, keep = w.treasury - amount, reserve(w)
        if round(keep - after) <= 0:
            return None
        if dept == "fire" and flood_ahead(w):
            return None
        f = self._funding()
        m = f.depts.get(dept) if f else None
        if m is None or m.kind not in RESERVE_REFUSED:
            return None
        firsts = [f"{i.dept} {money(i.amount)} ({i.kind})" for i in f.to_fund if i.kind in NEED_KINDS[:2]]
        wait = (f"first/supply and growth come first: {_some(firsts, FUNDED_SHOWN)} still wait" if firsts else
                "the reserve keeps the city solvent and pays for first buildings, supply and planning's road when they "
                "come (none waits now)")
        if is_open():                       # the rule, no queue of what to fund first
            wait = ("the reserve is kept for solvency, first buildings, supply, levees and planning's road, which may "
                    f"use the treasury down to {money(RESERVE_MIN)}")
        idle = ", ".join(f"{d} {money(v)}" for d, v in f.idle.items() if d != dept)
        buys = (f"for {m.what}" if m.kind == KINDS[2] else
                f"buys nothing now ({dept}: {m.state or 'nothing ready to buy'}) and" if m.kind == KINDS[4] else
                f"buys nothing now ({dept} holds {money(m.idle)} idle) and")
        if is_open():                       # the guard's classification, not what the money would buy
            held = f" ({dept} holds {money(m.idle)} above what it keeps)" if m.kind == KINDS[5] else ""
            buys = f"to {dept}{held} is not for a first building, supply, a levee or planning's road and"
        return (f"this top-up {buys} leaves the treasury at {money(after)}, {money(keep - after)} under the "
                f"{money(keep)} reserve{reserve_why(w)}; {wait}"
                + (advice(f"; take back idle money first ({idle})", f"; idle money elsewhere: {idle}") if idle else "")
                + (f". You borrowed {money(self.borrowed)} this turn, and a loan pays only for first buildings, supply "
                   "and planning's road" + advice(": fund this from money above the reserve next turn") + " (confirm: "
                   "true doesn't count after a loan)." if self.borrowed > 0 else
                   ". To fund it anyway, repeat this exact call with confirm: true."))

    def _take_back_refusal(self, dept: str, amount: float) -> str | None:
        """Why a take-back (allocate_budget with a negative `amount`; guarded()) must not run: it leaves `dept` below
        what it keeps (DeptMoney.keep, and what set it: keep_why) of what it held before this turn's top-ups
        (DeptMoney.held leaves them out), so this turn's top-ups and the money above its keep may go back; it names
        what this turn already took back from it, and what the money left can't pay for (DeptMoney.cant_pay; never
        "it can't build next quarter": review, false whenever the money left still paid for its next build). Smoke
        v18 seed 0, month 6: the mayor read "planning: funds $2,508,
        target $3,000, spent $492; idle $1,508", took back all $2,508, and planning had $0 at months 9 and 12; seed 1,
        month 6: it took $1,182 of planning's $1,479 (nothing idle), then its last $297, to fund health. None for a
        department not in the game (it keeps nothing), a take-back that leaves it what it keeps, and one the action
        refuses (an unknown department, more than it holds)."""
        w = self.world
        if dept not in DEPARTMENTS or dept not in self.env.roles or short_of(w.dept_funds[dept], -amount):
            return None
        m = DeptMoney(self.env, dept)
        left = m.funds + amount
        if not short_of(left, m.held) or not short_of(left, m.keep):     # only top-ups, or its keep stays
            return None
        top = m.funds - m.held
        free = f"its idle money is {money(m.idle)}" if m.idle else "it has no idle money"
        if top > 0:
            free += f"{', plus' if m.idle else ', only'} the {money(top)} you moved to it this turn"
        todo = advice(f": take back {money(m.idle + top)}" if m.idle or top > 0 else ": leave its funds")
        took = -sum(a for d, a in self._allocations() if d == dept and a < 0)
        took = f"; you already took back {money(took)} from it this turn" if took > 0 else ""
        if is_open():                       # the keep as the guard's rule: no reason naming a build, no "can't pay"
            return (f"a take-back must leave {dept} the {money(m.keep)} it keeps; this one leaves {money(left)}{took}; "
                    f"{free}. To take it anyway, repeat this exact call with confirm: true.")
        cant = m.cant_pay(left)
        return (f"that takes {dept} below the {money(m.keep)} it keeps ({m.keep_why})"
                + (f", so it can't pay for {cant}" if cant else "")
                + f"{took}; {free}{todo}. To take it anyway, repeat this exact call with confirm: true.")

    def _loan_refusal(self, args: dict) -> str | None:
        """Why the mayor's take_loan (guarded()) must not run, as one text: the treasury holds RESERVE_MIN or more,
        or this turn's own moves took it under (_own_loan_refusal); and, whatever the treasury at the turn's start,
        no first building, supply or planning road waits (no first/supply or growth item in Funding.to_fund), so the
        loan would pay for another building (the ones waiting named) or sit in the treasury. Its interest and the
        debt it leaves are named. Smoke v20 seed 0, month 9: in a hold that started at $550 the mayor borrowed
        $5,000, $3,000 and $2,500, each with confirm: true on the first call (no guard fired: the turn started under
        $3,000), to fund another police, fire and health building (plan section 6 deferred this guard until v20
        showed it). While first/supply or growth waits (fire's item too while flood_ahead: levee money), or
        operations would take the treasury below $0 before the mayor's next turn (bills_ahead over act_every months),
        the second part refuses only the part of the loan at least LOAN_SLACK over what those, the bills and the
        reserve need, naming that amount. Review replay of v20 seed 0, month 9: $5,000 borrowed where $3,850 was
        needed; a flood-forecast levee loan and a loan that pays next quarter's deficit were refused as "would pay for
        another building". The second part stays silent with the treasury below $0, where a loan pays the bills (the
        briefing's WARNING line counts the months in the red). The scripted mayor borrows only at a turn's start with
        the treasury under RESERVE_MIN, whatever waits (never on seeds 0-4 at either cadence); its fleet repeats a
        refusal with confirm: true. None for a loan the action refuses (not positive, over the debt cap: its result
        says so) and bad arguments."""
        w = self.world
        amount = _money_key("take_loan", args)[2]
        if amount is None or amount <= 0 or w.debt + amount > C.MAX_DEBT:
            return None
        interest = money(amount * C.LOAN_INTEREST)
        anyway = " To borrow anyway, repeat this exact call with confirm: true."
        f = self._funding() if w.treasury >= 0 else None
        flood = flood_ahead(w)
        firsts = [i for i in f.to_fund if i.kind in NEED_KINDS[:2] or (flood and i.dept == "fire")] if f else []
        own = self._own_loan_refusal(amount, interest, short=sum(i.amount for i in firsts) > f.dip if f else False)
        if f is None:
            return own + anyway if own else None
        extras = [f"{i.dept} {money(i.amount)}" for i in f.to_fund if i not in firsts]
        buys = "pay for another building" if extras else "sit in the treasury"
        debt = f"(debt would be {money(w.debt + amount)} of {money(C.MAX_DEBT)})"
        bills = bills_ahead(w, max(1, self.env.cfg.act_every))
        if firsts or w.treasury < bills:                 # it pays for first/supply, growth or the bills: up to that
            fund, keep = sum(i.amount for i in firsts), reserve(w)
            gap = max(0.0, keep + bills - w.treasury)
            need = math.ceil(fund + gap)
            if amount - need < LOAN_SLACK:
                return own + anyway if own else None
            if is_open():                   # the guard's rule, no amount "needed" from the to-fund model
                over = (f"a loan may pay only for the first buildings, supply and planning's road that wait and for "
                        f"keeping the treasury at the {money(keep)} reserve through operations to your next turn, and "
                        f"this {money(amount)} loan is more than those add up to")
                if own:
                    return f"{own} Also, {over}." + anyway
                return f"{over}; it costs {interest}/month interest until repaid {debt}." + anyway
            parts = ([f"{money(fund)} for first/supply and growth"] * bool(firsts)
                     + [f"{money(math.ceil(gap))} to keep the treasury at the {money(keep)} reserve"
                        + (f" through operations of {signed(-bills, money)} to your next turn" if bills else "")] * bool(gap))
            over = (f"only {money(need)} of this {money(amount)} loan is needed ({', '.join(parts)}): the other "
                    f"{money(amount - need)} would {buys}")
            if own:
                return f"{own} Also, {over}." + anyway
            return (f"{over} at {money((amount - need) * C.LOAN_INTEREST)}/month interest until repaid {debt}"
                    + advice(f": borrow {money(need)} or less") + "." + anyway)
        waits = f"only another: {_some(extras, FUNDED_SHOWN)}" if extras else "nothing is ready to fund"
        none = f"no first building, supply or planning road is waiting ({waits})"
        if is_open():                       # the guard's rule, not what waits or what the loan would buy
            none = "no first building, supply or planning road is waiting, and a loan may pay only for those"
            if own:
                return f"{own} Also, {none}." + anyway
            return f"{none}; this {money(amount)} loan costs {interest}/month interest until repaid {debt}." + anyway
        if own:
            return f"{own} Also, {none}: the loan would {buys}." + anyway
        return (f"{none}; this {money(amount)} loan would {buys} at {interest}/month interest until repaid {debt}"
                + (advice(": fund another building only from the money above the reserve") if extras else "") + "."
                + anyway)

    def _own_loan_refusal(self, amount: float, interest: str, short: bool = False) -> str | None:
        """_loan_refusal's first part, a sentence without the confirm advice: (a) the treasury holds RESERVE_MIN or
        more, unless `short` (the first/supply and growth top-ups waiting come to more than Funding.dip, what they may
        use: smoke v22 seed 1, month 9, $3,607 and a $5,000 first high school waiting, the loan the refusal itself
        allows was refused and overridden), or (b) the turn started with that much and this turn's own moves took it
        under: top-ups (net of
        take-backs) that aren't for a first building or supply (DeptMoney.buys), named with their amounts to take
        back instead, or a repayment (`repaid`), which the loan would only borrow back. Smoke v13-v19: 10 of 15 loans
        ($34.8k of $60.0k) came in turns that started at $3,000 or more, 14 of 15 after the mayor's own top-ups had
        emptied the treasury, and none was repaid; the 120-month run sat at the debt cap from month 45. Review: (b)
        advised taking back "a top-up that isn't a first building or supply" when every top-up was one (a loan the
        playbook allows), and blamed a $100 top-up for what a $3,000 repayment drained. None otherwise."""
        w = self.world
        if w.treasury >= RESERVE_MIN:
            return None if short else (f"the treasury holds {cash(w.treasury)}, over the {money(RESERVE_MIN)} minimum: a "
                    f"{money(amount)} loan costs {interest}/month until repaid (debt would be "
                    f"{money(w.debt + amount)} of {money(C.MAX_DEBT)})" + advice("; borrow only for a first building "
                                                                              "or supply need the treasury can't cover")
                    + ".")
        net = sum(a for _, a in self._allocations())
        if self.start_treasury < RESERVE_MIN or (net <= 0 and self.repaid <= 0):
            return None
        by: dict[str, float] = {}
        for d, a in self._allocations():
            by[d] = by.get(d, 0.0) + a
        others = [f"{d} {money(a)}" for d, a in by.items() if a > 0 and self._buys(d) != KINDS[0]]
        if not others and self.repaid <= 0:          # every top-up for a first building or supply: the playbook's case
            return None
        causes = " and ".join([f"this turn's top-ups ({money(net)} net)"] * (net > 0)
                              + [f"repaying {money(self.repaid)}"] * (self.repaid > 0))
        loan = f"a {money(amount)} loan at {interest}/month"
        back = f"take back (free) a top-up that isn't a first building or supply ({', '.join(others)})"
        todo = (f"{loan} borrows back what you just repaid; {back} instead" if others and self.repaid else
                f"{back} instead of {loan}" if others else f"{loan} borrows back what you just repaid")
        if is_open():
            mine = f"top-ups not for a first building or supply: {', '.join(others)}"
            todo = (f"{loan} borrows back what you just repaid; {mine}" if others and self.repaid else
                    f"{mine}; {loan}" if others else f"{loan} borrows back what you just repaid")
        return (f"the treasury started this turn at {money(self.start_treasury)}; {causes} took it under "
                f"{money(RESERVE_MIN)}: {todo}.")

    def _funding(self) -> Funding | None:
        """funding_gaps for the reserve and loan guards, None if it fails: a guard never blocks an action by failing
        (like _money_note and _buys)."""
        try:
            return funding_gaps(self.env, self.fleet)
        except Exception:
            return None

    def _buys(self, dept: str) -> str:
        """What `dept`'s money is for (DeptMoney.buys), "" for a department not in the game (nobody spends it)."""
        if dept not in DEPARTMENTS or dept not in self.env.roles:
            return ""
        try:
            return DeptMoney(self.env, dept).buys
        except Exception:
            return ""

    def _bank_warning(self, before: np.ndarray) -> str:
        """A warning for a zone action that took the last free riverbank lots beside roads (`before`: free_bank()
        before it) while the city still lacks a sewage outlet / treatment plant or has neither a pump nor a water
        tower (pumps and outlets need the bank; a tower supplies water without it)."""
        w = self.world
        missing = _bank_needed(w)
        if not missing or not before.any() or free_bank(self.world).any():
            return ""
        ys, xs = np.nonzero(before & np.isin(w.struct, list(C.ZONES)))
        if not len(xs):                                   # not taken by this zoning: someone else's doing
            return ""
        return (" WARNING: that zoned the last free riverbank lot(s) beside roads; utilities still needs one for its "
                f"{' and '.join(missing)}" + advice(f" - dezone one near ({xs[0]},{ys[0]})",
                                                    f" (one was ({xs[0]},{ys[0]}))"))

    def _funded_sites(self) -> dict[tuple[str, str], tuple[Price, list[tuple[int, int]]]]:
        """funded_sites, kept until an action succeeds: a zone's warning, planning's find_sites zone_* and a failed
        zone's suggestions read it. Only departments holding at least their cheapest priced first building's price
        get an observation. {} on any error: it is advice, and a zone must never fail for it."""
        if self._funded is not None:
            return self._funded
        w, lists = self.world, {}
        try:
            for d in DEPARTMENTS:
                if d in self.env.roles and may_fund(w, d):
                    lists[d] = site_lists(w, d, role_obs(self.env, d))
            self._funded = funded_sites(w, lists)
        except Exception:
            self._funded = {}
        return self._funded

    def _funded_spots(self) -> list[Spot]:
        """funded_spots from _funded_sites(): each funded building's claimed site."""
        return list(claim_spots(self._funded_sites()).values())

    def _funded_warning(self, before: dict, bank: bool = False) -> str:
        """A warning for a zone action that took a claimed site (claim_spots of `before`, _funded_sites() before it)
        and so left a funded building no unclaimed site: a fresh claim afterwards seats fewer buildings than before.
        It names the buildings a claim keeping the earlier ones (keep) leaves out and what to dezone for them
        (_dezone_fix); pump and outlet left to the riverbank warning when it fired (`bank`). "" on any error. Smoke v14 seed 0, months
        3-6: planning zoned the lots near homes the mayor had just funded fire, health and education to build on; v15
        seed 1, month 6: four buildings named one lot, and with the rest zoned no warning came (each still had a
        site). Review: police on P or Q, fire on Q or R, a zone took P; the kept claims left fire on Q and warned for
        police, which a fresh claim seats on Q (fire on R)."""
        try:
            w = self.world
            was = claim_spots(before)
            taken = [sp.xy for sp in was.values() if any(w.struct[y, x] in C.ZONES for x, y in sp.tiles)]
            if not taken:                                 # no claimed site zoned: not this zoning's doing
                return ""
            after, lists = {}, {}
            for (d, p), (pr, _) in before.items():
                if d not in lists:
                    lists[d] = site_lists(w, d, role_obs(self.env, d))
                after[d, p] = (pr, [(s["x"], s["y"]) for s in lists[d][p].sites])
            if len(claim_spots(after)) >= len(was):       # a fresh claim still seats everyone
                return ""
            now = claim_spots(after, keep=was)
            hit = [(sp, (d, p)) for (d, p), sp in was.items() if (d, p) not in now and not (bank and p in BANK_TOP_UP)]
            if not hit:
                return ""
            names = []
            for sp, (d, p) in hit:
                own, near = f"{d}'" + "s" * (not d.endswith("s")), " near homes" * ((d, p) in NEAR_NEED)
                names.append((f"{own} {sp.building}{near}", d, money(math.floor(w.dept_funds[d]))))
            if len(names) == 1:                           # several: each with what it holds, kept short
                what = f"{names[0][0]} ({names[0][1]} holds {names[0][2]} for it)"
            else:
                items = [f"{b} ({held} held)" for b, _, held in names]
                what = ", ".join(items[:-1]) + " and " + items[-1]
            if is_open():
                return f" WARNING: that zoned the last site for {what}"
            return f" WARNING: that zoned the last site for {what}: " + self._dezone_fix(before, hit, taken)
        except Exception:
            return ""

    def _dezone_fix(self, before: dict, hit: list[tuple[Spot, tuple[str, str]]], taken: list) -> str:
        """What to dezone for the buildings a zone left without a site (_funded_warning's `hit`, in turn order; `taken`:
        the claimed sites it zoned). One building: "dezone one, e.g. (20,22)", its own claimed site if zoned, else
        another. Several: a tile each, the first of its claimed site and then the rest of its list before the zone
        (`before`) that the zone took and no earlier pick covers, "dezone (46,29) for the fire station and (46,35) for
        the landfill". Smoke v18 seed 1, month 9: "...fire's fire station near homes ($2,500 held) and health's
        landfill ($2,000 held): dezone one, e.g. (46,29)", and planning dezoned that one tile, which seats one
        building. An AWAY_FROM_HOMES building's tile must be away from homes once dezoned (_away_once_dezoned; one
        or several), else it needs a lot at the town edge (_edge_fix): v18's landfill sites sat in the new strip of
        homes."""
        w = self.world
        if len(hit) == 1 and hit[0][1][1] not in AWAY_FROM_HOMES:
            (x, y), = list(dict.fromkeys([sp.xy for sp, _ in hit if sp.xy in taken] + taken))[:1]
            return f"dezone one, e.g. ({x},{y})"
        picks, edge, used = [], [], set()
        for sp, (d, p) in hit:
            fits = lambda t: (not t & used and any(w.struct[y, x] in C.ZONES for x, y in t)
                              and (p not in AWAY_FROM_HOMES or _away_once_dezoned(w, t)))
            sites = [sp.xy] + [xy for xy in before[d, p][1] if xy != sp.xy]
            tiles = next((t for xy in sites if fits(t := sp._replace(xy=xy).tiles)), None)
            if tiles:
                picks.append((_tiles_text(tiles), sp.building))
                used |= tiles
            elif p in AWAY_FROM_HOMES:
                edge.append(sp.building)
        if len(hit) == 1:
            return f"dezone one, e.g. {picks[0][0]}" if picks else self._edge_fix("it", True)
        fix = "dezone " + _and([f"{t} for the {b}" for t, b in picks]) if picks else ""
        if edge:
            fix += ("; " if fix else "") + self._edge_fix("the " + _and(edge), len(edge) == 1)
        return fix or "dezone one, e.g. " + " or ".join(f"({x},{y})" for x, y in taken[:len(hit)])

    def _edge_fix(self, who: str, one: bool) -> str:
        """"<who> needs a lot at the town edge: leave one by (40,19), the far end of road_extension (25,19)-(40,19)"
        (edge_site of planning's road_extension list, as _away_line words it), else "... by the far end of a new
        road into open land"."""
        w = self.world
        try:
            site = edge_site(w, site_lists(w, "planning", role_obs(self.env, "planning"))["road_extension"].sites)
        except Exception:
            site = None
        where = (f"({site['x2']},{site['y2']}), the far end of road_extension {_where(site)}" if site else
                 "the far end of a new road into open land")
        return f"{who} need{'s' * one} a lot at the town edge: leave one by {where}"

    def _ways_out(self) -> list[tuple]:
        """ways_out() before a zone runs; [] (no warning) on any error: it is advice, and a zone must never fail for
        it."""
        try:
            return ways_out(self.world)
        except Exception:
            return []

    def _branch_warning(self, ways: list[tuple], before: np.ndarray, prior: str = "") -> str:
        """A warning for a zone that left no way for a new road into open land (ways_out() empty) where there was one
        before it (`ways`, best first; `before`: the structures), naming the tiles it zoned on the best such way to
        dezone. After another warning on the same zone (`prior`: the riverbank or funded-site one, or the game's "no
        road access yet") it adds a clause to it, not a second WARNING. "" on any error. Smoke v17 seed 0, months 0-3:
        planning zoned both sides of its only road nearly end to end (R at y=22 and y=23, I at y=25) with no dead end
        left to extend; a wind turbine, a water tower and a police station took the last 3 free lots beside it, and
        planning spent months 6-9 on roads into buildings and past the map edge."""
        try:
            w = self.world
            if not ways or ways_out(w):
                return ""
            took = lambda way: [(x, y) for x, y in way if before[y, x] == C.NONE and w.struct[y, x] in C.ZONES]
            tiles = next((t for way in ways if (t := took(way))), [])
            where = _tiles_text(tiles) if tiles else ""
            if is_open():
                at = f" ({where})" if where else ""
                return (f"; it also zoned the last lot a new road could branch from{at}" if prior else
                        f" WARNING: that zoned the last lot a new road could branch from{at}")
            if prior:
                return "; it also zoned the last lot a new road could branch from" + (
                    f" (dezone {where} to leave one)" if where and where not in prior else "")
            return (" WARNING: that zoned the last lot a new road could branch from; leave one free"
                    + (f", e.g. dezone {where}" if where else ""))
        except Exception:
            return ""

    def _next_refresh(self, used: int) -> str:
        """For a successful action of planning or transport (ToolBox(next_line=True): the model's toolbox only): the
        role's Next line recomputed on the world as it is now (briefing.next_line, at most NEXT_REFRESH_ITEMS builds,
        the funds-short ask as the line words it), as " | next: ..." or " | next: nothing left: end_turn" (or that
        the turn's successful-action cap is reached: end_turn), cut to what
        RESULT_CHARS leaves after the `used` characters of the message; "" for other roles or on any error. The late
        smokes: planning did its briefing's Next-line calls first, then sent them again word for word 6-15 calls
        later, reading the briefing's stale line (_done_note)."""
        if is_open():
            return ""
        try:
            from .briefing import next_clause, next_line      # briefing imports this module
            cap = self.env.cfg.max_actions_per_turn
            if self.role in ("planning", "transport") and sum(r["ok"] for r in self.executed) >= cap:
                return f" | next: action limit reached ({cap} successful actions this turn): end_turn"
            clause = next_clause(next_line(self.env, self.role))
            room = RESULT_CHARS - used
            if len(clause) <= room:
                return clause
            return "" if room < len(" | next: ") + 20 else " " + cut(clause, room - 1)
        except Exception:
            return ""

    def _done_note(self, name: str, args: dict) -> str:
        """For a failed action that repeats, word for word, one that succeeded earlier this turn: that it is done, and
        that the briefing (its Next line, Ready sites) predates it; "" otherwise. The late smokes: planning ran its Next
        line's build_park and zone (s0 month 102) or zone (s1 month 96) at its first calls, then, re-reading the
        briefing many calls later, sent them again word for word ("no empty land tiles there"), and once more
        (refused "repeat"). The calls were valid when the briefing showed them; nothing said they were done."""
        done = next((t for t in reversed(self.took) if t.action == name and t.args == args), None)
        if done is None:
            return ""
        if is_open():
            return (f" - this exact call already succeeded at call {done.call} this turn, so it is done (your briefing "
                    "was written before it)")
        return (f" - this exact call succeeded at call {done.call} this turn: it is done (your briefing's Next line "
                "and Ready sites were written before it); go on to the next item, or end_turn")

    def _explain(self, name: str, args: dict, message: str) -> str:
        """How to fix a failed build_road, zone or dezone, appended to the core message in the tool result (v8, seed 0
        month 0: planning zoned row 23, then ran four roads north through those lots, still empty, told only
        "occupied by residential" and never that dezoning is free; it also re-built a road twice and zoned high
        density before 'Town'). A blocked road names the empty zoned lots and the buildings on its line (a multi-tile
        building once), and dezones that stay on the line: per straight leg, one per run of lots between buildings,
        from its first to its last lot (dezone razes developed tiles, and a box off the line would take other lots).
        Smoke v9-v11: 11 x zone "no empty land tiles in that area", often on the road just built (build_road
        (35,23)-(35,8), then zone R x=35 y 9-22), and 13 x "every tile on that line is already road" (e.g. the main
        road (0,24)-(47,24) again), told nothing more. So a zone or dezone that found nothing to do says what the
        rectangle holds, a zone adds the free roadside land beside it, and a road on existing road names the road
        lines it runs along (else, off every line, the connected roads' extent). Smoke v12-v16: 14 of 25 such zones
        and 4 of 13 such roads went over the role's own zone or road of that turn, so those name its calls
        (_own_zones, _own_roads). A road blocked by a building, or off the map, while no dead end extends also names
        the best branch road (_branch_hint). A build on a taken tile says what took it and the next free site
        (_taken_hint). "" otherwise."""
        w = self.world
        if is_open():
            return self._explain_open(name, args, message)
        if name == "zone" and message == w.unlock_note("high_density"):
            return ' - zone with density "low" now'
        if name == "zone" and message.startswith("costs $") and " budget only has " in message:
            return zone_fit(w, args, w.dept_funds["planning"], mayor_ask(self.env, self.fleet, "planning").ok)
        if (name, message) in (("zone", "no empty land tiles in that area"), ("dezone", "no zoned tiles in that area")):
            try:
                tiles = _rect(w, args)
            except Exception:
                return ""
            beside = _zone_beside(w, tiles, [t for sp in self._funded_spots() for t in sp.tiles]) if name == "zone" \
                else []
            mine = self._own_zones(tiles) if name == "zone" else ""
            if mine and (strip := self._next_strip(args)):
                mine += f"; {strip}"
            return (f" - {_contents(w, tiles)}" + (f"; {mine}" if mine else "")
                    + (f"; zone beside it instead: {' or '.join(beside)}" if beside else "")
                    + (self._bank_hint() if name == "dezone" else ""))
        if name == "build_park" and message == "no empty land tiles there":
            return self._park_hint(args)
        if name != "build_road":
            if name == "create_bus_line":          # the suggested stops as one call (the LLM runs: lines of 1-2 stops)
                return self._stops_hint(message)
            m = NOT_EMPTY.match(message)
            return self._taken_hint(name, args, (int(m[1]), int(m[2])), m[3]) if m else ""
        if message == "every tile on that line is already road":
            line = _line(w, args)
            on = [_road_line_text(ln) for ln in _road_lines(w)       # lines it runs along (a crossing tile isn't)
                  if sum(_on_road_line(ln, x, y) for x, y in line) >= min(2, len(line))]
            own = self._own_roads(args)
            mine = f"you built {_some(own, OWN_NAMED, ' and ')} this turn" if own else ""
            if not on:
                return (f" - {mine + '; ' if mine else ''}{_roads_text(_roads(w))}; start on the road and run into free "
                        "land (find_sites road_extension)")
            many = len(on) > 1                                    # at most 2: one per leg, or a crossing tile's two
            return (f" - the road line{'s' * many} {' and '.join(on)} {'are' if many else 'is'} already built"
                    + (f" ({mine})" if mine else "")
                    + f"; branch off {'them' if many else 'it'} into free land (find_sites road_extension)")
        if " is occupied by " not in message:
            outside = message.endswith(f" is outside the {w.w}x{w.h} map")
            return f" - {hint}" if outside and (hint := self._branch_hint()) else ""
        line = _line(w, args)                             # build_road got this far, so the arguments parse
        kind = {}                                         # what build_road can't take, as it walks the line
        for x, y in line:
            if w.struct[y, x] not in (C.NONE, C.ROAD, C.HIGHWAY):
                kind[x, y] = "lot" if w.struct[y, x] in C.ZONES and w.level[y, x] == 0 else "built"
        lots = [p for p, k in kind.items() if k == "lot"]
        firsts = {}                                       # a building once, at its first tile on the line
        for p, k in kind.items():
            if k == "built":
                firsts.setdefault(w._anchor_of(*p), p)
        built = list(firsts.values())
        ls, bs = "s" * (len(lots) > 1), "s" * (len(built) > 1)
        tiles = lambda pts: _some([f"({x},{y})" for x, y in pts], BLOCKERS_NAMED)
        around = ""
        if built and self.role == "planning":
            from .planning import detour_hint                 # planning imports this module
            around = detour_hint(w, line)
        tail = f"; route around the building{bs}" + (around or " (the Map block has the road extent)")
        if built and (hint := self._branch_hint()):
            tail += f"; {hint}"
        if not lots:
            return f" - in the way: building{bs} {tiles(built)}{tail}"
        n = next((i for i, (_, y) in enumerate(line) if y != line[0][1]), len(line))    # the horizontal leg comes first
        rects = []
        for leg in (line[:n], line[n:]) if n > 1 else (line,):
            run = []
            for p in leg + [None]:                        # a run of lots ends at a building or the leg's end
                if p is None or kind.get(p) == "built":
                    if run:
                        (xa, ya), (xb, yb) = run[0], run[-1]
                        rects.append(f"dezone x1={min(xa, xb)},y1={min(ya, yb)},x2={max(xa, xb)},y2={max(ya, yb)}")
                    run = []
                elif kind.get(p) == "lot":
                    run.append(p)
        head = f" - in the way: empty zoned lot{ls} {tiles(lots)}"
        fix = f"dezone the lot{ls} first (free): {_some(rects, DEZONES_NAMED, '; ')}"
        if not built:
            return f"{head}; {fix}, then retry"
        return f"{head} and building{bs} {tiles(built)}; {fix}{tail}"

    def _explain_open(self, name: str, args: dict, message: str) -> str:
        """_explain in the open briefing style (fmt.py): what the failed action found (what a rectangle or tile holds,
        this turn's own zones, roads and parks there, what is in a road's way), never what to do instead; "" on any
        error or when the core message says it all (over budget, locked, off the map)."""
        w = self.world
        try:
            if (name, message) in (("zone", "no empty land tiles in that area"),
                                   ("dezone", "no zoned tiles in that area")) or \
                    (name == "build_park" and message == "no empty land tiles there"):
                tiles = _rect(w, args)
                mine = self._own_zones(tiles) if name == "zone" else ""
                if name == "build_park":
                    calls = [t.call for t in self.took if t.action == "build_park" and t.tiles & set(tiles)]
                    mine = f"your park at call {_and([str(c) for c in calls])}" if calls else ""
                return f" - {_contents(w, tiles)}" + (f"; {mine}" if mine else "")
            if name == "build_road" and message == "every tile on that line is already road":
                line = _line(w, args)
                on = [_road_line_text(ln) for ln in _road_lines(w)
                      if sum(_on_road_line(ln, x, y) for x, y in line) >= min(2, len(line))]
                own = self._own_roads(args)
                mine = f"you built {_some(own, OWN_NAMED, ' and ')} this turn" if own else ""
                if not on:
                    return f" - {mine + '; ' if mine else ''}{_roads_text(_roads(w))}"
                many = len(on) > 1
                return (f" - the road line{'s' * many} {' and '.join(on)} {'are' if many else 'is'} already built"
                        + (f" ({mine})" if mine else ""))
            if name == "build_road" and " is occupied by " in message:
                line = _line(w, args)
                lots = [(x, y) for x, y in line if w.struct[y, x] in C.ZONES and w.level[y, x] == 0]
                built = list({w._anchor_of(x, y): (x, y) for x, y in line
                              if w.struct[y, x] not in (C.NONE, C.ROAD, C.HIGHWAY) and (x, y) not in lots}.values())
                tiles = lambda pts: _some([f"({x},{y})" for x, y in pts], BLOCKERS_NAMED)
                parts = ([f"empty zoned lot{'s' * (len(lots) > 1)} {tiles(lots)}"] * bool(lots)
                         + [f"building{'s' * (len(built) > 1)} {tiles(built)}"] * bool(built))
                return f" - in the way: {' and '.join(parts)}" if parts else ""
            if name != "build_road" and (m := NOT_EMPTY.match(message)):
                hint = self._taken_hint(name, args, (int(m[1]), int(m[2])), m[3])
                return hint.split("; next free ")[0].split("; no free ")[0]
        except Exception:
            return ""
        return ""

    def _park_hint(self, args: dict) -> str:
        """For a build_park with no empty land in its rectangle: what the rectangle holds, this turn's own parks there
        ("your park at call 1"), and the best park block now (planning's park_site, sparing funded sites and scarce
        riverbank lots as the parks line does, sized to the funds) with its call; "" on any error. s0 month 102: a
        build_park on the park it had built at call 1 was told only "no empty land tiles there"."""
        try:
            from .planning import call as as_call, park_site     # planning imports this module
            w, tiles = self.world, _rect(self.world, args)
            mine = [t.call for t in self.took if t.action == "build_park" and t.tiles & set(tiles)]
            own = f"; your park{'s' * (len(mine) > 1)} at call{'s' * (len(mine) > 1)} {_and([str(c) for c in mine])}" \
                if mine else ""
            avoid = {t for sp in self._funded_spots() for t in sp.tiles} | scarce_bank(w)
            site = park_site(w, w.dept_funds["planning"], avoid)
            best = (f"; the best park block now: {as_call('build_park', site.args())} ({money(site.cost)})"
                    if site is not None else "")
            return f" - {_contents(w, tiles)}{own}{best}"
        except Exception:
            return ""

    def _bank_hint(self) -> str:
        """For planning's failed dezone: the riverbank lot utilities waits for, when it waits for one (planning's
        dezone_hint); "" otherwise or on any error. v23 seed 0: 22 dezones failed on road, river, empty land and a
        coal plant while planning hunted for the bank lot utilities asked for, in 36 of its 40 turns."""
        if self.role != "planning":
            return ""
        try:
            from .planning import dezone_hint                 # planning imports this module
            if self._depts is None:
                self._depts = dept_lists(self.env)
            return dezone_hint(self.world, self._depts)
        except Exception:
            return ""

    def _zone_left(self) -> str:
        """After planning's successful zone that leaves it under ZONE_LEFT_SHOWN: what its funds left pay for, at each
        zone price (model only). 131 of the 134 zones that failed on money in the 120-month runs came after the turn's
        earlier actions had spent it (57 with under $50 left)."""
        left = self.world.dept_funds["planning"]
        low, high = zone_price("low"), zone_price("high")
        if left >= ZONE_LEFT_SHOWN:
            return ""
        return (f" ({money(math.floor(left + 1e-9))} left: {int((left + 1e-9) // low)} more tiles at low density, "
                f"{int((left + 1e-9) // high)} at high density or for I and M)")

    def _stops_hint(self, message: str) -> str:
        """For a failed create_bus_line: the suggested stops as one call (bus_line_call), after asking the mayor for
        what transport lacks when the line's price was the failure; with buses that have room (bus_room), why another
        line buys nothing instead; "" when no stops are suggested, or on any error."""
        try:
            w = self.world
            lst = site_lists(w, "transport", role_obs(self.env, "transport"))["bus_stops"]
            if not lst.sites:
                return ""
            if lst.short:
                return f" - {lst.note}"
            call = bus_line_call(bus_stops(lst))
            if (m := BUDGET_SHORT.search(message)) and (lack := float(m[1].replace(",", "")) -
                                                        float(m[2].replace(",", ""))) > 0:
                if not mayor_ask(self.env, self.fleet, "transport").ok:   # the hold-back would refuse it: review
                    return f" - {money(math.ceil(lack))} short; say so in your end_turn report. The line: {call}"
                return f" - ask the mayor for {money(math.ceil(lack))} (send_message), then {call}"
            return f" - the suggested stops, all on one road network: {call}"
        except Exception:
            return ""

    def _branch_hint(self) -> str:
        """For a failed build_road while no dead end extends (observation's road_suggestions empty): the best branch
        road (branch_sites) and how to build it; "" when a dead end extends, there is no branch, or on any error.
        Smoke v17 seed 0, months 6-9: with the only road's sides all zoned or built, planning ran 6 roads into
        buildings ("occupied by coal_plant / commercial / residential ... route around") and 2 past the map edge."""
        try:
            w = self.world
            found = [] if _road_extensions(w) else branch_sites(w, k=1)
            return (f"no dead end to extend: branch instead, e.g. {found[0].steps(w)} (find_sites road_extension)"
                    if found else "")
        except Exception:
            return ""

    def _own_roads(self, args: dict) -> list[str]:
        """This turn's successful build_road calls along the failed line `args` names, each as the legs of it that
        are ("y=24 x 19-34 at call 2"; Leg.along: a road it only crosses isn't). Smoke v15 seed 0 month 3: planning
        built (19,43)-(28,43) at call 5 and again at call 10; v16 seed 0 month 0: (19,24)-(34,24), then the old road
        up to it, (0,24)-(19,24)."""
        w, failed, out = self.world, _legs(self.world, args), []
        for t in self.took:
            if t.action == "build_road":
                legs = [leg.text() for leg in _legs(w, t.args) if any(leg.along(f) for f in failed)]
                if legs:
                    out.append(f"{' and '.join(legs)} at call {t.call}")
        return out

    def _own_zones(self, tiles: list[tuple[int, int]]) -> str:
        """For a zone on a rectangle (`tiles`) with no empty land: this turn's zone calls that zoned tiles there still
        zoned (each tile to the latest), "" for none: "you zoned them at call 17 this turn" (every tile), "you zoned 2
        of them at call 1 and 2 at call 3 this turn". Smoke v14 seed 0 month 6: zone C (45,27)-(46,30) at call 17,
        zone I there at call 18, "all 8 tiles are zoned C"."""
        w = self.world
        left, calls = {(x, y) for x, y in tiles if w.struct[y, x] in C.ZONES}, []
        for t in reversed(self.took):
            if t.action == "zone" and (mine := left & t.tiles):
                calls.insert(0, (t.call, len(mine)))
                left -= mine
        if not calls:
            return ""
        if sum(n for _, n in calls) == len(tiles):
            return f"you zoned them at call{'s' * (len(calls) > 1)} {_and([str(c) for c, _ in calls])} this turn"
        each = [f"{n} at call {c}" for c, n in calls]
        return f"you zoned {_and([each[0].replace(' at', ' of them at', 1)] + each[1:])} this turn"

    def _next_strip(self, args: dict) -> str:
        """For a zone that found no empty land on this turn's own zoning (_own_zones): the next free strip on its type's
        list (site_lists; strips sparing funded sites first, as the Ready sites block ranks them), or why there is
        none; "" on any error. v19 (120 months, seeds 0-1): 19 zones re-zoned the exact rectangle planning had just
        zoned with another type (R (0,25)-(19,26), then I there)."""
        try:
            letter = str(args.get("zone", "")).upper()[:1]
            purpose = f"zone_{letter}"
            if purpose not in SITE_PURPOSES["planning"]:
                return ""
            lists = site_lists(self.world, "planning", role_obs(self.env, "planning"))
            lst = spare_funded({purpose: lists[purpose]}, self._funded_spots())[purpose]
            if lst.sites:
                s = lst.sites[0]
                return (f"{ZONE_RULE}: next free {purpose} strip x1={s['x1']},y1={s['y1']},x2={s['x2']},y2={s['y2']} "
                        f"({s['tiles']} tiles" + (f", leave {s['leave']}" if "leave" in s else "") + ")")
            return f"{ZONE_RULE}: no free {purpose} strip ({lst.short})" if lst.short else ""
        except Exception:
            return ""

    def _taken_hint(self, name: str, args: dict, xy: tuple[int, int], want: str) -> str:
        """For a build refused because its tile (`xy`, the first one of its footprint not free) isn't empty `want`
        land / river / riverbank: what the tile is or holds, a building as "now holds your water pump (call 5 this
        turn)" when this turn's own action put it there, "holds your water pump" when it is the role's, else "holds
        a water pump"; then the next free site on the role's list for what it builds (build_purpose), or why the
        list is empty. "" on any error: it is advice. Smoke v14 seed 1 month 0: utilities built its pump at (29,23)
        at call 5 and its outlet there at call 6, told only "(29,23) is not an empty land tile"; in v12-v16, 15 of
        the 19 builds on a taken tile went where the role's own build had gone that turn."""
        try:
            w, (x, y) = self.world, xy
            s, water = int(w.struct[y, x]), bool(w.terrain[y, x] == C.WATER)
            if s in C.ZONES:
                what = f"is zoned {C.LETTER_BY_ZONE[s]}" + (" (developed)" if w.level[y, x] > 0 else "")
            elif s in (C.ROAD, C.HIGHWAY, C.NONE):
                what = "is " + ({C.ROAD: "a bridge" if water else "road", C.HIGHWAY: "the highway"}.get(s) or (
                    "river" if water else "off the riverbank" if want == "riverbank" else "land"))
            else:
                spec = w.facility_spec(*w._anchor_of(x, y))
                b = building_name(spec["name"]) if spec else C.STRUCT_NAME.get(s, "building").replace("_", " ")
                call = next((t.call for t in reversed(self.took) if (x, y) in t.tiles), None)
                what = (f"now holds your {b} (call {call} this turn)" if call is not None else
                        "holds " + (f"your {b}" if s in OWNED_STRUCTS.get(self.role, ()) else a_or_an(b)))
            nxt = self._next_site(name, args, xy)
            return f" - ({x},{y}) {what}" + (f"; {nxt}" if nxt else "")
        except Exception:
            return ""

    def _next_site(self, name: str, args: dict, xy: tuple[int, int]) -> str:
        """The first free site on the role's find_sites list for what a build asks for (build_purpose), or why the list
        is empty; "" when it has no list or the list is locked."""
        purpose = build_purpose(self.role, name, args)
        if purpose is None:
            return ""
        lst = site_lists(self.world, self.role, role_obs(self.env, self.role))[purpose]
        site = next((s for s in lst.sites if (s["x"], s["y"]) != xy), None)
        if site is not None:
            return f"next free {purpose} site: {_where(site)}"
        return f"no free {purpose} site listed ({lst.short})" if lst.short and not lst.short.startswith("locked") else ""

    def _money_note(self, name: str, args: dict, res: dict) -> str:
        """Where the money stands after one of the mayor's MONEY_ACTIONS (`res`: apply_action's result as a dict,
        `args` without type and confirm), for the model's message only: `executed` and the history log keep the core
        text (the turn's tool trace shows the message the model read, cut at agent.TRACE_CHARS). Recomputes
        funding_gaps (about 5 ms).
        - a top-up: " | police: " + topped_up, then "; " + treasury_state + "; still to fund: " + still_to_fund;
        - a take-back: " | " + taken_back, then the treasury and what is still to fund;
        - take_loan: " | interest now $15/month (+$15); $47k more can be borrowed; treasury ...; still to fund ...";
          at the debt limit " | no more credit: repay_loan from money above the reserve cuts the $250/month interest",
          " | only $2,000 more can be borrowed (interest $240/month)" when less than it asked is left;
        - repay_loan: " | interest now $225/month (saves $25/month); treasury ..." (" | no debt left: no more
          interest; treasury ..." once repaid);
        - "treasury only has" (a top-up or repay_loan the treasury can't pay): " | treasury $500, $2,500 under the
          $3,000 reserve; idle to take back: planning $1,508; still to fund: ...";
        "" for any other failure or action, and on any error: a note never fails an action that ran. Smoke v13-v19:
        234 of 338 allocation messages had under 40 characters of content, so the step-0 plan was lost and block
        order became funding order; the treasury ran dry mid-turn in 10 of 56 turns, 8 of 25 failures drained by an
        earlier call of the same batch."""
        if name not in MONEY_ACTIONS:
            return ""
        try:
            w, ok, msg = self.world, bool(res.get("ok")), str(res.get("message", ""))
            rate = C.LOAN_INTEREST
            if name == "take_loan" and not ok and msg.startswith("debt limit"):
                room, interest = C.MAX_DEBT - w.debt, f"{money(w.debt * rate)}/month"
                if round(room) <= 0:
                    return advice(f" | no more credit: repay_loan from money above the reserve cuts the {interest} "
                                  "interest", f" | no more credit: debt at the {money(C.MAX_DEBT)} limit, {interest} "
                                              "interest")
                return f" | only {money(room)} more can be borrowed (interest {interest})"
            if not ok and "treasury only has" not in msg:
                return ""
            f = funding_gaps(self.env, self.fleet)
            where, still = treasury_state(w, f), advice(f"; still to fund: {still_to_fund(f)}")
            if not ok:
                idle = ", ".join(f"{d} {money(v)}" for d, v in f.idle.items()) or "none"
                return f" | {where}; " + advice("idle to take back", "idle") + f": {idle}{still}"
            if name == "take_loan":
                room = C.MAX_DEBT - w.debt
                credit = f"{money(room)} more can be borrowed" if round(room) > 0 else "no more can be borrowed"
                added = signed(float(args["amount"]) * rate, money)
                return f" | interest now {money(w.debt * rate)}/month ({added}); {credit}; {where}{still}"
            if name == "repay_loan":
                if w.debt <= 1e-9:                    # it repaid what was left, maybe less than it asked
                    return f" | no debt left: no more interest; {where}"
                saved = money(float(args["amount"]) * rate)
                return f" | interest now {money(w.debt * rate)}/month (saves {saved}/month); {where}"
            dept, amount = _allocation(args)
            m = f.depts.get(dept)
            if m is None:                             # a department not in the game: nobody spends it
                what = f"{dept} now {money(w.dept_funds[dept])} (not in this game)" if amount < 0 else \
                    f"{dept}: not in this game, so nothing spends it"
            elif is_open() and amount > 0:            # nothing on what it buys: the core message has its funds now
                return f" | {where}"
            else:
                what = taken_back(m) if amount < 0 else f"{dept}: {topped_up(m)}"
            return f" | {what}; {where}{still}"
        except Exception:
            return ""

    # ------------------------------------------------------------- look-ups
    def _lookup_tools(self) -> list[Tool]:
        if self.role == "mayor":                        # read-only: every department's lists, "<department>.<purpose>"
            purposes = [f"{d}.{p}" for d in DEPARTMENTS if d in self.env.roles for p in SITE_PURPOSES[d]]
            what = ("Check which sites a department has (read-only), e.g. before believing \"no sites\": the list "
                    "the department's own find_sites returns for <department>.<purpose>, up to "
                    f"{MAX_SITES} ranked sites (crowded spots last; sorted by distance to (near_x, near_y) when "
                    "given), or a note saying why there are none.")
        else:
            purposes = list(SITE_PURPOSES[self.role])
            what = (f"Up to {MAX_SITES} suggested sites for a purpose (ranked; crowded spots last), sorted by distance "
                    "to (near_x, near_y) when given.")
        tools = [
            Tool("city_overview", "Headline numbers: population, happiness, jobs, money, taxes, demand, utilities "
                 "supply vs demand, service coverage, scores and the change over the last 3 months.",
                 self.city_overview),
            Tool("map_overview", "The whole city as an ASCII map with rulers and a legend, plus where the "
                 "highway, river, roads and built-up area are. The Map block in your briefing usually suffices: call "
                 "this only when you need tile-level layout, at most once per turn; inspect for details.",
                 self.map_overview),
            Tool("inspect", f"Look at the map around (x, y) within `radius` tiles (0-{MAX_RADIUS}, default 2): every "
                 "non-empty tile with what is there (zone letter + level such as R3, a building's tier name, road, "
                 "river), plus power / water / pollution for zones; empty land is counted. x = column, y = row.",
                 self.inspect, {"x": {"type": "integer"}, "y": {"type": "integer"},
                                "radius": {"type": "integer", "description": f"0-{MAX_RADIUS}, default 2"}},
                 ("x", "y"), trim_hint="use a smaller radius"),
            Tool("my_facilities", "Your department's buildings (load, capacity, tier) and the tier upgrades on offer "
                 "for them (run one with upgrade_facility)." if self.role != "mayor" else
                 "Your levers: service funding levels, policies, service imports and department upkeep.",
                 self.my_facilities),
            Tool("unlocks", "Options still locked by population milestones, the population each needs, and the next "
                 "milestone.", self.unlocks),
            Tool("get_plan", "The mayor's city plan" + (" (goals, department budgets and priorities, notes)."
                 if self.role == "mayor" else ": the city goals and your department's budget target and priority."),
                 self.get_plan),
            Tool("get_messages", f"Council messages you can see from the last {self.env.cfg.council_window} months, "
                 "newest last.", self.get_messages, newest_last=True),
            Tool("history", "The fleet log, newest last: " + (
                 "actions of all roles ('action', or failed ones with 'failure'), reports" if self.role == "mayor"
                 else "your department's actions ('action', or failed ones with 'failure') and reports")
                 + ", plus city events and monthly metrics. Omit kind (or \"all\") for everything.", self.history,
                 {"kind": {"type": "string", "enum": [*HISTORY_KINDS, "all"]},
                  "months": {"type": "integer", "description": f"how many recent months (1-{MAX_MONTHS}, default 3)"}},
                 newest_last=True, trim_hint="ask for fewer months or one kind"),
        ]
        if purposes:
            tools.insert(3, Tool(
                "find_sites", what + " Rectangles and lines also carry x1, y1, x2, y2.",
                self.find_sites, {"purpose": {"type": "string", "enum": purposes},
                                  "near_x": {"type": "integer"}, "near_y": {"type": "integer"}}, ("purpose",)))
        return tools

    def city_overview(self) -> dict:
        return overview(self.env)

    def inspect(self, x: int, y: int, radius: int = 2) -> dict:
        w = self.world
        x, y = int(round(x)), int(round(y))
        if not w.in_bounds(x, y):
            raise ToolError(f"({x},{y}) is outside the {w.w}x{w.h} map (x 0-{w.w - 1}, y 0-{w.h - 1})")
        r = max(0, min(MAX_RADIUS, int(round(radius))))
        x0, x1, y0, y1 = max(0, x - r), min(w.w - 1, x + r), max(0, y - r), min(w.h - 1, y + r)
        buildings, zones, ground, seen, empty = [], [], [], set(), 0
        cls = {v: k for k, v in C.ROAD_CLASSES.items()}
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                st = int(w.struct[ty, tx])
                if st == C.NONE:
                    if w.terrain[ty, tx] == C.WATER:
                        ground.append({"x": tx, "y": ty, "what": "river"})
                    else:
                        empty += 1
                elif st in C.ZONES:
                    zones.append({"x": tx, "y": ty, "what": f"{C.LETTER_BY_ZONE[st]}{int(w.level[ty, tx])}",
                                  "powered": bool(w.powered[ty, tx]), "watered": bool(w.watered[ty, tx]),
                                  "pollution": round(float(w.pollution[ty, tx]), 2)})
                elif st == C.ROAD:
                    t = {"x": tx, "y": ty, "what": "bridge" if w.terrain[ty, tx] == C.WATER else "road"}
                    if int(w.road_class[ty, tx]) > 1:
                        t["class"] = cls.get(int(w.road_class[ty, tx]))
                    ground.append(t)
                else:
                    ax, ay = w._anchor_of(tx, ty)
                    if (ax, ay) in seen:
                        continue
                    seen.add((ax, ay))
                    spec = w.facility_spec(ax, ay)
                    b = {"x": ax, "y": ay, "what": spec["name"] if spec else C.STRUCT_NAME.get(st, str(st))}
                    if spec:
                        b["tier"] = spec["tier"]
                    fw, fh = w.footprint_at(ax, ay)
                    if fw * fh > 1:
                        b["size"] = f"{fw}x{fh}"
                    buildings.append(b)
        out = {"ok": True, "x": x, "y": y, "radius": r, "window": [x0, y0, x1, y1],
               "tiles": buildings + zones + ground, "empty_land": empty}
        if r != int(round(radius)):          # models ask for 10 or 20 to see more; say they got less
            out["note"] = f"radius {radius} clipped to {r} (0-{MAX_RADIUS}); map_overview shows the whole map"
        return out

    def map_overview(self) -> dict:
        return map_overview(self.world, parts=MAYOR_MAP if self.role == "mayor" else MAP_PARTS)

    def find_sites(self, purpose: str, near_x: int | None = None, near_y: int | None = None) -> dict:
        if (near_x is None) != (near_y is None):
            raise ToolError("give both near_x and near_y, or neither")
        w = self.world
        role, key = purpose.split(".", 1) if self.role == "mayor" else (self.role, purpose)
        lst = site_lists(w, role, role_obs(self.env, role))[key]
        if role == "planning" and key.startswith("zone_"):
            lst = spare_funded({key: lst}, self._funded_spots())[key]
        sites = lst.sites
        if key == "bus_stops" and sites:           # one line through all of them (near_x/near_y don't reorder it)
            stops = bus_stops(lst)
            out = {"ok": True, "purpose": purpose, "stops": stops,
                   "line": f"{len(stops)} stops for one line" + advice(f": {bus_line_call(stops)}")}
            if lst.note:
                out["note"] = lst.note if role == self.role else f"({role} department's note) {lst.note}"
            return out
        if near_x is not None:
            near = (int(near_x), int(near_y))
            sites = sorted(({**s, "distance": _distance(s, near)} for s in sites), key=lambda s: s["distance"])
        out = {"ok": True, "purpose": purpose, "sites": sites[:MAX_SITES]}
        if lst.note:                  # written to the department: the mayor mustn't read "ask planning" as its own task
            out["note"] = lst.note if role == self.role else f"({role} department's note) {lst.note}"
        return out

    def my_facilities(self) -> dict:
        w, role = self.world, self.role
        obs = role_obs(self.env, role, site_hints=False)
        if role == "mayor":
            return {"ok": True, **{k: obs[k] for k in ("funding_pct", "policies", "service_imports",
                                                       "department_upkeep")}}
        if role == "utilities":
            nets = [{**{k: v for k, v in n.items() if k != "plants"}, "plants": [_plant(p) for p in n["plants"]]}
                    for n in obs["networks"]]
            return {"ok": True, "networks": nets, "disconnected": obs["disconnected_facilities"],
                    "upgrades": obs["upgrades"]}
        if role == "transport":
            d = obs["transport"]
            return {"ok": True, **{k: d[k] for k in ("train_stations", "harbours", "metro_stations", "bus_lines",
                                                     "metro_lines", "freight")},
                    "airports": [{"x": x, "y": y} for x, y in w.facilities(C.AIRPORT)]}
        if role == "planning":
            marks = {k: [[x, y] for x, y in w.facilities(v["struct"])] for k, v in C.LANDMARKS.items()}
            return {"ok": True, "landmarks": {k: v for k, v in marks.items() if v},
                    "casinos": [[x, y] for x, y in w.facilities(C.CASINO)], "historic": obs["historic"]}
        facilities = {}
        for st in C.STRUCTS_OF_DEPT.get(role, []):
            lst = [dict(f) for f in w.service_info.get(C.STRUCT_NAME[st], [])]
            for f in lst:
                sp = None if "name" in f else w.facility_spec(f["x"], f["y"])
                if sp:
                    f.update(name=sp["name"], tier=sp["tier"])
            if lst:
                facilities[C.STRUCT_NAME[st]] = lst
        out = {"ok": True, "facilities": facilities, "upgrades": obs[role].get("upgrades", [])}
        if role == "fire":
            out["levee_tiles"] = obs["fire"]["levee_tiles"]
        if not facilities:
            out["note"] = "you have no buildings yet"
        return out

    def unlocks(self) -> dict:
        w = self.world
        locked: dict[int, list[str]] = {}
        for k in sorted(w.locked_now(), key=lambda k: (C.UNLOCK_POP[k], k)):
            locked.setdefault(C.UNLOCK_POP[k], []).append(k)
        return {"ok": True, "population": w.metrics()["population"],
                "locked_by_population": locked, "next_milestone": w.next_milestone()}

    def get_plan(self) -> dict:
        return plan_slice(self.fleet, self.role)

    def get_messages(self) -> dict:
        env = self.env
        msgs = [env._local(m) for m in env.council.visible_to(env._name(self.role),
                                                             since_month=self.world.month - env.cfg.council_window)]
        return {"ok": True, "messages": msgs[-MAX_ENTRIES:]}

    def history(self, kind: str | None = None, months: int = 3) -> dict:
        kind = None if kind == "all" else kind
        months = max(1, min(MAX_MONTHS, int(months)))
        roles = None if self.role == "mayor" else (self.role, "city")    # departments: their own and the city's
        entries = self.fleet.history(kind, months, role=roles, now=self.world.month)
        return {"ok": True, "months": months, "entries": entries[-MAX_ENTRIES:]}

    # --------------------------------------------------------- turn control
    def _turn_tools(self) -> list[Tool]:
        others = [r for r in self.env.roles if r != self.role]
        strings = {"type": "array", "items": {"type": "string"}}
        cap = f"at most {LIMITS['list_items']} items of {LIMITS['item_chars']} chars"
        tools = [
            Tool("send_message", "Post on the council board (delivered at the end of your turn): to a role, or "
                 "'all'. Several roles in one call (to: \"police,fire\") each get their own message.",
                 self.send_message, {"to": to_schema([*others, "all"]), "text": {"type": "string"}}, ("to", "text"),
                 lists=("to",)),
            Tool("update_memory", f"Save your notes for later turns. Each list you give replaces the old one ({cap}); "
                 "lists you leave out are kept.", self.update_memory,
                 {"commitments": strings, "open_tasks": strings, "lessons": strings}),
            Tool("end_turn", "Finish your turn with a short report for the mayor (what you did, what you need).",
                 self.end_turn, {"report": {"type": "string"}}, ("report",)),
        ]
        if self.role == "mayor":
            per_dept = lambda t: {"type": "object", "properties": {d: {"type": t} for d in DEPARTMENTS}}
            tools.insert(2, Tool(
                "update_plan", f"Write the city plan the departments follow. goals ({LIMITS['goals']} max) and notes "
                "replace the old ones; budgets (a $ target per department) and priorities (one line per "
                "department) merge per department, so send only the ones that change. To clear one, set its budget "
                "to 0 or its priority to \"\" (never null). Budgets are targets: move the money with allocate_budget.",
                self.update_plan, {"goals": strings, "budgets": per_dept("number"),
                                   "priorities": per_dept("string"), "notes": {"type": "string"}}))
        return tools

    def send_message(self, to: str, text: str) -> dict:
        """Queue one council message per recipient (`to`: one address, or several separated by commas: recipients()),
        posted at the end of the turn, so per-role digests, the Team tab's threads and the mayor's Open requests see
        one message each. A department's money ask to the mayor it can't get now is held back once a turn
        (_ask_refusal: the model only, in the guided style); sent to others too, only the mayor's copy is held back."""
        if not text.strip():
            raise ToolError("text is empty")
        from .region import message_targets
        names = recipients(to, message_targets(self.env, self.role))
        refusal = self._ask_refusal(text) if "mayor" in names else ""
        if refusal and names == ["mayor"]:
            return {"ok": False, "refused": "ask", "error": refusal}
        if refusal:                                   # the others still get it: review, "mayor,planning" lost both
            names = [r for r in names if r != "mayor"]
        self.messages += [{"to": r, "text": text} for r in names]
        each = " (one message each)" if len(names) > 1 else ""
        msg = f"message to {_and(names)}{each} will be posted at the end of your turn"
        return {"ok": True, "message": msg + (f"; not to the mayor: {refusal}" if refusal else "")}

    def _ask_refusal(self, text: str) -> str:
        """The model's department (max_steps given: ToolAgent) sending the mayor a money ask (collab.is_request and
        MONEY) it can't get now (mayor_ask: not first/supply, growth or another building the money covers): the
        refusal, once a turn (the next send_message posts it); "" otherwise. The scripted fleet never sees it, nor does
        the open briefing style (fmt.py): there every ask goes through and the mayor decides."""
        if (self.role == "mayor" or self.max_steps is None or self.held_ask or is_open()
                or not is_request("mayor", text) or not MONEY.search(text)):
            return ""
        a = mayor_ask(self.env, self.fleet, self.role)
        if a.ok:
            return ""
        self.held_ask = True
        return ask_refusal(a)

    def update_memory(self, **fields) -> dict:
        try:
            return {"ok": True, "message": self.fleet.update_memory(self.role, **fields)}
        except ValueError as e:
            raise ToolError(str(e))

    def update_plan(self, **fields) -> dict:
        """Save the plan. The result gives, needs-based whatever the targets (funding_gaps): the money to spend
        (`to_spend`: the treasury above the reserve plus the idle money, 0 in a hold), the top-ups the needs lines
        ask for, most urgent first (`to_fund`: department, kind, amount, upkeep_per_month), and the first/supply
        departments at a $0 target (`first_buildings`: first buildings and utilities' supply); then the idle money to
        take back (`idle`) and the departments below their budget target (`below_target`, with the treasury). `next`
        is the briefing's to spend and to fund lines (spend_lines), plus "first/supply at a $0 target: education" when
        there are any. Smoke v16-v19: the to-do line came from targets alone; education was at $0 in 15 of 15 plans
        and first buildings at a $0 target were funded 16 times in 36."""
        w = self.world
        try:
            out = {"ok": True, "message": self.fleet.update_plan(w.month, **fields)}
        except ValueError as e:
            raise ToolError(str(e))
        f = funding_gaps(self.env, self.fleet)
        if is_open():                       # the money as it stands, no to-fund list or next steps
            out.update(treasury=math.floor(w.treasury), reserve=round(f.reserve))
            if f.gaps:
                out["below_target"] = f.gaps
            if flags := nothing_to_fund(f, self.fleet.plan["budgets"], fields.get("priorities")):
                out["nothing_to_fund"] = flags
            return out
        if f.idle:
            out["idle"] = f.idle
        if f.gaps:
            out.update(below_target=f.gaps, treasury=math.floor(w.treasury))
        out["to_spend"] = max(0, round(f.spendable))
        out["to_fund"] = [{"department": i.dept, "kind": i.kind, "amount": i.amount,
                           "upkeep_per_month": round(i.upkeep)} for i in f.to_fund]
        nxt = [ln.removeprefix("- ") for ln in spend_lines(w, f)]
        if f.firsts_at_zero:
            out["first_buildings"] = list(f.firsts_at_zero)
            nxt.append(f"first/supply at a $0 target: {', '.join(f.firsts_at_zero)}")
        if flags := nothing_to_fund(f, self.fleet.plan["budgets"], fields.get("priorities")):
            out["nothing_to_fund"] = flags
            nxt.append("nothing to fund now: " + ", ".join(f"{d} ({wait_why(f.depts[d])})" for d in flags))
        out["next"] = "; ".join(nxt)
        return out

    def end_turn(self, report: str) -> dict:
        """End the turn with the report. The model's mayor and planning (max_steps given: ToolAgent) are first refused
        once a turn ("refused": "open_requests") while requests to them are still open after this turn's top-ups,
        actions and messages (open_requests), unless fewer than REMIND_MIN_LEFT calls would be left; the next end_turn
        ends it."""
        if self.role in ("mayor", "planning") and self.max_steps is not None and not self.reminded \
                and self.max_steps - self.n_calls >= REMIND_MIN_LEFT:
            self.reminded = True
            reqs = open_requests(self.env, self.fleet, self.messages, to=self.role)
            if reqs:
                return {"ok": False, "refused": "open_requests",
                        "error": (requests_reminder if self.role == "mayor" else planning_reminder)(reqs)}
        self.ended = True
        self.close(report)
        return {"ok": True, "message": "turn ended"}

    def close(self, report: str) -> None:
        """end_turn's record of the turn's report, without its refusals; `ended` is end_turn's to set. A turn cut at
        the token budget (fleet/budget.py) ends with this, so the mayor reads why the department said nothing."""
        report = report.strip()[:REPORT_MAX_CHARS]
        self.report = report
        self.fleet.reports[self.role] = report
        self.fleet.record(self.world.month, self.role, "report", text=report)

