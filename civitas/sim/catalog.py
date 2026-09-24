"""Static game data: tile types, building catalog, and tunable constants.

Everything numeric that shapes game balance lives here so it can be tuned
in one place (or overridden from config via `apply_overrides`).
"""
from __future__ import annotations

# --- terrain -------------------------------------------------------------
LAND = 0
WATER = 1

# --- structures (one per tile) ------------------------------------------
NONE = 0
ROAD = 1
RES = 2
COM = 3
IND = 4
PARK = 5
TOWNHALL = 6
COAL = 7
GAS = 8
SOLAR = 9
WIND = 10
PUMP = 11
TOWER = 12
HIGHWAY = 13  # outside connection: the city's link to the region
OFF = 14      # office zone
POLICE = 15
JAIL = 16
FIRE = 17
SHELTER = 18
CLINIC = 19
HOSPITAL = 20
LANDFILL = 21
INCINERATOR = 22
CEMETERY = 23
SCHOOL = 24
HIGHSCHOOL = 25
UNIVERSITY = 26
LIBRARY = 27
OUTLET = 28
TREATMENT = 29
DAM = 30
LEVEE = 31
TRAIN = 32
HARBOUR = 33
PLAZA = 34
STADIUM = 35
MONUMENT = 36
MIX = 37      # mixed-use zone: shops below, flats above
METRO = 38
CASINO = 39
AIRPORT = 40

ZONES = (RES, COM, IND, OFF, MIX)
ZONE_BY_LETTER = {"R": RES, "C": COM, "I": IND, "O": OFF, "M": MIX}
HOMES = (RES, MIX)            # zones people live in
SHOPS = (COM, MIX)            # zones with commercial jobs
LETTER_BY_ZONE = {v: k for k, v in ZONE_BY_LETTER.items()}

POWER_PLANTS = {"coal": COAL, "gas": GAS, "solar": SOLAR, "wind": WIND, "hydro": DAM}
WATER_PLANTS = {"pump": PUMP, "tower": TOWER}
SEWAGE_PLANTS = {"outlet": OUTLET, "treatment": TREATMENT}
UTILITY_TYPES = set(POWER_PLANTS.values()) | set(WATER_PLANTS.values()) | set(SEWAGE_PLANTS.values())

STRUCT_NAME = {
    NONE: "empty", ROAD: "road", RES: "residential", COM: "commercial",
    IND: "industrial", OFF: "office", PARK: "park", TOWNHALL: "town_hall", COAL: "coal_plant",
    GAS: "gas_plant", SOLAR: "solar_farm", WIND: "wind_turbine", DAM: "hydro_dam",
    PUMP: "water_pump", TOWER: "water_tower", HIGHWAY: "highway",
    OUTLET: "sewage_outlet", TREATMENT: "treatment_plant",
    POLICE: "police_station", JAIL: "jail", FIRE: "fire_station", SHELTER: "shelter",
    CLINIC: "clinic", HOSPITAL: "hospital", LANDFILL: "landfill", INCINERATOR: "incinerator",
    CEMETERY: "cemetery", SCHOOL: "elementary_school", HIGHSCHOOL: "high_school",
    UNIVERSITY: "university", LIBRARY: "library", LEVEE: "levee", TRAIN: "train_station",
    HARBOUR: "harbour", PLAZA: "plaza", STADIUM: "stadium", MONUMENT: "monument", MIX: "mixed_use",
    METRO: "metro_station", CASINO: "casino", AIRPORT: "airport",
}

# One character per tile for the ASCII map and for replay frames.
# lower-case zone letter = zoned but undeveloped; upper-case = developed.
CHAR = {
    NONE: ".", ROAD: "#", PARK: "P", TOWNHALL: "T", COAL: "E", GAS: "G",
    SOLAR: "S", WIND: "W", PUMP: "U", TOWER: "V", HIGHWAY: "H", DAM: "Y",
    OUTLET: "%", TREATMENT: "*", POLICE: "K", JAIL: "J", FIRE: "F", SHELTER: "^",
    CLINIC: "A", HOSPITAL: "h", LANDFILL: "L", INCINERATOR: "X", CEMETERY: "Z",
    SCHOOL: "N", HIGHSCHOOL: "B", UNIVERSITY: "Q", LIBRARY: "D", LEVEE: "|",
    TRAIN: "@", HARBOUR: "&", PLAZA: "+", STADIUM: "$", MONUMENT: "!", METRO: ":", CASINO: "?", AIRPORT: ">",
}
WATER_CHAR = "~"
BRIDGE_CHAR = "="
FRAME_FORMAT = 3        # v2: water tower 'O' -> 'V' ('O' = offices); v3: hospital 'M' -> 'h' ('M' = mixed use)

# buildings bigger than one tile: (width, height); x, y is the top-left tile
FOOTPRINT = {HOSPITAL: (2, 2), UNIVERSITY: (2, 2), STADIUM: (2, 2), CASINO: (2, 2), AIRPORT: (3, 3)}

# --- building catalog ----------------------------------------------------
# cost: one-off build cost, upkeep: per month (paid from the treasury,
# scaled by the owning department's funding level).
CATALOG = {
    "road":   {"cost": 10,   "upkeep": 1.0},
    "bridge": {"cost": 60,   "upkeep": 2.0},
    "zone":   {"cost": 5,    "upkeep": 0.0},
    "zone_high": {"cost": 8, "upkeep": 0.0},
    "park":   {"cost": 150,  "upkeep": 3.0},
    "bulldoze": {"cost": 5,  "upkeep": 0.0},
    "coal":   {"cost": 4000, "upkeep": 80, "power": 250, "pollution": 25.0},
    "gas":    {"cost": 6500, "upkeep": 60, "power": 200, "pollution": 10.0},
    "solar":  {"cost": 5000, "upkeep": 15, "power": 70,  "pollution": 0.0},
    "wind":   {"cost": 2500, "upkeep": 12, "power": 40,  "pollution": 0.0},
    "hydro":  {"cost": 9000, "upkeep": 45, "power": 300, "pollution": 0.0, "on_river": True},
    "pump":   {"cost": 1500, "upkeep": 25, "water": 250, "needs_river": True},
    "tower":  {"cost": 2200, "upkeep": 30, "water": 120},
    # sewage: outlets dump raw sewage into the river; treatment plants clean 80% of it
    "outlet":    {"cost": 1200, "upkeep": 15, "sewage": 250, "needs_river": True, "discharge": 1.0},
    "treatment": {"cost": 6000, "upkeep": 55, "sewage": 220, "needs_river": True, "discharge": 0.2},
    "townhall": {"cost": 0,  "upkeep": 20},
    "levee":  {"cost": 300,  "upkeep": 1.0},
}

COVERAGE_FULL = 0.6     # services work at full strength up to 60% of their radius

# City services: build with the owning role's budget. radius in road steps,
# capacity in people (or buildings for fire, garbage units for collection).
SERVICES = {
    "police_station":    {"struct": POLICE, "dept": "police", "cost": 2500, "upkeep": 55, "radius": 14,
                          "capacity": 4000, "holding": 25},
    "jail":              {"struct": JAIL, "dept": "police", "cost": 4000, "upkeep": 45, "capacity": 250},
    "fire_station":      {"struct": FIRE, "dept": "fire", "cost": 2500, "upkeep": 55, "radius": 14,
                          "capacity": 260, "responses": 5},
    "shelter":           {"struct": SHELTER, "dept": "fire", "cost": 1500, "upkeep": 10, "radius": 8,
                          "capacity": 2000},
    "clinic":            {"struct": CLINIC, "dept": "health", "cost": 2000, "upkeep": 40, "radius": 10,
                          "capacity": 2500},
    "hospital":          {"struct": HOSPITAL, "dept": "health", "cost": 8000, "upkeep": 120, "radius": 14,
                          "capacity": 9000},
    "landfill":          {"struct": LANDFILL, "dept": "health", "cost": 2000, "upkeep": 30, "radius": 14,
                          "collect": 450, "fill": 30000, "pollution": 4.0},
    "incinerator":       {"struct": INCINERATOR, "dept": "health", "cost": 5500, "upkeep": 60, "radius": 14,
                          "collect": 350, "power": 30, "pollution": 10.0},
    "cemetery":          {"struct": CEMETERY, "dept": "health", "cost": 1500, "upkeep": 12, "plots": 2500,
                          "rate": 30},
    "elementary_school": {"struct": SCHOOL, "dept": "education", "cost": 2500, "upkeep": 45, "radius": 14,
                          "seats": 450},
    "high_school":       {"struct": HIGHSCHOOL, "dept": "education", "cost": 5000, "upkeep": 75, "radius": 22,
                          "seats": 700},
    "university":        {"struct": UNIVERSITY, "dept": "education", "cost": 12000, "upkeep": 150, "radius": 60,
                          "seats": 1500},
    "library":           {"struct": LIBRARY, "dept": "education", "cost": 1500, "upkeep": 18, "radius": 8,
                          "capacity": 6000},
}
SERVICE_BY_STRUCT = {v["struct"]: k for k, v in SERVICES.items()}

_UPKEEP_KEY = {
    ROAD: "road", PARK: "park", TOWNHALL: "townhall", COAL: "coal", GAS: "gas",
    SOLAR: "solar", WIND: "wind", PUMP: "pump", TOWER: "tower", OUTLET: "outlet", TREATMENT: "treatment",
    DAM: "hydro", LEVEE: "levee",
}

# which department's funding level scales a structure's upkeep and effect
DEPT_OF_STRUCT = {ROAD: "planning", PARK: "planning",
                  **{s: "utilities" for s in (COAL, GAS, SOLAR, WIND, PUMP, TOWER, OUTLET, TREATMENT, DAM)},
                  LEVEE: "fire",
                  **{v["struct"]: v["dept"] for v in SERVICES.values()}, TRAIN: "transport", HARBOUR: "transport",
                  METRO: "transport", AIRPORT: "transport", CASINO: "planning"}


def upkeep_of(struct: int, on_water: bool = False) -> float:
    if struct == ROAD and on_water:
        return float(CATALOG["bridge"]["upkeep"])
    key = _UPKEEP_KEY.get(struct)
    if key:
        return float(CATALOG[key]["upkeep"])
    if struct == TRAIN:
        return float(TRAIN_STATION["upkeep"])
    if struct == HARBOUR:
        return float(HARBOUR_SPEC["upkeep"])
    if struct == METRO:
        return float(METRO_STATION["upkeep"])
    if struct == CASINO:
        return float(CASINO_SPEC["upkeep"])
    if struct == AIRPORT:
        return float(AIRPORT_SPEC["upkeep"])
    lm = LANDMARK_BY_STRUCT.get(struct)
    if lm:
        return float(LANDMARKS[lm]["upkeep"])
    svc = SERVICE_BY_STRUCT.get(struct)
    return float(SERVICES[svc]["upkeep"]) if svc else 0.0


KIND_BY_STRUCT = {v: k for k, v in {**POWER_PLANTS, **WATER_PLANTS, **SEWAGE_PLANTS}.items()}

# --- zone development ----------------------------------------------------
MAX_LEVEL = 5
RES_CAPACITY = [0, 12, 40, 100, 160, 240]   # residents per tile by level
MIX_RES = [0, 8, 24, 60, 100, 150]          # mixed use: flats above ...
MIX_JOBS = [0, 4, 10, 24, 40, 60]           # ... shops below
MIX_DEMAND_BONUS = 0.05                     # walkable neighbourhoods: a little extra demand
COM_JOBS = [0, 6, 18, 40, 65, 100]         # jobs per tile by level
IND_JOBS = [0, 10, 25, 45, 60, 80]
OFF_JOBS = [0, 10, 28, 55, 90, 140]
POWER_PER_LEVEL = {RES: 1.0, COM: 2.0, IND: 3.0, OFF: 2.0, MIX: 1.5}
WATER_PER_LEVEL = {RES: 1.0, COM: 1.0, IND: 2.0, OFF: 1.0, MIX: 1.2}
TOWNHALL_POWER = 5.0
TOWNHALL_WATER = 5.0
IND_POLLUTION_PER_LEVEL = 3.0
# generic, agriculture, forestry, ore, oil, high-tech
IND_POLLUTION_FACTOR = [1.0, 0.3, 0.5, 1.3, 1.5, 0.25]
RESOURCE_DEPLETION = [0.0, 0.0003, 0.0006, 0.001, 0.001, 0.0]   # share of the deposit used per level per month
HIGHTECH_EDU = 0.38            # adults with high school+ needed before high-tech industry appears

ACCESS_RADIUS = 2       # zones need a connected road within this Chebyshev distance
LOW_DENSITY_MAX = 2     # low-density zones stop at level 2
LV_FOR_L3, LV_FOR_L4, LV_FOR_L5 = 0.40, 0.58, 0.75
LAND_VALUE_FOR_L3 = LV_FOR_L3

# --- wealth & households ---------------------------------------------------
WEALTH_LV = (0.50, 0.68)          # land value thresholds for middle / high income homes
WEALTH_TAX = [1.0, 0.85, 1.15, 1.5]   # taxable income multiplier by wealth class (index 0 = none)
WEALTH_INCOME = [1.0, 1.0, 1.5, 2.3]
WEALTH_RENT = [1.0, 1.0, 1.2, 1.6]
WAGE_BY_EDU = [7.0, 9.0, 13.0, 19.0]   # $/month per employed worker: none / elementary / high / university
UNEMPLOYMENT_SUPPORT = 2.5
RENT_BASE, RENT_PER_LV = 0.8, 2.6       # $/resident/month
HIGH_DENSITY_RENT = 0.85
LIVING_COST = 1.6
REF_DISPOSABLE = 1.6                    # disposable income per resident that means "normal" shopping
OFFICE_SHARE_OF_SKILLED = 0.5
# demand caps (SimCity 4): big cities need a regional link and parks to keep attracting people
CAP_POP_CONNECTION = 6000
CAP_POP_PARKS = 2000
RESIDENTS_PER_PARK = 400
CAP_LEVEL = 0.25

# --- roads & traffic -----------------------------------------------------
ROAD_CLASSES = {"street": 1, "avenue": 2, "highway": 3}
ROAD_CAPACITY = {1: 900.0, 2: 2200.0, 3: 5200.0}   # commuters per month through one tile
ROAD_UPKEEP = {1: 1.0, 2: 2.5, 3: 5.0}             # per tile per month (x2 on bridges)
ROAD_UPGRADE_COST = {2: 30, 3: 110}                # per tile, on top of the existing street (x3 on bridges)
ROAD_NOISE = {1: 1.0, 2: 1.6, 3: 2.5}
HUB_BLOCK = 12                # job hubs: the busiest road tile of each 12x12 block
MAX_HUBS = 8
REGIONAL_COMMUTE = 0.08       # trips to/from the region at each outside connection (share of city jobs)
TRAIN_REGIONAL_BOOST = 1.5
COMMUTE_DECAY = 15.0          # tiles: gravity model distance decay
BPR_ALPHA, BPR_BETA = 0.15, 4.0
COMMUTE_BASE_MIN, MIN_PER_TILE = 3.0, 0.8
COMMUTE_OK_MIN, COMMUTE_BAD_SPAN = 18.0, 40.0
CONGESTION_SERVICE_PENALTY = 0.35
TRAFFIC_POLLUTION = 0.0008    # air pollution source per commuter on a road tile
STOP_RADIUS = 3               # bus stops serve tiles within 3
TRANSIT_BASE = 0.35           # share of covered trips that ride the bus at a $1 fare
LINE_CAPACITY = 1500.0        # riders per month per bus line
BUS_LINE_UPKEEP, BUS_STOP_UPKEEP = 60.0, 8.0
BUS_LINE_COST = 1500
MAX_STOPS = 10
DEFAULT_FARE = 2.0
FARE_RANGE = (0.0, 5.0)
TRAIN_STATION = {"cost": 8000, "upkeep": 70}

# --- river ---------------------------------------------------------------
# The river flows north (y=0) to south. Pollution entering at a row is carried
# downstream and slowly dilutes.
RIVER_DECAY = 0.95            # share of river pollution left after one row downstream
RIVER_NORM = 400.0            # raw load that makes the river ~63% polluted
IND_RUNOFF = 0.5              # per industrial level, for industry within 2 tiles of the river
UNSEWERED_HEALTH = 0.12       # health penalty where sewage is not collected
LAND_VALUE_FOR_L3 = 0.55

# --- population & education ---------------------------------------------
BIRTH_RATE = 0.0015                 # per adult per month
AGE_RATE = {"child": 1 / 48, "teen": 1 / 36, "adult": 1 / 300}   # compressed game time
SENIOR_DEATH_RATE = 1 / 90
ADULT_DEATH_RATE = 0.0004
GRAD_RATE = {"elementary": 1 / 24, "high": 1 / 24, "university": 1 / 40}
STUDENT_AGE_SHARE = 0.15            # share of high-school-educated adults of university age
LIBRARY_BOOST = 0.2
MIGRANT_ADULT_EDU = [0.24, 0.42, 0.26, 0.08]   # none / elementary / high / university
LABOR_PARTICIPATION = 0.76
TEEN_PARTICIPATION = 0.1
SICK_SHARE = 0.3                    # share of workers off sick at health 0
# required education of jobs: none / elementary / high / university
JOB_EDU = {"C": [0.25, 0.45, 0.25, 0.05], "O": [0.0, 0.1, 0.5, 0.4], "G": [0.45, 0.45, 0.1, 0.0],
           "I": [0.45, 0.40, 0.13, 0.02], "I1": [0.6, 0.35, 0.05, 0.0], "I2": [0.5, 0.4, 0.1, 0.0],
           "I3": [0.4, 0.4, 0.18, 0.02], "I4": [0.35, 0.4, 0.2, 0.05], "I5": [0.0, 0.1, 0.5, 0.4]}

# --- safety, fire, health -------------------------------------------------
CRIME_BASE = 0.06
ARREST_RATE = 0.004
RELEASE_RATE = 0.05
JAIL_OVERCROWD_EFF = 0.7
FIRE_RISK = {RES: 0.3, COM: 0.45, IND: 0.8, OFF: 0.3, MIX: 0.4}
FIRE_CHANCE = 0.004                 # monthly ignition chance per tile at risk 1 with no fire cover
FIRE_SPREAD = 0.4
FIRE_MAX_SPREAD = 12
GARBAGE_PER_LEVEL = {RES: 1.0, COM: 1.5, IND: 2.5, OFF: 0.8, MIX: 1.2}
GARBAGE_NORM = 25.0                 # uncollected units on a tile that count as "piled up"
HEALTH_BASE = 0.72
HOSPITAL_MORGUE = 10                # deaths per month a hospital can process

# --- economy -------------------------------------------------------------
TAXABLE_PER_RESIDENT = 8.0    # $/month of taxable income per resident
TAXABLE_PER_JOB = 12.0        # $/month per filled commercial/industrial job
TAXABLE_PER_OFFICE_JOB = 16.0  # offices pay the commercial rate on more taxable income
SERVICE_COST_PER_RESIDENT = 0.3   # $/month: administration (services are built explicitly)
DEFAULT_TAX = 0.09
MAX_TAX = 0.20
LOAN_INTEREST = 0.005         # per month on outstanding debt
MAX_DEBT = 50000
BANKRUPTCY_MONTHS = 12        # consecutive months with a negative treasury

# --- funding, policies, approval (mayor) ---------------------------------
FUNDED_DEPARTMENTS = ("utilities", "transport", "police", "fire", "health", "education")
FUNDING_MIN, FUNDING_MAX = 0.5, 1.5       # 50% .. 150%; effect = min(1.5, pct)^0.7, upkeep x pct

# Policies: monthly cost = fixed + per_resident * population (+ per_ind_job * industrial jobs)
POLICIES = {
    "recycling":          {"fixed": 30, "per_resident": 0.06, "garbage": 0.7,
                           "desc": "30% less garbage"},
    "neighborhood_watch": {"fixed": 0, "per_resident": 0.04, "crime": 0.9,
                           "desc": "10% less crime"},
    "smoke_detectors":    {"fixed": 0, "per_resident": 0.03, "fire": 0.8,
                           "desc": "20% lower fire risk"},
    "education_boost":    {"fixed": 50, "per_resident": 0.08, "rate": 1.25,
                           "desc": "students graduate 25% faster"},
    "industry_filters":   {"fixed": 0, "per_ind_job": 0.12, "pollution": 0.7, "demand": -0.1,
                           "desc": "industry pollutes 30% less, industrial demand -0.1"},
    "high_rise_ban":      {"fixed": 0, "max_level": 3, "land_value": 0.03,
                           "desc": "no buildings above level 3; +3% land value (heritage)"},
    "free_transit":       {"fixed": 100, "ridership": 1.5,
                           "desc": "bus ridership +50%, no fare income"},
}
POLICY_EFFECTS = POLICIES

# happiness = weighted share of satisfied needs
NEED_WEIGHTS = {
    "power": 0.10, "water": 0.09, "sewage": 0.05, "garbage": 0.06, "safety": 0.08, "health": 0.08,
    "education": 0.05, "commute": 0.07, "leisure": 0.06, "noise": 0.05, "pollution": 0.09, "tax": 0.06,
    "affordability": 0.06, "jobs": 0.10,
}
HAPPINESS_STRETCH = (1.3, -0.35)   # happiness = a * (weighted needs) + b: unmet needs hurt more
RICH_NEED_WEIGHT = {"pollution": 1.6, "safety": 1.6, "noise": 1.5, "leisure": 1.3}   # wealthy households care more
APPROVAL_START = 0.6
APPROVAL_MIN_POP = 100          # approval only moves once people live in the city
APPROVAL_TAX_SHOCK = 0.02       # approval lost per percentage point of tax increase
RECALL_APPROVAL = 0.30
RECALL_MONTHS = 12
NOISE_SOURCES = {COAL: 3.0, GAS: 2.0, INCINERATOR: 2.0, LANDFILL: 1.0, JAIL: 0.5, FIRE: 0.5, AIRPORT: 4.0, CASINO: 1.5,
                 STADIUM: 1.5}


# --- events & disasters ----------------------------------------------------
EVENT_EFFECTS = {
    "heatwave": {"power_demand": 1.2, "water_demand": 1.3, "fire_risk": 1.5},
    "recession": {"industry_demand": 0.6, "commercial_demand": 0.8, "export_price": 0.6, "tourism": 0.7},
}
FLOOD_REACH = 2               # land tiles within 2 of the river flood
LEVEE_PROTECT = 2             # a levee tile protects the land within 2 tiles of it
FLOOD_CASUALTY_RATE = 0.02    # of displaced residents, without shelters
FLOOD_OFFLINE_MONTHS = 2
DISASTER_MORALE = 0.08
COMPLEX_POLLUTION = 2.5       # an accepted mega-factory pollutes this much more than normal industry

# --- districts -------------------------------------------------------------
MAX_DISTRICTS = 8
DISTRICT_TAX_RANGE = 5.0      # percentage points above / below the city rate

# --- milestones, unlocks, landmarks ---------------------------------------
MILESTONES = [
    {"name": "Town", "pop": 1000, "reward": 5000},
    {"name": "City", "pop": 5000, "reward": 10000},
    {"name": "Metropolis", "pop": 10000, "reward": 20000},
    {"name": "Megalopolis", "pop": 20000, "reward": 40000},
]
UNLOCK_POP = {
    "high_density": 1000, "office_zone": 1000, "avenue": 1000, "high_school": 1000, "library": 1000,
    "treatment_plant": 1000, "jail": 1000, "incinerator": 1000, "plaza": 1000, "districts": 1000,
    "university": 5000, "hospital": 5000, "highway": 5000, "train_station": 5000, "hydro_dam": 5000,
    "stadium": 5000, "harbour": 5000, "monument": 10000,
    "mixed_use": 1000, "historic_preservation": 1000, "casino": 5000, "metro": 10000, "airport": 10000,
}
LANDMARKS = {
    "plaza":    {"struct": PLAZA, "cost": 3000, "upkeep": 20, "radius": 5, "leisure": 0.5, "land_value": 0.12, "tourism": 60},
    "stadium":  {"struct": STADIUM, "cost": 15000, "upkeep": 150, "radius": 6, "leisure": 1.0, "land_value": 0.06, "tourism": 500},
    "monument": {"struct": MONUMENT, "cost": 30000, "upkeep": 80, "radius": 8, "leisure": 0.6, "land_value": 0.25, "tourism": 900},
}
LANDMARK_BY_STRUCT = {v["struct"]: k for k, v in LANDMARKS.items()}
TOURIST_SPEND = 1.2           # city income per tourist per month (tickets, hotel tax)


# --- trade, freight, businesses, utilities market ---------------------------
FREIGHT_CAPACITY = {"highway": 1500.0, "train": 2500.0, "harbour": 4000.0}   # goods units per month
GOODS_OUTPUT = [1.0, 0.8, 0.8, 1.0, 1.2, 1.5]   # per filled industrial job: generic..high-tech
GOODS_PER_SHOP_JOB = 1.5
RAW_PER_FACTORY_JOB = 0.5       # generic factories use raw materials (local or imported)
PRICE = {"goods": 1.0, "raw": 0.7}
EXPORT_FEE = 0.3                # city income per unit of exports (port / rail / road tolls)
EXPORT_JOBS_PER_SPARE = 0.15    # spare freight capacity creates industrial demand (jobs per unit)
EXPORT_DEMAND_CAP = 1500.0
FREIGHT_TRIPS = 0.25            # truck trips per unit of goods moved to/from the region
TRUCK_WEIGHT = 2.0              # a truck loads the road like two cars
BASE_MARGIN = 0.05              # business profit margin in a healthy economy
CLOSE_AFTER_MONTHS = 6
AGEING_START_YEARS, AGEING_RATE, AGEING_FLOOR = 5.0, 0.05, 0.6   # power plants lose 5%/year after 5 years
MODULES = {
    "filter":    {"cost_share": 0.4, "pollution": 0.5, "for": ("coal", "gas", "incinerator", "landfill")},
    "expansion": {"cost_share": 0.6, "capacity": 1.5, "for": "any"},
    "overhaul":  {"cost_share": 0.3, "for": ("coal", "gas", "solar", "wind", "hydro")},
}
IMPORT_PRICE = {"power": 1.5, "water": 0.8}   # $ per unit per month, rising with city size
IMPORT_PRICE_GROWTH = 0.01                    # +1% per 1,000 residents
IMPORT_SHARE_MAX = 0.25                       # imports cover at most 25% of demand
POWER_SELL_PRICE = 0.25                       # surplus power sold to the region, per unit
HARBOUR_SPEC = {"cost": 10000, "upkeep": 90}
HARBOUR_ROWS = 6                              # the harbour sits near the river mouth (south edge)


# --- historic preservation ----------------------------------------------------
HISTORIC_MIN_LEVEL = 2        # only established buildings can be listed
HISTORIC_LV_BONUS = 0.06      # land value within 3 tiles of listed buildings
HISTORIC_TOURISTS = 6         # tourists per listed tile per month
HISTORIC_MAX_TILES = 80

# --- soil pollution --------------------------------------------------------
# long-lived: builds up under dirty industry, landfills and polluting plants, fades 0.5% a month
SOIL_SOURCE = {"industry": [0.010, 0.002, 0.003, 0.020, 0.025, 0.001],   # per level, by industry type
               LANDFILL: 0.03, COAL: 0.015, GAS: 0.005, INCINERATOR: 0.01, OUTLET: 0.008}
SOIL_DECAY = 0.995
SOIL_SPREAD = 0.15            # share that leaks to the 8 neighbours each month
SOIL_CLEANUP_COST = 40        # $ per tile (health department)
SOIL_CLEANUP_SHARE = 0.6

# --- wind & seasons ---------------------------------------------------------
PREVAILING_WIND = (1.0, 0.25)     # blows mostly towards the east (dx, dy)
WIND_DRIFT = 2.0                  # how many tiles air pollution is pushed downwind
SEASONS = ("winter", "spring", "summer", "autumn")
SEASON_EFFECTS = {                # multiply the matching event factors
    "winter": {"power_demand": 1.2, "road_capacity": 0.85, "tourism": 0.7, "fire_risk": 0.9},
    "spring": {},
    "summer": {"power_demand": 1.1, "water_demand": 1.15, "tourism": 1.3, "fire_risk": 1.25},
    "autumn": {"tourism": 0.9},
}

# --- metro ------------------------------------------------------------------
METRO_STATION = {"cost": 6000, "upkeep": 60}
METRO_TRACK_COST = 80             # per tile of straight-line distance between stations
METRO_LINE_UPKEEP = 120
METRO_RADIUS = 5
METRO_BASE = 0.5                  # share of covered trips that take the metro
METRO_CAPACITY = 6000.0           # riders per line per month
METRO_MIN_PER_TILE = 0.35         # much faster than driving

# --- casinos, airports ------------------------------------------------------
CASINO_SPEC = {"cost": 20000, "upkeep": 100, "jobs": 120, "tourism": 700, "crime": 0.55, "crime_radius": 9,
          "tax_per_tourist": 2.5}
RAW_PER_TECH_JOB = 0.3          # high-tech factories need raw materials too (buy them from mining cities)
REGIONAL_ZERO = {"goods_in": 0.0, "goods_out": 0.0, "raw_in": 0.0, "raw_out": 0.0, "cash": 0.0,
                 "commuters_in": 0.0, "commuters_out": 0.0, "tourists": 0.0}
# links between cities (region mode): capacity per month for goods and for people (commuters, tourists)
CONNECTOR = {"highway": {"cost": 1500, "goods": 1500.0, "people": 800.0},
             "rail": {"cost": 8000, "goods": 2500.0, "people": 1500.0}}
FEEDER_MAX = 12                  # a connector builds up to 12 tiles of feeder road inland to the network
LINK_CAPACITY = {**{k: {"goods": v["goods"], "people": v["people"]} for k, v in CONNECTOR.items()},
                 "boat": {"goods": 4000.0, "people": 200.0}, "air": {"goods": 300.0, "people": 1500.0}}
INTERCITY_PRICE = {"goods": 1.15, "raw": 0.85}   # value per unit sold to a linked city (better than the outside region)
INTERCITY_CITY_SHARE = 0.4       # share of that value the selling city collects (sales taxes, fees)
COMMUTE_SHARE = 0.5              # of the unemployed / unfilled jobs that commuting can match per month
TOURIST_SHARE = 0.03             # residents of linked cities who visit each month (split by attraction)
AIRPORT_SPEC = {"cost": 25000, "upkeep": 250, "freight": 3000.0, "tourism_boost": 1.5, "noise": 4.0}


def apply_overrides(overrides: dict | None) -> None:
    """Override catalog entries / constants from config, e.g.
    {"CATALOG.coal.cost": 3000, "SERVICES.clinic.radius": 9, "TAXABLE_PER_RESIDENT": 10}."""
    if not overrides:
        return
    g = globals()
    for key, value in overrides.items():
        parts = key.split(".")
        if parts[0] in ("CATALOG", "SERVICES", "LANDMARKS") and len(parts) == 3:
            g[parts[0]][parts[1]][parts[2]] = value
        elif parts[0] in g:
            g[parts[0]] = value
        else:
            raise KeyError(f"unknown balance override: {key}")
