"""Role definitions: who may do what, and which budget pays for it.

To add a role: add an entry here (and to TURN_ORDER), its department in
`sim/world.py:DEPARTMENTS`, its action handlers in `actions.py`, a KPI in
`bench/scoring.py`, and a baseline in `agents/scripted.py`.
"""
from __future__ import annotations

ROLES: dict[str, dict] = {
    "mayor": {
        "title": "Mayor (Town Hall)",
        "budget": None,  # the mayor controls the treasury itself
        "mission": (
            "Keep the city solvent and residents happy while funding growth. You own the "
            "city treasury, set the tax rates, and transfer money to the departments (planning, "
            "utilities, transport, police, fire, health, education), who can only spend what you give them. "
            "Upkeep of all roads, parks, plants and service buildings is paid automatically from "
            "the treasury every month, so keep a reserve. You also set service funding levels "
            "(50-150%) and enact policies. Your approval rating follows happiness and taxes; "
            "12 months below 30% and you lose the recall election (game over). 12 consecutive "
            "months with a negative treasury = bankruptcy."
        ),
        "actions": ["set_tax", "allocate_budget", "set_funding", "enact_policy", "repeal_policy",
                    "set_district_tax", "accept_offer", "decline_offer", "set_service_imports", "take_loan",
                    "repay_loan", "noop"],
    },
    "planning": {
        "title": "Planning & Housing Department",
        "budget": "planning",
        "mission": (
            "Grow the city. You lay roads, zone land Residential (R), Commercial (C), Industrial (I) "
            "or Offices (O) at low or high density, or mixed use (M: shops below flats), place parks and "
            "landmarks, list historic buildings, and bulldoze. Zones only develop if they "
            "are within 2 tiles of a road that connects to the highway (H), and each level up to 5 needs "
            "more services (see the level rules). Keep zones balanced with demand, keep homes away from "
            "industry and pollution, use the river and parks for land value, and zone industry on "
            "natural resources for specialised farms, sawmills, mines and oil fields."
        ),
        "actions": ["build_road", "zone", "dezone", "build_park", "build_landmark", "paint_district",
                    "specialize_district", "preserve_historic", "delist_historic", "build_casino", "bulldoze",
                    "demolish", "noop"],
    },
    "utilities": {
        "title": "Utilities Department (power, water & sewage)",
        "budget": "utilities",
        "mission": (
            "Keep every building powered, watered and connected to sewage at reasonable cost and "
            "pollution. You build power plants (coal, gas, solar, wind), water facilities (river pumps, "
            "water towers) and sewage facilities (outlets, treatment plants). A facility must touch a "
            "road (8-neighbourhood); it serves the road network it touches. Pumps and sewage "
            "facilities must sit on the riverbank. The river flows from north (y=0) to south: sewage "
            "and industrial runoff dirty the water downstream, so put pumps upstream of outlets. "
            "Power plants lose output as they age (overhaul them), wind turbines produce more where it "
            "is windy, and filters or expansions upgrade facilities. Shortfalls are covered by expensive "
            "imports from the region (the mayor can stop them); surplus power is sold. "
            "Aim for 10-60% spare capacity."
        ),
        "actions": ["build_power_plant", "build_water", "build_sewage", "upgrade_facility", "demolish", "noop"],
    },
    "transport": {
        "title": "Transport Department",
        "budget": "transport",
        "mission": (
            "Keep the city moving. Workers commute from home to the job centres along the shortest "
            "road route; roads carrying more commuters than their capacity get congested, which "
            "lengthens commutes, adds noise and exhaust and slows police, fire and ambulances. You "
            "upgrade streets to avenues (2.5x capacity) or highways (6x, but no zone access next to "
            "them), run bus lines between stops (buses take trips off the roads) and set the fare, "
            "and build train stations on the map edge and a harbour at the river mouth (links to the "
            "region that carry exports: industry needs freight capacity to sell its goods). Planning "
            "builds the streets."
        ),
        "actions": ["upgrade_road", "create_bus_line", "remove_bus_line", "set_fare",
                    "build_train_station", "build_harbour", "build_airport", "build_connector",
                    "build_metro_station", "create_metro_line",
                    "remove_metro_line", "upgrade_facility", "demolish", "noop"],
    },
    "police": {
        "title": "Police Department",
        "budget": "police",
        "mission": (
            "Keep crime low where people live and work. Crime grows with unemployment, dense "
            "buildings, low education and low land value, and falls where police stations cover "
            "the area. A station covers ~10 road steps and handles a limited number of people; "
            "arrests fill jails, and overcrowded jails make the police less effective. Crime "
            "lowers land value and happiness."
        ),
        "actions": ["build_police_station", "build_jail", "upgrade_facility", "demolish", "noop"],
    },
    "fire": {
        "title": "Fire & Emergency Department",
        "budget": "fire",
        "mission": (
            "Prevent and fight fires, and prepare for disasters. Every developed building has a fire risk "
            "(industry most, denser buildings more, education less). Fires in areas a fire station covers "
            "are put out; uncovered fires spread and destroy buildings. Each station covers ~14 road steps, "
            "a limited number of buildings and a few call-outs per month. River floods are forecast a few "
            "months ahead: levees on the riverbank protect the land behind them, shelters save lives."
        ),
        "actions": ["build_fire_station", "build_levee", "build_shelter", "upgrade_facility", "demolish", "noop"],
    },
    "health": {
        "title": "Health & Sanitation Department",
        "budget": "health",
        "mission": (
            "Keep residents healthy and the streets clean. Every building produces garbage: "
            "landfills (fill up over time) and incinerators (produce some power, pollute) collect it "
            "within their truck range; uncollected garbage piles up and hurts health and land value. "
            "Clinics and hospitals raise health; pollution, dirty water and garbage lower it. "
            "Unhealthy workers stay home sick. Deaths need cemetery capacity."
        ),
        "actions": ["build_landfill", "build_incinerator", "build_clinic", "build_hospital",
                    "build_cemetery", "clean_soil", "upgrade_facility", "demolish", "noop"],
    },
    "education": {
        "title": "Education Department",
        "budget": "education",
        "mission": (
            "Educate the population. Children need elementary school seats, teens need high school "
            "seats, and university seats turn high-school graduates into graduates. Only students "
            "with a seat nearby graduate. Educated workers fill skilled jobs, commit less crime and "
            "cause fewer fires. Libraries speed up learning nearby."
        ),
        "actions": ["build_school", "build_library", "upgrade_facility", "demolish", "noop"],
    },
}

TURN_ORDER = ["mayor", "planning", "utilities", "transport", "police", "fire", "health", "education"]
