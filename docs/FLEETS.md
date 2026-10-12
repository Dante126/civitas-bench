# Agent fleets

The `fleet` agent kind: each department is a tool-calling model with a short briefing, look-up and action
tools, a city plan and memory between turns. This is the reference for briefings, guards, money, tools,
limits (incl. the token budget), playbooks, modes, warm-ups, regions, the scripted fleet and reports.
Dated smoke-run results live in [HISTORY.md](HISTORY.md).

Contents:

- [Turns and briefings](#turns-and-briefings)
- [Guards](#guards)
- [Next lines: transport and planning](#next-lines-transport-and-planning)
- [Money guards and money results](#money-guards-and-money-results)
- [Tools](#tools)
- [Memory and limits per turn](#memory-and-limits-per-turn)
- [Coaching with playbooks](#coaching-with-playbooks)
- [Modes and supervised sessions](#modes-and-supervised-sessions)
- [Late-game smokes and the yardstick](#late-game-smokes-and-the-yardstick)
- [Config](#config)
- [Regions](#regions)
- [Scripted fleet](#scripted-fleet)
- [Reports and the collaboration score](#reports-and-the-collaboration-score)

`kind = "fleet"` plays a role with a tool-calling agent (`civitas/fleet/`) instead of the
one-shot JSON `llm` agent. A fleet is built to measure how well a team of agents manages
tasks and money, and to let people coach it.

## Turns and briefings

- **One manager, seven departments.** The mayor manages. It writes the city plan (goals, a
  budget target and a one-line priority per department, notes) and moves the money. The
  seven departments carry the plan out, and each one sees only its own slice. Turn order
  doesn't change, so the mayor acts first each month.
- **A turn is a tool loop.** The agent starts from a short briefing and then calls tools.
  The mayor's briefing covers the city, the finances (below) and the departments, most
  urgent first. Each department's line gives its funds, its target (`funds $0 < target
  $2,500` when it still needs funding; `target $7,000 (unusable now)` when it has nothing to
  buy, so a top-up would sit idle: no `<` and not in `update_plan`'s `below_target`, which
  lists it, with any priority given it in that call, under `nothing_to_fund`; not planning,
  nor a department blocked for lack of land) and what it spent over the last 3 months. Then comes
  one note on its money: what its needs line buys (below), what is blocked, its idle money,
  or what its money does without a needs line: `funds cover its builds` (every priced
  purpose), `funds cover its next build`, `supply has room` (utilities sites are ready but
  no supply is short, nor will be 90% used within 3 months at its growth over the last 3),
  `only a library ready ($1,500)` (only supplementary sites), or
  `nothing ready to buy`. Last is its latest report, cut at 170 characters. Two lines close
  the block: the money there is to spend and what to fund with it.
  - **Finances.** The treasury's change since the trend month is split into what the city
    earned and what the mayor moved: `- treasury $7,922 (-$12.3k since month 3: operations
    +$977, top-ups -$13.3k), debt $0`. Loans and `other` (a milestone reward, an offer's
    bonus) show only when nonzero. Last month's costs follow: `- last month: income $597,
    expenses $469: buildings $312 (utilities $82, roads $75, 3 more $155), city services
    $157, interest $0`. City services are the rest of the expenses (town hall, policies),
    and imports (services bought from outside, a region's trade deficit and deal payments)
    show as `imports $173` before the interest when there are any, so the parts add up.
    With debt, a third line gives the interest in dollars and what can still be borrowed
    (`repay_loan`'s description and the playbook say what a repayment saves). In smoke
    v13–v19 the change mixed in the mayor's own transfers, so 30 of 81 turns showed a
    "loss" while operations ran a surplus. Interest never showed in dollars either, and
    there were 0 `repay_loan` calls in 96 turns.
  - **Needs line.** It names the department's next build, and says whether it is the
    first. That is the cheapest ready first building the city lacks (not a supplementary
    or per-tile site; a first jail only once inmates pass 80% of the cells), else the
    cheapest ready building. In the review of this change, a $30 avenue stretch hid
    transport's first $1,500 bus line on 9 mayor turns of seed 0 (months 42–69, no bus
    line, `funds cover its next build`), and another elementary school hid education's
    first high school and university:
    - `needs $2,500 for its first police station (5 sites, $55/month)` while the city has
      none;
    - `needs $2,500 for another police station (coverage 74%, $55/month)` once it has one.
      Clinics, landfills, schools and high schools show their coverage too, and bus lines
      the transit share.
    - **Keeping up with coverage.** When that coverage fell over the last quarters it reads
      `coverage falling: 94 -> 86 -> 81%` (3 months apart). Under 90% (`KEEP_UP_GOAL`)
      the line estimates what more buildings bring: the sim's own coverage computed with
      buildings placed at the department's ready sites, picked greedily where each adds
      the most (`tools.keep_up`, written nothing to the world), stopping at 90%, at 4, or
      when the next adds under 1 point. A department short by more than one building
      names them all when the money covers them (`keep_up_counts`: first what its own
      funds pay for, then one more at a time in turn order while the treasury above the
      reserve plus idle money still covers every other top-up; a hold keeps the
      one-building line): `needs $5,000 for 2 more police stations (coverage falling:
      75 -> 29%, ~75% with them, $110/month)`, the to-fund line `police $5,000 (2
      buildings)`, the department's ask and money notes the same count, and its Your
      budget line where and to build all of them this turn (`- keep up with growth: 2
      more police stations at (22,30) and (20,27) (...): build both this turn: your funds
      cover them ($5,000)`), naming the Ready line's best site when another adds more.
      With one building it still names the site that adds the most. When no ready site
      adds coverage, the line says where the uncovered homes are and how much is lost
      downtown, where tier 1 covers 60% at most (`ready sites add under 1%: the uncovered
      homes near (39,26) are out of their road reach; 7 points are lost downtown (level
      4-5), ...: upgrade_facility to tier 2`), and the department is told to ask planning
      for a lot or road there instead. Late smoke 20261007 seed 2 (the LLM fleet from
      month 96, act_every 3): population grew 30% in a year while fire coverage went 94 ->
      86 -> 81 -> 78% with $35–39k above the reserve. Replaying its tool calls showed
      why: at month 99 fire and police built at their Ready lines' best site by need,
      (37,23) and (36,23), which served one building (load 1 of 260) while (36,30) would
      have taken each from 86–87% to ~92%; by month 102 no ready site reached the homes
      without cover (another road), and 7 of the 19 missing points were downtown. The
      guards and the scripted fleet don't change: one more building is still "another",
      and what a department keeps for a take-back doesn't move (`fleet_scripted` stays
      68.8 ± 2.0 / 7,341; the estimates add about 0.9 s an episode).
    - The last figure is the building's monthly upkeep at the department's funding level
      (the tier's or off-tier spec's, read when shown, so balance overrides reach it; a bus
      line with the 2 stops it needs at least). Every building adds it for good. In the 120-month run, service upkeep was a median 34% of
      income, and at month 48 the city had 22 service buildings against the scripted
      fleet's 7.
    - A hospital counts as a clinic. For utilities, any building that makes the same
      supply counts, and a supply is short only while the city lacks it or uses at least
      90% of it. In smoke v14–v15 seed 1 the line named a water tower with water spare,
      and towers were built until water was 4.7–4.9× demand.
    - Utilities' line names one building per short supply, bought together, most urgent
      first (a lot develops at all only with power, to level 2 with water, to 3 with
      sewage): `needs $6,200 for its first coal plant and water tower ($110/month)`. For
      each supply it is the cheapest ready building whose output covers max(1.5 × demand,
      demand + 60 power or 50 water or sewage) less the city's own output (imports left
      out), else the one with the most output. The 60 and 50 are a young city's first
      quarter (smoke v9–v19: a median 62 power and 50 water from month 0 to 3), so a wind
      turbine's 40 (× its site's wind) falls short of a fresh map's 66 power and a coal
      plant's 250 doesn't. Only one-tile first buildings count while one is ready: a 2×2
      water works, advanced treatment or nuclear plant is named only when none is (past
      10,000 residents water wants 700 or more, which only the $12,000 water works
      covers, and a review state at month 90 named it over six ready $1,500 pumps). With
      one free riverbank lot the outlet, which can go nowhere else, gets it and water a
      tower. A short supply whose building has no site yet is named after the price:
      `needs $6,200 for its first coal plant and water tower ($110/month); sewage outlet
      waits for a road to the river`. One supply alone reads as before, with the share
      used: `another water tower (water 117% used, $30/month)`. In smoke v20 seed 1 the
      line named only the cheapest ($2,200 tower), the mayor gave exactly that, and power
      stayed at 0–18% until month 10.
    - On a tie each building at that price is named (`another clinic (coverage 21%,
      $40/month) or another landfill (coverage 83%, $30/month)`). A first one is named alone
      (`its first landfill (4 sites, $30/month)` once a clinic is built).
    - A per-tile site counts its best stretch whole, upkeep included: `needs $90 for 3
      avenue tiles (5 sites, $4/month)`.
    - Planning's is its best `road_extension` site (a dead end's extension or a branch
      road), priced as `build_road` charges it plus any bulldozing, plus zoning the land
      it opens at low density (`zone`'s $5 a tile): `needs $505 for a road into open land
      and zoning it (opens 71 tiles)`. A road is no building, so it is neither a first nor
      another, shows no upkeep, and never shows on the leave-free or blocked lines. In
      the live 120-month run on seed 0, at month 9, the mayor read planning as `idle
      $1,243`. It took that back and $1,000 more. Planning's $150 extension then failed
      twice on "the planning budget only has $0" while its line showed no need. In smoke
      v18 seed 0, at month 9, the line priced the road alone (`needs $50 for a road into
      open land (opens 15 tiles)`); the mayor moved exactly $50, which left nothing to zone
      the land the road opened.
  - **Most urgent first.** Each department's money has a kind, and the block lists the
    departments in this order, turn order within a kind:
    1. first/supply: a first building, utilities' supply (its needs line names a building
       only while the supply is short), or a levee;
    2. growth: planning's road and zoning;
    3. another: a second building;
    4. blocked, then nothing to fund, then idle money.

    The order is shown, not imposed. It replaces the old `- first buildings still unfunded`
    line and the plan's priorities, which `get_plan` still has. In smoke v16 seed 0 the
    treasury ran short. The mayor funded second police and fire stations and a second
    clinic first, and education never got its school (33.2; v17: 33.3 and 33.4). In smoke
    v16–v19 the mayor funded in block order in 32 of 54 turns. Education, last in turn
    order, was topped up last in 11 of them, and at v19 seed 0 month 9 planning's $315 road
    came tenth.
  - **Idle money.** A department keeps max(2 × its 3-month spending, its dearest ready
    first building, $1,000); for utilities only buildings the needs line could name
    count, those it names together with any short supply's building blocked for lack of
    land (a fresh map's coal plant, tower and outlet: $7,400), and for planning its road
    and zoning. What set it is named: `2 x its $658
    spending`, `for its first university`, `for another fire station`, or `the $1,000
    minimum`. Funds that are $1,000 or more above that,
    excluding this month's top-ups, show as `idle $3,400 (keeps $2,000)`. The to-spend
    line counts them as money to take back (a negative `allocate_budget`) before any loan
    or new top-up. A take-back that would leave a department below what it keeps is
    refused (see Guards).
  - **Blocked buildings.** A first building with no lot for lack of land or roads counts
    toward the keep, because the department still needs that money. That covers:
    - no free roadside land;
    - no free roadside lot near the homes that need it, or no free roadside land at all,
      for a need-ranked station, clinic or school when at least 15% of what it serves is
      under 30% coverage (a well-covered department with no land isn't blocked);
    - no roadside lot away from homes, for a landfill or jail;
    - no free 2x2 roadside block;
    - for a pump or sewage outlet, no road to the river or the bank taken, but only while
      the city lacks one or its demand is at least 90% of supply. The 90% test also
      applies to a water tower (on water) and a power plant or wind turbine (on power).

    When a department has no ready site, its line says `; blocked: a police station`, and
    one line under the block says `- blocked for lack of land: money buys nothing there
    until planning frees lots (ask planning)`.
    In smoke v13 seed 0, planning had zoned every roadside lot, and the mayor took the
    station, clinic and school money back from four departments as idle. A department
    whose funded first buildings all have no lot of their own (every site of theirs is
    claimed first by an earlier department, see planning's leave-free line) reads
    `; blocked: an elementary school` too; its ready site already keeps the money. In smoke
    v18 seed 1, education's only site was also police's, and education never built.
  - **Below target.** A department counts as below its plan target only if it isn't idle
    and would use a top-up: it spent more than half of the most it held over those 3
    months, or it never held its keep. In smoke v9–v12, planning spent about $1,000 a
    quarter and still read "funds $7,834 < target $9,000", so the mayor topped it up.
    The `<` marker still reads the plan target. The to-fund line doesn't.
  - **To spend and to fund.** Two lines close the block, and `update_plan`'s `next` is the
    same two lines:
    - `- to spend: $2,698 above the $3,711 reserve (3 x $1,237 expenses) + $1,508 idle to
      take back = $4,206; operations +$233/month; first/supply and growth may use $4,917
      (treasury + idle above $3,000)`. The reserve is $3,000 or 3 months of expenses (6
      after a deficit month), with its reason when above the minimum. The deficit test works
      on rounded dollars; v16 seed 1 doubled the reserve on a $0.70 deficit. There are no
      operations before the first month.
    - First buildings, supply and planning's road may take the treasury down to $3,000:
      they may use the treasury and idle money above $3,000, in a hold or not, and the line
      names that amount when it is more than the money to spend and the to-fund line lists
      one. Anything else only gets the money above the reserve. In review, a hold's
      allowance that stopped at the hold's edge cut what a first school could use from
      $1,605 to $1 when the treasury rose $2 past the reserve.
    - In a hold (nothing to spend once idle money counts) the line says how far under the
      reserve the treasury is and when operations close the gap (left out past the game's
      end). `- to spend: none (hold): the treasury is $1,723 under the $3,711 reserve (3 x
      $1,237 expenses); operations +$381/month close that in 5 months; first/supply and
      growth may use $900 (treasury + idle above $3,000)`, or `nothing above $3,000`. In the
      120-month run "hold" showed in 35 of 40 turns with no way out, and $108.7k was still
      allocated in them.
    - `- to fund: first/supply: health $2,000, education $2,500 ($4,500, upkeep
      +$85/month) | growth: planning $315 | another: police $2,500 (upkeep +$55/month):
      $4,917 covers health, education, planning; the other $2,500 would go under the
      reserve`. Each amount is what the department's needs line lacks, whatever its plan
      target. The kinds go in the block's order, and a kind with several departments gets
      their sum and upkeep. The cover clause is greedy: one that doesn't fit is skipped, and
      a cheaper one after it can still fit. First/supply and growth are covered from what
      they may use, another from what is left above the reserve; the rest "would go under
      $3,000" when only first/supply and growth are left. In a hold it covers first/supply
      and growth only (`$400 covers no first/supply or growth; their $540 would go under
      $3,000`). With no needs line asking it reads `- to fund: nothing ready`.
    - In smoke v13–v19, education's month-0 target was $0 in 15 of 15 plans. First
      buildings at a $0 target were funded 16 of 36 times, against 36 of 40 with a target,
      and utilities got $113k of its $135k top-ups beyond what its needs line asked.
  - **Open requests.** Under those two lines, `## Open requests to you` lists the
    departments' requests to the mayor that are still open: a council message to the mayor
    that asks for something, with no top-up to that department and no message back to it
    since, inside its window of 2 × `act_every` months. That is the collaboration score's own
    rule (`bench/collab.py:answer`, used by both). There is one line per department, the
    longest waiting first, at most 4 (then `+N more`), with repeats merged: `- police: m6 x2
    (waiting 3 months), asks $2,500; to fund $1,006 (first/supply), covered: "Requesting
    $2,500 to build the first police station..."`. Each line gives what the department's
    money needs: the top-up and whether the money covers it (`not covered (hold)` when it
    doesn't), or why it needs none (`no top-up needed (nothing ready to buy)`, `(holds $1,508
    idle)`, `(blocked for lack of land)`). It adds `last turn to answer` when the next turn
    falls outside the window, and quotes the latest request. The digest counts those
    messages on one line and quotes one fewer other message per department line, so the
    briefing stays within its bound. The system prompt asks the mayor to answer each request
    the turn it sees it: a top-up, or one line by `send_message` to that department saying
    why not or when. One `send_message` answers several departments: `to` takes several
    roles separated by commas (`"police,fire"`, or a JSON list), and each gets its own
    council message, so digests, the Team tab's threads and this block see one each. A model mayor's first `end_turn` with requests still open is refused
    once (`refused: "open_requests"`, naming them) while at least 2 tool calls are left, and
    the next `end_turn` ends the turn. The scripted fleet gets no reminder. In the 120-month
    trinity runs (v19, v23) the mayor answered 59–81% of the requests sent to it, against
    about 100% for every other department. In v23 seed 0, 102 money requests went
    unanswered. The mayor's digest showed 94 of them, and in every case it funded someone
    else in the window (first/supply in a hold for 66 of them). 80 of them asked for
    another building, and it replied to few. In the region smoke that added the block (both
    cities on trinity, 12 months) the mayor answered 93–94% of the requests, but most
    answers were a no (follow-through 25 / 29): of 46 requests to it, 9 were for a first
    building or supply, 9 for planning's road, 15 for another building the money above the
    reserve didn't cover and 13 came with nothing ready to buy (transport's "below 1,500
    residents" notes among them), and the south mayor spent all 20 steps of month 9 on
    refusals. Departments now ask only for what the mayor can fund (below).

    The plan comes last.

  A department's briefing covers its plan slice, its budget (with one line on asking the
  mayor for money, `fleet/tools.py:mayor_ask`: `- ask the mayor for $2,500 for your first
  police station (send_message, with a reason): ...` when its kind is first/supply or growth,
  for its needs line's top-up, or another building the mayor's money covers now
  (`fund_cover`); otherwise `- no money ask to the mayor now: another police station ($2,500
  short) waits: the mayor is in a hold ... This turn: build only what your funds pay for; say
  what you need in your end_turn report`, or why it needs none: nothing ready to buy, idle
  money, blocked for lack of land (ask planning), supply has room; a no-ask also says, when
  the plan target is above the funds, that the target is the mayor's ceiling, not money to
  ask for, when an ask would be valid (`Ask when a supply will be 90% used within 3 months
  at its growth (now water 60% used, 64% in 3 months)`, or when a site is ready) and what
  to do this turn instead (build the next building its funds cover, `no supply needs
  building yet`, ask planning for a lot), as the hold-back below does; when its funds buy none of its ready sites, a line names the cheapest, all of
  them on a tie, and tells it to ask the mayor for that top-up, or to report instead, and
  end the turn instead of calling build tools; a model department's `send_message` to the
  mayor that asks for money it can't get now is held back once a turn, `refused: "ask"`,
  and the next one is posted; the scripted fleet is never held), its
  key metrics, a "since your last turn" digest (failed actions, events, and the newest
  council messages addressed to it or to `all`, quoted with sender and month), a Ready
  sites block (one line per `find_sites` purpose: the count and best site with the price
  of the cheapest unlocked building the sites are for (a bus line for bus stops; per tile
  for levees, `$300/tile`, and for congested roads, `$30/tile` for an avenue, with
  stretches that are avenues already listed last), and `; $2,500 - you have $1,000` when
  the funds are short; or why there is none, e.g. locked, under 1,500 residents for bus
  stops, no connected road reaches the river, the riverbank beside the roads is taken, no
  free roadside lot near the homes that need it (around a named tile), no free 2x2
  roadside block, or no roadside lot away from homes for a landfill, cemetery or jail; a site on one of
  the last two free riverbank lots is marked to keep for a pump or sewage outlet) and its
  memory. Utilities' lines say which one a scarce lot is kept for, the one its needs
  line builds there: `- wind: 6 sites, best (34,23), wind 1.39 (not (33,23): riverbank
  kept for sewage)`, `- pump: 1 site, best (33,23), pollution 0 (riverbank: kept for
  sewage)`, `- sewage: 1 site, best (33,23) (riverbank: kept for sewage)`. While a supply
  is short or at least 90% used, now or within 3 months at its growth over the last 3
  (`tools.py:demand_ahead`, `AHEAD_MONTHS`; then `water 84% used, 92% in 3 months`; the
  growth leaves out season and event multipliers, metrics' `demand_factor`, so summer
  water or a heatwave isn't read as growth), its
  budget block adds each such supply in the words of
  the mayor's needs line and what its funds pay for, most urgent first, with the
  buildings its needs line names: `- supply needed now: power 25% covered (supply 1.80,
  demand 7, 1.80 imported), sewage 0% covered (supply 0, demand 7): your $4,400 pays for
  a coal plant at (0,23) ($4,000); a sewage outlet at (33,23) ($1,200) needs $800 more`
  (as `short now: power (111 of 103)` it called a supply with room short, under Key
  metrics' "coverage 100%"). When its funds pay for no ready site, the can't-pay line
  under it names that amount (`the supply above needs $800 more ... ask the mayor for
  $800`), and names no purpose whose supply has room. It only names: utilities decides. In
  smoke v20 seed 1, month 3, utilities held $4,400 with the coal plant ready, read "best
  (33,23) ... (riverbank: keep for pump/sewage)" for wind and "same site as wind" for the
  pump and the outlet, spent all 20 steps inspecting sewage sites and built nothing.
  Every purpose outside planning is priced, and so is planning's `road_extension`,
  with the road and the zoning shown apart
  (`opens 71 land tiles; $150 road + $355 zoning - you have $0`, "you have" when its funds
  are short of the two), though never on its budget line. A library (it only adds to
  schools), a shelter (it only matters in disasters), a cemetery (the landfill comes first) and a
  train station, harbour or metro station (bus lines come first) are priced on their
  lines, but the mayor's needs line never names one, and the budget line only when nothing
  else is ready (smoke v9's mayor funded fire $1,500 for "a shelter", and fire never
  reached the $2,500 station). Prices are what the build actions charge, and funds count
  as short exactly when the build action would refuse them. Both have a short Map block
  (size, highway entry, river columns by row band, the extent of the connected roads and
  their straight road lines, e.g. `y=24 x 0-47; x=19 y 19-29`, up to 8 with the longest
  first, any road tiles not connected to the highway, and the built-up area) and end with
  the next milestone. While high density is locked, planning's Ready sites block says so
  and tells it to zone at low density. When no dead end extends into open land, planning's
  `road_extension` line (and `find_sites road_extension`) offers branch roads instead:
  `- road_extension: none - no dead end; branch: dezone (23,25) (empty lot, free), then
  build_road (23,25)-(23,39), opens 75 land tiles; $150 road + $375 zoning`.
  - A branch leaves a road tile of the connected network and runs straight into open land
    (up to 15 tiles, bridges allowed). It starts from a free lot beside the road, or else
    through up to 2 empty zoned lots to dezone. Failing those, it bulldozes the cheapest
    buildings (`bulldoze (17,23) (a level-1 home, $5)`).
  - Branches are ranked by land opened (as the dead-end extensions count it), then by
    fewer tiles to clear, then by distance to the town hall. `find_sites` lists up to 6,
    each with its cost, the road tile it leaves and what to run first.
  - In smoke v17 seed 0 the only road crossed the map with every tile beside it zoned or
    built. Planning read "no straight extension from a dead end" and spent three turns on
    roads into buildings and past the map edge.

  Two more lines there are about the other departments' buildings:
  - **`- leave free (funded): fire station (20,22), clinic (21,25): zone around them`**
    covers each department whose funds already pay for its cheapest ready first
    building (all of them on a tie; utilities' only while that supply is at least 90% used). It gives each building a site, in turn order,
    that no earlier building has claimed. A building with every site claimed first gets a
    note after the list: `; education (elementary school): no lot of its own - leave or
    free one near (38,25)`, at its best site. Past 240 characters whole items are dropped
    (`(+2 more)`), the claimed sites before the notes.
    - Zone strips that cover a claimed site are listed after the others. When one still
      comes first, it's marked `(leave (20,22) for the fire station)`.
    - A failed zone's "zone beside it" suggestions skip these sites.
  - **`- blocked for lack of land: utilities (sewage outlet: a road to the river), police
    (police station: near homes): extend a road into open land, or free a lot, for
    them`** gives each department's cheapest blocked first building, in turn order, and
    what frees it. Past 240 characters the tags are dropped (`away from homes` first),
    then the building names.
    - A landfill or jail is tagged `(landfill: away from homes)`, and a line under it says
      where to leave its lot: `- away from homes (health's landfill): leave a lot free by
      (12,9), the far end of road_extension (12,23)-(12,9), with no homes within 4 tiles`.
      It names the first extension or branch whose far end has under 3 homes within 4
      tiles; without one it says to extend a road into open land at the town edge. In
      smoke v15 seed 1 the landfill read "no roadside lot away from homes" from month 6,
      planning never left one, and garbage coverage stayed 0 all run.

  In smoke v14 seed 0, planning zoned the lots near homes after the mayor had funded fire,
  health and education, and fire held $2,500 for nine months without a site.

  Actions run immediately and return `ok`, a message,
  the cost and the money left, so a failed action can be fixed in the same turn. Bad arguments, unknown tools and
  refused actions come back as `{"ok": false, "error": ...}` for the model to correct. A
  failed `build_road`, `zone` or `dezone` also says how to fix it: a road blocked by empty zoned
  lots names them and the `dezone` rectangles to run first (free; they stay on the road's
  line, at most 2 named), and one blocked by a building says to route around it.
  - While no dead end extends, a road blocked by a building or run off the map also names
    the best branch road (`no dead end to extend: branch instead, e.g. dezone (12,23)
    (empty lot, free), then build_road (12,23)-(12,9) (find_sites road_extension)`).
  - A model's `build_road` that is a listed branch road blocked only by empty zoned lots
    runs the free `dezone` first within the same call, when the road is affordable and the
    turn has room for both actions (the scripted fleet runs its own dezones). In the
    120-month runs (v19), 10 of planning's 29 roads into zoned lots were the briefing's own
    branch road sent without its dezone.
  - A zone the planning budget can't pay says what a tile costs (industry and mixed use
    always at the high-density price), how many tiles the money buys, and the part of the
    rectangle that fits (`zone x1=5,y1=10,x2=7,y2=11 (6 tiles, $30) now; then ask the mayor for $18 for the rest`). In v19, 43 zones
    failed on a few dollars, 41 with money for some tiles, and planning asked the mayor
    instead of zoning less. The first wording ("zone fewer tiles rather than waiting for funds", and "rather than asking" in the playbooks) cut planning's requests on the v23 120-month seed 0 from 37 to 18 and its spending from $13.3k to $7.8k, under full residential demand, so the hint now asks for the rest as well.
  - One zone type per strip. Each zone line in Ready sites shows a strip no earlier zone
    line shows (`next free (0,22)-(16,23), 34 tiles (one zone type per strip)`) instead of
    `same strip as zone_R`. A zone that fails on the role's own zoning this turn names the
    next free strip for its type. In v19, planning re-zoned the rectangle it had just zoned
    with another type 19 times.
  - "Already road" names the road line it runs along ("branch off it into free land") and
    points to `find_sites road_extension`. When the line runs along the role's own
    `build_road` of that turn, it names that call (`you built y=24 x 19-34 at call 2 this
    turn`).
  - A zone on land that isn't free says what the rectangle holds ("all 14 tiles are road",
    "6 zoned C (4 developed), 2 road"), and names the role's own zones of that turn that
    took those tiles (`you zoned them at call 17 this turn`). Where free roadside land lies
    beside it, the hint suggests up to 2 one-tile-wide strips there to zone instead. While
    the city lacks a pump or a sewage outlet, those strips skip free riverbank lots.
  - A failed dezone says what is there.
  - High density before its unlock gets `zone with density "low" now`.
  - A build on a tile that isn't free says what is there (`(29,23) now holds your water
    pump (call 5 this turn)` when the role's own build took it that turn) and names the next
    free site on the role's list for that building, or why the list is empty. In smoke
    v12–v16, 15 of 19 such builds went where the role had just built.

  These hints go to the model only; the history log and the turn's action results keep the
  game's message. From the 4th `inspect` in a turn, while `map_overview` hasn't been called,
  the result points to `map_overview`. After it, from the 3rd further `inspect`, the result
  says its map already shows the layout and to inspect only a tile about to be acted on.

## Guards

- **Guards.** A `bulldoze` that would cut highway-connected road tiles off the highway
  (the road network relabelled without the removed tiles, as `World.refresh()` does) is
  refused before it runs (`refused: "disconnect"`: not executed, not charged as an invalid
  action, still a tool call), naming how many tiles and where; the agent can build a new link
  first or repeat with `confirm: true`. Transport's `demolish` of a train station, harbour or
  airport that is some roads' only link to the region (metro stations don't link) is refused
  the same way; the refusal suggests asking planning for a road to the highway or building
  another link first. Transport's `upgrade_road` is refused (`refused: "bus_line"`) while
  the city has no bus line, its first one is ready and the upgrade would leave transport
  short of it; the refusal names the line as one `create_bus_line` call. In the 120-month
  LLM runs (v19) transit scored 0.00 and 0.28: seed 0 never built a bus line, and at month
  78 put $690 of $2,000 into avenues, after which the $1,500 line failed. Transport's
  briefing says the first line comes before any avenue, and its Key metrics count bus
  riders against seats. The bus_stops list reads as one line (`6 stops for one line:
  create_bus_line stops [[40,5],...]`, and `find_sites bus_stops` returns `stops`),
  since the model had built lines of 1–2 stops from "6 sites, best (40,5)". It offers
  another line while every line carries 90% of its seats, or while transit is under the
  12% transport's score counts and the next line would carry at least 1% more of the
  trips (at most 4 lines). A trip rides the best line covering both its ends, so a second
  line on the same stops buys nothing; the next line's stops are today's big job centres
  plus the home areas no stop reaches. `transit_estimate` runs the traffic model's own
  trip split (bus after metro, at the fare, up to the seats) on the lines as they are now,
  writing nothing, so a line built this turn counts at once and what a new one adds is
  exactly what the next month's traffic pass gives. In
  the v23 120-month runs one 7-stop line from month 12–18 took transit to 15–21%, then
  the city outgrew it (4–6% by month 48, 124–196 riders of 1,500 seats) and no second
  line was offered; the rule after it (40% of residents beyond every stop) still offered
  none at 61% near a stop and 9.6% transit (region run, north, month 66), as job centres
  moved. Congested roads are listed from 1.0× capacity, as the scripted transport
  upgrades them. A failed `create_bus_line` names the suggested stops.

## Next lines: transport and planning

- **Transport's next step.** Transport's budget block has a Next line every turn, showing
  the build its needs line (`first_build`) names, so the Next line, the mayor's line and
  transport's ask always agree on the build and the amount: the first bus line, else
  another line its list offers (`Next: another bus line ($1,500): transit 6.6% now, about
  13% with it (your score counts transit up to 12%)`), else the most jammed stretch with
  its whole price and `upgrade_road` call, else, with nothing else ready, the cheapest
  train station, harbour or metro station and its build call, else that nothing is to
  build (before 1,500 residents, how far off the first line is). Each successful action
  of a model transport agent ends with this line recomputed (`| next: ...`, or `| next:
  nothing left: end_turn`; see Planning's next step). `first_build` names the
  line before any avenue stretch, so the mayor reads
  `needs $1,500 for another bus line (transit 6.6%, $76/month)` and transport asks for
  it. In the 120-month LLM runs (v19, v23 and the region run, 240 transport turns) the
  cheapest site was named instead: a $30 avenue tile undercut the $1,500 line on 131 of
  the mayor's transport lines (another line: 9), transport got 0.8–3.2% of the mayor's
  top-ups, 47 of 52 region turns that listed another line read "funds cover your next
  build" with $30–750 and never asked, and 156 of 240 turns did nothing. The playbooks no
  longer say "don't spend before ~1,500 residents" or "another only when buses run 90%
  full": 122 turns past 1,500 residents still had "waiting for 1,500 residents" in memory.
- **Planning's next step** (`fleet/planning.py`). Planning was the lowest LLM department
  (v19 / v23 / the region run / the late smokes: 62.5–76.6) and repeated its mistakes
  across quarters (178 recurrences in the Problems tab, the next role 33). Measured over
  its 244 turns in those runs: no parks (1 park tile built; R demand sat at the +0.25 cap
  for lack of parks from about month 24 to the end in every run, no home was within 4
  tiles of a park, and developed land value was 14–19% against 44% in the scripted
  warm-up city; land value and demand balance are its two biggest score losses);
  utilities' riverbank lot (in 103 turns its pump or outlet waited for a bank lot by a
  road, and planning spent 10–20 inspects a turn hunting for it: 417 in 36 turns of v23
  seed 0, 52 turns on the step limit); zoning over budget (134 failures, 131 after the
  turn's earlier actions spent the money, 72 recurring the next turn; the zone lines had
  no price); requests to planning (176, 98 of them utilities' bank lots, 85 unanswered;
  in the late smokes police's jail site and health's landfill and cemetery at the town
  edge got no answer). So its budget block has a Next line every turn with up to three
  builds, each priced with its exact call: the riverbank lot utilities waits for (dezone
  a zoned bank lot, else a road that reaches free bank), a road whose far end is away
  from homes for a blocked jail, landfill or cemetery, a park block while R demand is
  capped for lack of parks (the 2x2 block that raises the most developed homes, shops and
  offices, off the roadside lots), the zone type with the largest demand and fewer than
  10 tiles waiting on the strip its Ready line shows (sized to the funds left, sparing
  funded lots; a locked type is skipped, one whose strip has no usable tile gives way to
  the next), then a road into open land; short of the builds' total it asks the mayor for
  the difference only when it may ask (the mayor's needs line still prices only its road):
  `- Next: 2 park tiles by 20 homes (parks 14 of 16; R demand capped at +0.25 until 16)
  ($300): build_park {...}; then zone 10 tiles R high density (...) ($80): zone {...};
  then a road into open land (...) ($300): build_road {...}`. Its zone lines carry their
  price (`zone_R: 6 sites, best (10,12)-(14,13), 10 tiles; $50`), a parks line names the
  park tiles against one per 400 residents and the best block, Key metrics say what caps
  demand (`- demand capped: R +0.25, C +0.25: parks 14 of 16 tiles (yours: build_park), no
  train station or harbour (transport's)`), and an Open requests block (as the mayor's)
  lists each department asking it, with what to do (`dezone {...}`, `leave (0,29) free,
  the far end of road_extension ...`, or reply where or why not); the model's first
  `end_turn` with any open is refused once. A failed dezone names the bank lot utilities
  waits for, a road blocked by a building names a detour (`around it: build_road {...},
  then build_road {...} ($80)`; s1 v23 sent `build_road` through the same water pump 16
  times), and a zone leaving under $100 says what the rest pays for. At a warm-up
  hand-over the memory the scripted fleet wrote is emptied (the late smoke's planning read
  "build_park: no empty land tiles there" as a lesson, and its "asked mayor: ..." open tasks
  were requests the new agent never sent). Each successful action of a model planning or
  transport agent ends with its Next line recomputed on the world as it is now, at most two
  builds and the funds-short ask as the line words it (`zoned 8 tiles C (high density) |
  next: a road into open land (...) ($350): build_road {...}`, `| next: nothing left:
  end_turn`, or that the turn's action cap is reached); about 15 ms an action on the
  seed-1 month-96 city. The late smokes' failed Next-line calls (s1 month 96, s0 month
  102) had all succeeded at the turn's first calls: 6-15 calls later the model re-read its
  briefing and sent them again word for word. A failed action that repeats one that
  succeeded this turn now says it is done and that the briefing predates it, and a
  `build_park` on taken land names what holds it and the best park block now. None of
  this changes what an action does: the scripted fleet plays turn for turn as before.

## Money guards and money results

The mayor's `allocate_budget` identical to one that already succeeded
this turn (same department and amount; allocations add up) is refused too
(`refused: "duplicate"`), naming what the department now holds; other amounts pass. An
identical take-back is a duplicate as well (`you already took $1,140 back from planning
this turn (it now has $1,240); take-backs add up`); in v19 seed 1 month 6 the mayor
repeated its take-back pairs. So is a take-back (a negative amount) that would leave a
department below what it keeps (see Idle money) of what it held before this turn's
top-ups (`refused: "take_back"`): `that takes planning below the $1,316 it keeps (2 x
its $658 spending), so it can't pay for a road into open land and zoning it ($640); it
has no idle money: leave its funds. To take it anyway, repeat this exact call with
confirm: true.` It names why the department keeps that much, what the money left can't
pay for (only when it is short of the department's next build; in review, "so it can't
build next quarter" was false with planning's $640 road still covered by the $950
left), what this turn already took back from it, and what may go back instead (`its
idle money is $1,508`, `only the $800 you moved to it this turn`).
Taking back idle money or this turn's top-ups passes, as does any take-back from a
department that isn't in the game. In smoke v18 seed 0, at month
6, the mayor read planning as `idle $1,508` and took back all $2,508; planning had $0
at months 9 and 12. On seed 1 it took $1,182 of planning's $1,479, nothing of it idle,
then its last $297.

A top-up for another building (the department's kind is "another" on the to-fund line),
or one that buys nothing now (kind "nothing" or "idle"), that would leave the treasury
at least $1 under the reserve is refused too (`refused: "reserve"`): `this top-up for
another fire station leaves the treasury at $1,803, $1,413 under the $3,216 reserve (6
x $536 expenses: last month's deficit was $22); first/supply and growth come first:
education $2,500 (first/supply), planning $315 (growth) still wait. To fund it anyway,
repeat this exact call with confirm: true.` With nothing of those waiting it says what
the reserve is for (`the reserve keeps the city solvent and pays for first buildings,
supply and planning's road when they come (none waits now)`), a top-up to a department
with nothing ready reads `this top-up buys nothing now (transport: nothing ready to
buy) and leaves ...`, and it names idle money elsewhere to take back first when there
is some. First buildings, supply, planning's road, a department blocked for lack of
land, and a top-up that leaves the treasury at or above the reserve all pass,
as does fire's while the briefing's forecast names a flood (levees are first/supply, and
the playbook's flood rule funds fire at once, although fire's needs line names its
cheaper next station). A top-up larger than the treasury gets the action's own failure
and its money note. In smoke v20 seed 0, month 6, the to-fund line read `$4,305 covers
planning, police; the other $6,500 would go under the reserve`, and the mayor funded
another fire station and another clinic anyway, leaving the treasury at $303 of a
$3,216 reserve.

A `take_loan` is refused (`refused: "loan"`) while the treasury holds $3,000 or more
(`the treasury holds $3,598, over the $3,000 minimum: a $5,000 loan costs $25/month
until repaid (debt would be $5,000 of $50k); borrow only for a first building or supply
need the treasury can't cover`). It is also refused when the turn started at $3,000 or
more and this turn's own moves took the treasury under: top-ups that aren't for a first
building or supply, named with their amounts (`take back (free) a top-up that isn't a
first building or supply (police $2,500, fire $698) instead of a $3,000 loan at
$15/month`), or a repayment, which the loan would only borrow back (`repaying $4,000 took
it under $3,000: a $4,000 loan at $20/month borrows back what you just repaid`). When
every top-up was for a first building or supply, that part stays silent. Whatever the
treasury at the turn's start, a loan is also refused while no first building, supply or
planning road waits on the to-fund line: `no first building, supply or planning road is
waiting (only another: police $2,500, fire $2,500); this $3,000 loan would pay for
another building at $15/month interest until repaid (debt would be $3,000 of $50k): fund
another building only from the money above the reserve.` With nothing listed at all it
says the loan `would sit in the treasury`. While a first building, supply or planning
road waits (fire's top-up too while a flood is forecast: levee money), or operations
would take the treasury below $0 before the mayor's next turn (last months' pace ×
`act_every`), it refuses only a loan $1,000 or more over what those, the bills and the
reserve need: `only $3,850 of this $5,000 loan is needed ($1,400 for first/supply and
growth, $2,450 to keep the treasury at the $3,000 reserve): the other $1,150 would pay
for another building at $6/month interest until repaid (debt would be $5,000 of $50k):
borrow $3,850 or less.` It stays silent while the treasury is below
$0, where a loan pays the bills. With one of the first two cases it comes
after that case's text as one refusal (`Also, no first building, supply or planning road
is waiting (...): the loan would pay for another building.`), which one override
passes. A loan the action itself refuses (over the $50,000 cap, not positive) gets the
action's message. In smoke v13–v19, 10 of 15 loans ($34.8k of $60.0k) came in turns
that started at $3,000 or more, and 14 of 15 after the mayor's own top-ups had emptied
the treasury. The 120-month run sat at the debt cap from month 45 and paid $23.9k in
interest. In smoke v20 seed 0, month 9, a turn that started at $550 took loans of
$5,000, $3,000 and $2,500, each sent with `confirm: true` on the first call. No guard
fired, and the money paid for another police, fire and health building. Replayed on a
state rebuilt from that briefing (its treasury, reserve and needs lines), the police and
fire top-ups are refused for the reserve, and the second and third loans for the
treasury and because only other buildings wait. The treasury case stays silent while
the first buildings, supply and planning road waiting cost more than the treasury and
idle money above $3,000 cover: in smoke v22 seed 1, month 9, a $4,500 loan with $3,607
in the treasury and a $5,000 first high school waiting was refused and overridden.

**Earned confirm.** For the mayor's `allocate_budget` and `take_loan` the guard always
runs, and `confirm: true` overrides a refusal only on a repeat of the exact call
(action, department, amount) that was refused this turn. Each refusal earns one
override. The result then carries `"override"` with the refusal's kind, and the fleet
report counts it. A call that sent `confirm: true` without earning it is refused with
`(confirm: true counts only when repeating this exact call after it was refused)`. In
v19 all 3 take-back refusals were overridden at once, and 3 of 7 duplicate refusals.
One exception: after a loan this turn, a reserve refusal can't be overridden (a loan pays
only for first buildings, supply and planning's road). In smoke v22 seed 1, month 9, the
mayor borrowed $4,500 for a first high school, then confirmed another landfill that left
$1,107 of a $3,000 reserve. The other guards stay soft. The harness never picks
an amount or enforces the reserve: the reserve and extra-building refusals cost one
repeat of the call.

**Money results.** The mayor's `allocate_budget`, `take_loan` and `repay_loan` results
say where the money stands, for the model only (the history log and `executed` keep the
game's message):
- a top-up says what the department can buy now: `transferred $2,500 to police (now
  $2,500) | police: can pay for another police station (upkeep +$55/month); treasury
  $2,788, $212 under the $3,000 reserve; still to fund: education $2,500
  (first/supply)`. Short of it, the note reads `still $1,000 short of its first
  elementary school: less buys nothing` (`less buys fewer tiles` for levee or avenue
  tiles, and nothing added for planning's road, where less buys part of it). It can
  also read `needs nothing now: supply has room`, `had only a library ready ($1,500)`,
  `had nothing ready to buy` or `blocked for lack of land`.
- a take-back says what the department holds against its keep: `planning now $1,240,
  what it keeps (nothing idle)`, `$508 over what it keeps`, or `$1,316 below what it
  keeps (2 x its $658 spending): it can't pay for a road into open land and zoning it
  ($640)` (the last part only when what is left is short of its next build).
- a loan gives the interest and the credit left: `interest now $15/month (+$15); $47k
  more can be borrowed`. At the cap it reads `no more credit: repay_loan from money above
  the reserve cuts the $250/month interest`. A repayment gives `interest now $225/month
  (saves $25/month)`.
- a top-up or repayment the treasury can't pay names the idle money to take back and
  what is still to fund.

The `take_loan` and `repay_loan` descriptions give the interest in dollars ($5 a month
per $1,000, read when the toolbox is built, so a `LOAN_INTEREST` override reaches it;
nothing about the score, which no other LLM agent is told). In smoke v13–v19, 234 of
338 allocation messages had under 40 characters
of content, so the model's step-0 plan was lost and block order became funding order.
The treasury also ran dry mid-turn in 10 of 56 turns.

A `zone` that takes the last free riverbank lot beside the connected
roads while the city still has no pump or water tower, or no sewage outlet / treatment
plant, succeeds, with a warning naming what utilities still needs the bank for. So does
a `zone` that leaves a department with funds for its cheapest ready first building no
unclaimed site for it (`WARNING: that zoned the last site for fire's fire station near homes (fire holds
$2,500 for it): dezone one, e.g. (20,22)`): only when claiming the sites afresh in turn
order seats fewer of the funded buildings than before; the buildings named are those
left out when each keeps its earlier site where it can. With several, each gets its own
tile from its sites the zone took (`dezone (46,29) for the fire station and (46,35) for the
clinic`); in smoke v18 seed 1 planning dezoned the one tile named for two buildings. A
landfill or jail gets a tile that is away from homes once dezoned, else the warning says
it needs a lot at the town edge (`leave one by (40,19), the far end of road_extension
(25,19)-(40,19)`). A `zone` that leaves no dead end to extend and
no free lot beside the roads from which a straight road reaches open land warns as well
(`WARNING: that zoned the last lot a new road could branch from; leave one free, e.g.
dezone (12,23)`). After another warning on the same zone (the riverbank one, or the
game's "no road access yet"), it adds `; it also zoned the last lot a new road could
branch from` instead of a second warning.

## Tools

| tools | |
|---|---|
| look-ups (no side effects) | `city_overview` (headline numbers, a 3-month trend, `operations_3_months`, the treasury change split as in the Finances block, and `costs_last_month` by kind), `map_overview` (the whole map as ASCII with rulers and a legend, downsampled on maps too big for one tool result, plus the facts of the briefing's Map block; when the map and the facts together don't fit, as from about month 66 of a long game, the road lines go first, the longest kept while they fit, then the facts give way to a pointer to the Map block), `inspect(x, y, radius)` (tiles within 4: zones and levels, buildings and tiers, roads, river, power / water / pollution), `find_sites(purpose, near_x, near_y)` (ranked sites for the role, e.g. `zone_R`, `road_extension`, `pump`, `clinic`, `bus_stops`, `levee`, `library` (the school list); sites on roads cut off from the highway are dropped, and the pump and sewage lists then top up from free riverbank lots beside the connected roads; the mayor's read-only version takes `<department>.<purpose>`, e.g. `utilities.sewage`, to check which sites a department has), `my_facilities` (the role's buildings with load, capacity, tier and the upgrades on offer; for the mayor, funding levels, policies and imports), `unlocks`, `get_plan`, `get_messages`, `history(kind, months)` |
| turn control | `send_message(to, text)` (`to`: a role, `all`, or several roles separated by commas, one message each), `update_memory(commitments, open_tasks, lessons)`, `update_plan(goals, budgets, priorities, notes)` (mayor only; the result names any idle money to take back first, lists the departments below their budget target that would use a top-up (as in the briefing) with the treasury, gives `to_spend`, `to_fund` (department, kind, amount, upkeep per month, in the briefing's order) and `first_buildings` (first/supply departments at a $0 target, utilities' supply included), and its `next` is the briefing's to-spend and to-fund lines; the reserve rule in `fleet/tools.py:reserve`, $3,000 or 3 months of expenses, 6 after a deficit on rounded dollars, mirrors the fleet configs' mayor playbook), `end_turn(report)` |
| actions | one tool per action of the role (see [Roles and actions](SIMULATION.md#roles-and-actions)). The JSON schemas come from `actions.action_docs()`, and the actions run through `apply_action`, so validation, costs and milestone locks are the same as for every other agent |

## Memory and limits per turn

**Three memory layers.** Nothing else carries over between turns.

1. **City plan**, owned by the mayor. In `update_plan`, goals (up to 5) and notes replace the old ones, while budgets and priorities merge per department. Every version is kept.
2. **Department memory**, owned by each agent: `commitments`, `open_tasks` and `lessons`, up to 8 items of 200 characters each. It's shown in full in the agent's briefing.
3. **History log**, written by the game: every action with its result and cost, end-of-turn reports, city events and a monthly metrics snapshot. Agents query it with `history(kind = action | failure | event | report | metrics | all, months up to 12)`. A department sees its own entries and the city's; the mayor sees everything.

The memory lives on the env (`env.fleet`) for the length of the run. The final plan and
lessons go into the fleet report, and every edit is in the tool traces.

**Limits per turn** are set in `[fleet]` or per role:

- `max_steps` tool calls (default 20)
- `max_tokens` per model call (16,000)
- `turn_tokens` completion tokens per turn (80,000)
- `turn_budget` prompt (request characters / 4) + completion tokens per turn (0 = off; civitas-v3: 100,000, with
  `budget_reserve` 4,000, the least kept for the next reply; see [civitas-v3](EVALS.md#civitas-v3-an-equal-token-budget-per-turn))
- the game's `max_actions_per_turn` successful actions

A model that answers without a tool call gets one nudge, and a second one ends the turn.
A reply cut off at `max_tokens` before any tool call gets its own nudge instead (keep the
reasoning short, pick one action and call its tool now), up to 2 a turn; after those it
counts as any other reply without a call. No nudge is sent once the turn has used
`turn_tokens`: the turn ends there as `turn_tokens`.
Tool calls a model leaves as text (`<tool_call>` blocks in its reasoning, which
trinity-large-thinking sometimes does) still run, and the trajectory counts them in
`recovered_calls`.
On a limit or an API error the turn ends, and completed actions stay. Every turn records
why it ended: `end_turn`, `max_steps`, `turn_tokens`, `turn_budget`, `no_tool_calls` or `api_error`.
Every tool result carries `steps_left` (except `end_turn`'s and the "not run" ones for calls
after `end_turn` or past the limit), plus a `wrap_up` note once 3 or fewer calls are left.
Within a turn, an identical repeat of a look-up (same tool, same arguments after type
coercion) returns the earlier result marked `repeat`, and an identical repeat of an action
that failed is refused without running (`refused: "repeat"`, not charged as an invalid
action), as is one of an invalid call (bad arguments or JSON, or an unknown tool; same
arguments as sent), with the earlier error, until the agent's next successful action. A
successful `update_plan` also refreshes look-ups, and a failed action refreshes `history`
(which lists it). Every kind of repeat still counts as a tool call. A guard refusal (above)
is not remembered as a failure: an identical repeat is checked, and refused, again.

## Coaching with playbooks

**Coaching with playbooks.** The system prompt starts with a fixed part: how to play (start
from the briefing and look up only what it lacks), the role's mission and the game rules.
Then comes the coach's playbook: the fleet playbook (`[fleet].playbook`, shared by every role) and the role's own (`[agents.<role>].playbook`).
Coaches can steer strategy this way but can't change the rules or the tools.
The "hold" advice and the to-spend line follow a reserve rule that lives in code
(`fleet/tools.py:reserve`): a coach who changes the mayor playbook's reserve rule must
change it there too. The example configs' mayor playbook reads the money lines: fund every
first building a needs line names from month 3, whatever its target, and give utilities
what its needs line asks after month 0. Each turn it takes back idle money first, then
funds what the to-fund line says the money covers, in its order (the cover is greedy, so
it doesn't stop at the first item that doesn't fit). First buildings, supply and
planning's road may use the treasury above $3,000, anything else only the money above the
reserve. It borrows only for a first building or supply need the money can't cover,
repays from the money above the reserve, and does what a refusal says instead of
confirming it.
`civitas prompt -c configs/fleet_trinity.toml --role planning` prints a fleet agent's
system prompt, its first briefing and its tool names.

## Modes and supervised sessions

**Modes.** `mode = "autonomous"` fixes the instructions at month 0 and the fleet plays to the
end on its own; this is the standard eval track. `"supervised"` plays in sessions, and the
coach may revise the playbooks between them. The mode is recorded in the run (the signature
ends in `[autonomous]` or `[supervised]`) and shown in briefings. All fleet agents in a run
share one mode.

- **Pausing.** With `[fleet].session_months = 12` (default 0: no pauses), a supervised run
  stops every 12 months, before that month's turns, prints how to resume and leaves three
  files in `runs/<run_id>/`:
  - `checkpoint.ckpt`, the whole game as a gzipped pickle: the env with its world and RNG,
    the council, the scores history and the fleet's plan, memory and log; every agent's
    state; the runner's stats; and where the logs ended. With a fake model, fleet_smoke's
    fleet paused at month 12 left 49 KB (1.1 MB unzipped), and 81 KB with scripted
    departments growing the city. Chat clients aren't saved (they hold the API key) and are
    rebuilt on resume.
  - **Signed.** Unpickling runs code, so the file starts with an HMAC-SHA256 of the pickle,
    keyed by a secret made on the first pause: `~/.config/civitas/checkpoint.key`, mode
    0600 (`$XDG_CONFIG_HOME` moves it, and so does `CIVITAS_CHECKPOINT_KEY_DIR`, which the
    tests use). Resume checks it before unpickling. A checkpoint with no signature or a
    wrong one is refused, saying why: the file was changed, or it was made on another
    machine or account. So is one with no key to check against, or a key file other users
    can read. A checkpoint only resumes where it was made, so a shared run folder can be
    replayed but not resumed.
  - `playbooks.toml`, the run's current playbooks and `session_months`, ready to edit.
  - `PAUSED`, with the month, the session and the resume command.
- **Resuming.** `civitas resume <run_id>` merges `playbooks.toml` over the run's config, or
  the file given with `-c` instead (an edited copy of the run's config works). Tables merge
  key by key, and so does each `[agents.<role>]`. It then applies `--agent` / `--set`, builds
  the fleet agents again from the merged config, keeping their usage and state, and plays the
  next session with the new system prompts.
  - Playbooks and the other fleet settings may change: model, limits, `session_months`
    (`--set fleet.session_months=0` plays to the end).
  - A change to `[game]`, `[scoring]`, the line-up or the mode is refused, naming what differs.
- **Logs.** Frames and trajectories append; lines an interrupted session logged after the
  checkpoint are dropped first. Each trajectory record carries its `session`. `meta.json`
  gets a `sessions` list: months, status, `fleet_version`, a hash of each playbook, and the
  roles whose system prompt changed with the new prompts (`sessions[i].fleet_prompts`, which
  those roles' later turns point to; `fleet_prompts` keeps session 1's). The summary counts
  the sessions. A run that played more than one `fleet_version` lists them in
  `fleet_versions` and gets a `fleet_version` hashed from the sequence.
- **Determinism.** `session_months` is left out of `fleet_version`, and a run resumed with
  unchanged playbooks plays exactly like one that never paused: same frames, trajectories
  and scores (`tests/test_sessions.py` resumes in a fresh process).
- Autonomous runs and `civitas bench` ignore `session_months`. `civitas serve` lists a
  paused run as `paused`, and `POST /api/resume {"run_id": ...}` resumes it with its
  `playbooks.toml` (the viewer has no button for it yet).

```bash
civitas run -c coached.toml                 # [fleet] mode = "supervised", session_months = 12
$EDITOR runs/<run_id>/playbooks.toml        # after each pause
civitas resume <run_id>                     # or: civitas resume <run_id> -c edited.toml
```

## Late-game smokes and the yardstick

**Late-game smokes.** A 12-month smoke from month 0 only ever tests the opening. A warm-up
lets the fleet start from a grown city instead: `[game] warmup_months = 60` (or `--warmup 60`
on `run`, `bench` and `prompt`) plays months 0–59 with scripted agents and then hands the same
game to the configured agents. `game.months` stays the whole game, so 60 warm-up months and 12
fleet months are `months = 72`; a warm-up as long as the game is refused.
`configs/fleet_smoke_late.toml` is fleet_smoke's fleet that way, and
`configs/fleet_region_late.toml` is fleet_region_both_120's: both cities, each from month 96
(`months = 108`, `warmup_months = 96`: at month 60 the seed-0 city had only 2,724 residents,
transit at 17% and no shortage, so a 60-month warm-up still tested the middle game).

- **Who plays the warm-up.** Every fleet agent becomes the scripted fleet (`model =
  "scripted"`, its playbooks kept, the scripted fleet's own 20 steps). So the plan, the
  department memory and the history log the fleet inherits were written through the fleet
  tools, and the game up to month N is exactly configs/fleet_scripted.toml's with the same
  `[game]` (`tests/test_warmup.py` compares the frames). `llm` and `remote` agents play the
  warm-up as scripted agents. Scripted, noop and random agents play through as configured.
  In a region (the scripted fleet plays single cities only) the warm-up uses plain scripted
  agents, with their deals and links. At the hand-over each fleet mayor gets the plan the
  scripted fleet would have written: its goals, and the warm-up mayor's working balances as
  budgets. Each fleet agent also counts as having taken the warm-up's last turn, so its first
  digest starts there.
- **Hand-over.** The configured agents are built and bound when the run starts. They take
  over at the first acting month at or after N, keeping the env, council, scores and fleet
  state. Turn order and `act_every` go on unchanged.
- **Logs.** `frames.jsonl` covers the whole game, and the viewer shades the warm-up on the
  timeline ("scripted warm-up"). `trajectories.jsonl` holds every turn, the warm-up's marked
  `"warmup": true`; the Agents tab tags them "(warm-up)". `meta.json` records `warmup_months`
  and the warm-up line-up (`warmup`). The signature ends in `warmup:N` (before the mode), so
  leaderboards keep warm-up runs apart.
- **Scores.** The final score is still that of the whole game (scoring is a function of the
  metrics history). `summary.json` adds `warmup`: the city score and population at the
  hand-over and at the end, the change (what the configured agents' months added), their
  average city score, and the warm-up's own agent stats. `agent_stats` count only the
  configured agents' turns. The run's last line prints the same, and its scripted yardstick:
  `after the warm-up (months 96-107): city 65.3 -> 67.5 (+2.3), population 5,040 -> 6,736 (+1,696); scripted from the same start: +1.4 / +868`.
- **Yardstick.** At the end of a warm-up run the runner plays the same game again with no
  logs and no model: the run's seed, `[game]` and `[scoring]`, every agent scripted for the
  whole game (`warmup_config`'s line-up: the scripted fleet in a single city, plain scripted
  agents in a region). Months 0 to N−1 are the run's warm-up, so the city at the hand-over is
  the run's (`matches_run`), and only the agents after N differ. It takes ~5 s and goes to
  `yardstick.json`: `summary["warmup"]`'s shape (city score and population at the hand-over
  and the end, the change, the average city score after), the departments' final scores, the
  city score and population per month from N (per city in a region, `by_city`), and the run's
  own change under `run`. `--no-yardstick` skips it, `civitas bench` never plays it, and
  `civitas yardstick <run_id>` computes it for a finished run (it refuses a run without a
  warm-up). A yardstick never calls a model: `bench/yardstick.py` refuses a line-up config
  with an `llm` or `remote` agent, or a fleet agent on a model other than `scripted`, and then
  checks the built agents' exact classes (the scripted baselines, the scripted fleet, noop,
  random) and that none holds a model client, before a month is played (`ModelAgentError`).
  `GET /api/runs/<id>/yardstick` returns it, or `{"missing": true}` with why; for a finished
  warm-up run without one the server starts `civitas yardstick` in the background
  (`computing`). The viewer's left dock reads "since hand-over +2.3 (scripted +1.4)" and the
  residents at the playhead, and the Compare view shows it beside the run's numbers. The
  late runs' yardsticks (months 96-107): single city seeds 0 / 1 / 2 +1.1 / +268, +6.2 /
  +1,532, +1.4 / +868; region seeds 0 / 1 +0.7 / +1,360, +5.1 / +3,520 (seed 0 from month 60
  in a 72-month game: city 55.8 -> 62.3, population 2,724 -> 4,436).
- **Reports.** The fleet report, the Problems tab, the Team tab and the collaboration score
  cover the months after the warm-up only (`run_collab` reads `warmup_months` as `since`).
  The report's header names the warm-up and the score change. In a region, `deals` keeps the
  whole life of a deal made during the warm-up.
- **Prompts.** `civitas prompt` plays the warm-up first, with no model and no logs, and
  prints the role's first briefing after it (in the hand-over month, the roles before it play
  with the warm-up line-up). It takes about 3 s for 60 months.
- **Sessions.** A supervised run counts its sessions from the hand-over: with
  `session_months = 6` the first pause is at month N + 6, and resume works as usual.
  `civitas bench --warmup N` passes it to every seed.

On seed 0 a 60-month warm-up (`--set game.months=72 --warmup 60`) starts the fleet with 2,724
residents and a city score of 55.8. Transit is at 17% on one bus line (230 of 1,500 seats), no road is congested, and the
5,000-resident milestone (rail, harbour, hospital, university) is still ahead. The mayor
has $10.2k, $11.8k to spend, and four buildings to fund: education's first high school, plus
another police station, fire station and clinic. From month 96 (the configs) it has 6,036
residents and a score of 66.8; transit is still 17-20% on seeds 0-1 (the scripted transport keeps
it up), and seed 0 has a jammed stretch for an avenue.


The first late smokes and what they changed are in [HISTORY.md](HISTORY.md#late-game-smokes-2026-10-0607).

## Config

```toml
[game]
act_every = 3                  # the city runs monthly; the fleet plans by the quarter

[fleet]                        # every fleet agent; any key can be overridden per role
mode = "autonomous"
session_months = 0             # "supervised" only: pause every N months (run-wide, see Modes)
provider = "arcee"
model = "trinity-large-thinking"
max_steps = 20
max_tokens = 16000
turn_tokens = 80000
temperature = 0.4
extra_body = {}                # extra request fields (a role's extra_body replaces this one)
playbook = """
- Keep a treasury reserve; utilities first, then services as the city grows ...
"""

[agents.mayor]
kind = "fleet"
playbook = """
- First turn: write the plan, then fund it ...
"""

[agents.health]
kind = "fleet"
model = "trinity-mini"         # a cheaper model for one department
max_steps = 12
```

Fleet and non-fleet roles mix freely, for example a fleet mayor over scripted departments.
`configs/fleet_trinity.toml` is the full fleet: all eight roles on trinity-large-thinking for
120 months, acting every 3 months (320 turns), with a default fleet playbook and one
playbook per department. The fleet playbook tells departments that their briefing already
has their plan slice, budget, new messages, map and ready sites, so they shouldn't re-read
them with `get_plan` / `get_messages` / `history`, and that when it says they can't pay for
any ready site they should send at most one request and end the turn.
`configs/fleet_smoke.toml` is the same fleet for 12 months.
`meta.json` records `fleet_version` and every fleet agent's system prompt once
(`fleet_prompts`); `summary.json` records `fleet_version` too. `fleet_version` is a hash of
`[fleet]` and the fleet agents' settings, so runs with the same coaching can be compared: the
leaderboard groups fleet runs by signature and `fleet_version` (shown in its own column). The
signature reads `role=fleet:<model> ... [mode]`.

## Regions

**Regions.** Fleets play region games too: `[agents.<role>]` puts the fleet in every city
and `[agents."north.<role>"]` in one, so fleet and scripted cities, and fleet and non-fleet
roles, mix freely. A fleet agent binds to its own city's env (`fleet/region.py`), so each
city has its own plan, department memory, history log and briefings, and the north mayor
plans and funds north's departments only. The system prompt gains a Region section (the
other cities and where they lie, the council's addresses, the river, links and, for the
mayor, deals). The mayor's briefing gains a Region block: per other city its population,
specialisations, surplus and shortfall, unemployed and open jobs, links, what a connector
would cost transport and last month's goods, commuters and tourists between the two, then
the river's pollution and up to 4 deals with what to do about them (an offer to the city
with the month it is open until, whether it is worth taking or why not, e.g. "can't afford
it" or "already covered by deal #2", and the `accept_deal` call; an active deal with its last
month and, when the city should end it, the `cancel_deal` call; an offer that expired since
the last turn). Its City line shows supply bought from outside apart ("power 1,046 own +
35 imported / 1,081"). Below those come its deal
opportunities, at most one per resource, each with why, the exact `offer_deal` call and
what it pays or earns a month; without one, its spare beyond 90% of its own supply. The
system prompt's Deals rule gives the import and resale prices that make a deal pay.
Transport's block has
each neighbour's best connector site priced as `build_connector` charges it, and boat and
air links; utilities' has the river and what the deals do to supply. Every role gets
`region_overview`, the mayor gets `offer_deal`, `accept_deal` and `cancel_deal` (run as
the region runs any mayor's deals, and counted in the action cap), and `send_message`
takes a role of the city (`"mayor"`), a team (`"south"`), another city's agent
(`"south.mayor"`) or `"all"`, or several of them separated by commas. The fleet report has the region scoreboard and one section
per city with fleet agents. `configs/fleet_region_smoke.toml` runs the `fleet_smoke.toml`
fleet in the north city against the scripted baseline in the south.

## Scripted fleet

**Scripted fleet.** `model = "scripted"` makes a fleet agent play the role's scripted
baseline (`agents/scripted.py`) through the fleet tools instead of calling a model
(`fleet/scripted.py`). It costs nothing and runs a 120-month game in about 5 s (10 s acting
monthly). Each turn:

- it gets the briefing (logged as the turn's first message, like a model's);
- it reads what the policy needs with look-ups on top of `role_obs()`: `map_overview` for
  the departments that read the map, `get_messages` for the mayor's council requests;
- it runs the scripted decision as tool calls, arguments sent as JSON text. The mayor first
  writes its working balances into the plan with `update_plan` (goals, a budget and a
  priority per department). Each action is one call (a duplicate allocation, a
  take-back below a department's keep, a top-up for another building under the reserve,
  or a loan refusal that the policy means is repeated with `confirm: true`, which earns
  the override; the scripted mayor tops up only from the treasury above its own reserve,
  bar its flood rule, which the reserve guard lets through, and borrows only at a turn's
  start with the treasury under $3,000, which on seeds 0–4 never happened),
  then come `send_message`, `update_memory` (open
  requests, one lesson per kind of failed action) and `end_turn` with a report.

`configs/fleet_scripted.toml` plays all eight roles with the game settings of
`fleet_trinity.toml` (120 months, `act_every = 3`). `--agent police=fleet:scripted` puts
one scripted fleet role into any line-up, for example an LLM fleet department with scripted
colleagues that talk to it through the same plan, messages and reports.

It plays exactly as the scripted baselines: on seeds 0–4, at both cadences, every turn
has the same actions, results, costs and council messages
(`tests/test_fleet_scripted.py` checks a 24-month game). The tool layer changes nothing
for these policies. The mayor's money picture (the Finances split, the need-ordered
Departments block, the to-spend and to-fund lines, the money results, earned confirm and
the loan guard) and the reserve and extra-building guards added after smoke v20 changed
only what the fleet reads. The scripted baseline (77.5 ± 1.5), the
fleet reference below (68.8 ± 2.0, 7,341 residents) and the region games score exactly as
before (civitas-v1 numbers). On seeds 0–4, at both cadences, no guard refused a scripted call and
nothing was overridden in civitas-v1; from civitas-v2 the scripted mayor takes idle money back and the
take-back guard refuses 2–7 of its 5–9 take-backs per game (every 3 months, seeds 0–4: what the guard keeps,
twice 3 months' spending or the dearest ready first building, differs from the policy's working balance), each
confirmed at once as the policy means it, so the two still play turn for turn alike. Before the flood exception, the reserve guard met the monthly games of seeds 3
and 4 three times, each time on the policy's $3,000 for fire before a flood. On seed
0 the scripted mayor's longest briefing is 4,577 characters (4,474 before; 4,397 before the
review fixes named first buildings ahead of cheaper sites and what first/supply may use).

| seeds 0–4, 120 months | city | population | mayor | planning | utilities | transport | police | fire | health | education |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| monthly: `baseline.toml`, or `fleet_scripted.toml --set game.act_every=1` | 76.6 ± 1.6 | 13,598 | 89.6 | 67.3 | 90.6 | 73.6 | 74.8 | 87.4 | 86.1 | 83.8 |
| every 3 months: `fleet_scripted.toml`, or `baseline.toml --set game.act_every=3` | 69.2 ± 1.3 | 7,373 | 87.2 | 65.9 | 91.5 | 89.4 | 79.3 | 87.4 | 90.3 | 86.4 |
| civitas-v1, monthly | 77.5 ± 1.5 | 14,466 | 88.1 | 68.4 | 89.4 | 75.9 | 74.9 | 85.7 | 86.6 | 83.2 |
| civitas-v1, every 3 months | 68.8 ± 2.0 | 7,341 | 86.3 | 64.5 | 91.4 | 89.7 | 77.2 | 83.8 | 86.5 | 86.0 |

So compare a fleet with 69.2, not the monthly 76.6. The 7.4 points are the quarterly
cadence, since the scripted policies do about a month's work per turn (the notes below are civitas-v1's runs):

- **Planning** lays about one road and up to three zone strips per type a turn. It built 18
  road segments and 157 zone strips instead of 46 and 311, and spent $12.5k instead of $31k.
- **The mayor** reads the requests posted since its previous turn. At `act_every = 3` that
  is about 106 of the 108 that departments sent per game (seeds 0-4: 532 of 538; the rest
  came in its last turn; monthly: 529 of 531). The observation keeps 3 months of council
  messages, so this holds up to `act_every = 3`. It used to read only those from the month before its turn, which at `act_every =
  3` was none of the 146 sent, and this row read 67.0 ± 2.8, population 7,319 (mayor 84.3,
  education 70.0). Now the upgrades departments ask for get funded ($5,000 for education,
  $8,300 for police and fire), so fewer asks are repeated.
- **Money:** the mayor moved $231k to departments instead of $435k. Police and education
  asked for their first building from month 6–9 and built it in months 33–48 and 39–66
  (monthly: 34–40 and 35–48, the baseline's own slow start). Their $2,500 asks are under
  the $3,000 working balance the mayor already aims for, so reading them changes nothing
  there: the treasury is what they wait for.
- **Transport** scores higher (89.7) because it spends little in a smaller city.

What the runs showed about the tools:

- No guard fired. The policies never bulldoze or demolish, and never send the same
  allocation twice.
- There were no tool errors, no result was trimmed, and briefings stayed under 4,700
  characters for all 120 months.
- All 47 "zone beside it instead" suggestions after a failed zone worked when applied.
- Both funded-site warnings were right. A zone took utilities' only sewage-outlet lot, and
  sewage coverage then fell to 81–85% for months (the scripted planner doesn't read
  results).
- No briefing said "can't pay" on a turn the policy built something, and none said "no
  site" for a building the policy then built, except once: health's policy falls back to a
  clinic lot for a landfill when, as the briefing said, no lot away from homes is left.
- The coaching hints were noisy on a long game. The idle rule fired for 6 or 7 departments
  per run. That covered departments waiting for money they asked for every turn (police
  held $1,494–2,436 against a $2,500 station in months 6–33 of seed 0) and departments with
  nothing left to do (fire at 83–98% coverage). Planning's repeated failures (a plaza spot
  off the road, zones and parks on taken land) gave 3–4 repeated-failure hints per run.
  With the idle rule split by what the briefing showed and failures grouped by cause (below),
  seeds 0–4 give 27 hints instead of 46: 9 idle (30 before), 5 for the mayor (one per run,
  naming 5–6 departments short of their first building, education longest at 11–20 turns),
  3 for planning (utilities' sewage outlet lot, taken for 9–15 months), 9 repeated-failure
  (15) and 1 idle-money. The idle hints left:
  - transport holding $1,500–2,000 for a ready bus line (4 runs), or an avenue upgrade while
    it asked for $10k;
  - health or police holding $3,000 for a ready clinic, landfill or station while asking
    the mayor for $7,700–8,300 for an upgrade.

  The policies had reasons the briefing doesn't show (bus ridership, the upgrade).

  Since the mayor reads the requests sent since its previous turn, the same seeds give 29
  hints: 9 idle, 5 for the mayor (education longest at 11–21 turns), 2 for planning (the
  sewage outlet lot, 6 months each), 9 repeated-failure and 4 idle-money. The idle-money
  hints are utilities (3 runs) or health holding up to $6,044–7,200 against a $3,000 budget
  for 6–12 months: the mayor funds an ask one turn after it was sent, and the budget falls
  back once the department stops asking. No idle hint has police or health asking for an
  upgrade any more; new ones are education holding $2,880–4,245 for a ready school while
  asking for $5,000, and planning holding $7,000 after 2 failed actions (1 run each).

## Reports and the collaboration score

**Reports.** Every run with fleet agents writes `fleet_report.json` and `fleet_report.md`
next to its summary. They contain:

- the scoreboard
- the final plan and the months it changed
- per-role stats: turns, tool calls, actions, failures, tool errors, repeated look-ups, refused calls (repeats and guard refusals) and overrides (earned `confirm: true` repeats; fleet_report.md's `repeats (look-ups / refused / overrides)` column), spending, tokens, turns ended on the step limit and, apart, turns cut by civitas-v3's token budget (with a Token budget section: cuts, units per turn, overshoot), length stops and turns cut short by an API or briefing error
- a collaboration score (see below)
- the top failures, last reports and the lessons in memory
- coaching hints

**Collaboration score.** The report scores how the departments worked together, 0–100, from
`frames.jsonl` through the Team tab's request matching (`bench/collab.py`). It is for the
report only: it never enters `bench/scoring.py` and agents never see it. Only requests
between two roles in the game count. A request still inside its window (2 × `act_every`
months, the Team tab's "open") is left out. There are four components:

- **responsiveness**: the share of requests that were met, partly met, acted on or replied
  to within the window. A money request to the mayor with no top-up counts as replied to
  when the mayor messages the sender back, since a no with a reason is still an answer.
- **follow-through**: the same requests weighted by how far they were met: met 1, partial
  0.6, acted on 0.5, replied 0.25, unanswered 0
- **hand-offs**: the share of planning's roads, freed lots and cleared land that another
  department built on within the window. Openings still inside their window count only once
  used.
- **money aim**: the share of the mayor's top-up dollars that went where they were needed.
  A top-up counts when it went to a department's first building or supply, planning's road,
  or another building the money covered (the kinds of the mayor's Departments block and the
  to-fund line's cover clause), up to what the department lacked that month, or when it
  answered the department's money request. Take-backs are left out. The needs come from
  `frames.jsonl`: with fleet agents, each frame of a month the agents act in carries
  `needs`, `{department: [kind, top_up, covered]}` as the mayor's briefing computed them
  that month (`FleetState.needs`, from `funding_gaps()`; per city in a region frame; report
  only, agents never see it). A run without that record (scripted line-ups, runs from before
  it) keeps the old rule, the dollars that answered a money request, and the report says
  "message-based: this run has no needs record". If computing the record fails, the run goes
  on: the first failure is logged with its traceback, the frame goes without a record and the
  report adds "N months without a needs record". Since departments ask only for what the
  mayor can fund and the mayor funds first buildings unasked, the message rule alone read 43
  / 37 on the 120-month region run of 2026-10-05.

The score is the mean of the components that have at least 3 cases. A component with fewer
is n/a, and so is the score when every component is. `fleet_report.json` has it under
`collaboration`: the components with their counts, per-department rows and the three oldest
unanswered requests. `fleet_report.md`'s Collaboration section shows the score, the components
and a per-department table (asked, answered by others, asked of it, answered for others,
responsiveness, follow-through, hand-offs), then lists the oldest unanswered requests. A region
report has one per city; requests to another city are left out. On a finished run folder,
`fleet.report.run_collaboration(run_dir[, city])` computes the same score without the env. The
viewer shows it too, at the playhead (Team tab, below).
Scripted fleet (`fleet_scripted.toml`), seeds 0–4: 70.5 ± 2.0 (seed 0: 68.0; responsiveness
84, follow-through 59, hand-offs 48, money aim 81.5). With the message-based money aim it was
67.2 ± 1.7 (seed 0 64.9, money aim 69). The stored LLM runs below have no needs record and
keep the message rule; rebuilt offline from the mayors' to-fund lines, the 2026-10-05 region
run's money aim would read 92 / 89 (teamwork 74.5 / 73.6) and v23's 99.9 / 100.
On the 120-month trinity runs: 57.2 (v17, seed 0), then 74.0 / 76.6 (v19) and 64.8 / 77.0
(v23, seeds 0 / 1). A reply is no request of
its own (`collab.replies`): a message answering, within the window, a request from its
recipient, or a mayor's message with a money decision ("approved", "unable to fund",
`collab.DECISION`, which also covers answers to end_turn reports); a money ask back to the mayor
still counts. Without that, the 120-month region run of 2026-10-05 counted 32 of the north
mayor's 60 replies as unanswered requests, and the more the mayor replied the lower the score
went (v19 was 73.3 / 75.2 and v23 65.0 / 76.1 before it). Before the mayor's replies
to money requests counted, v19 scored 72.7 / 73.5 and v23 63.6 / 74.5. The scripted mayor
never replies, so its score didn't move. A status report that says it needs nothing is no
request (`collab.NO_ASK`: a negated need such as "No budget needed at this time" and $0 are
struck before the ask words are read): that took 1 to 2 transport and education reports out
of each of those runs (v19 seed 0 was 73.1, v23 seed 0 64.9; v17's 55.4 had gone stale
already) and left the scripted fleet's score as it was.

A hint fires on any of these:

- 3 or more failures of one role with one cause. Taken land, no road access and money
  group their wordings ("no empty land tiles there", "(21,26) is not an empty land tile",
  "tile (35,23) is occupied by coal_plant"); any other wording is a cause of its own. The
  hint names each action's first wording and count, and the lesson to write down.
- a department holding more than 2× its plan budget for 3+ months
- 2+ consecutive department turns of one kind without a successful action, spanning 6+
  months. The kind comes from what the turn's briefing showed (its funds, Ready sites and
  Key metrics):
  - **idle**: its funds covered a ready site it needed. The department's hint names the
    funds, the building, the metrics, what it asked the mayor for and its failed actions.
  - **starved**: its funds were short of the cheapest such site. One hint for the mayor
    names every such department (the three longest runs in full), what it was short of and
    how often it asked.
  - **no land**: none was ready, but some was blocked for lack of land or roads. One hint
    for planning names the departments and the cheapest buildings.
  - **nothing needed**: no hint. That covers coverage at 80% or more (for a utility, also
    demand under 90% of supply), no congested road, "nothing needs it", locked, and
    supplementary buildings (library, shelter, cemetery, train station, harbour, metro).

  Planning goes by its priced `road_extension` line, the road and its zoning summed: idle
  when its funds covered them, starved when short (`planning short of its $505 road into
  open land and zoning it`; older briefings priced the road alone, and are read that way).
  Without a road to build it is idle with $1 or more, since it can still zone.
  Without a briefing (older records), the funds come from the history log: under $1
  against a plan budget is starved, under $1 without one is nothing needed, and anything
  else is idle.
- a negative treasury
- a role ending more than 20% of its turns on a limit

`summary.json` gets a `fleet` entry with the report name and the number of hints. In
`trajectories.jsonl`, each fleet turn carries its full `tool_trace` (tool, arguments and a
slim result) along with the messages, reasoning, token usage and finish reasons. The viewer's
Agents tab shows the trace in each fleet turn's Tool calls panel, fetched when the panel is
opened from `GET /api/runs/<id>/trace?month=M&role=R` (the server keeps an index of where each
turn starts in the file).
