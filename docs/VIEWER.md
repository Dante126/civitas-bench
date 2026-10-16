# The viewer

`civitas serve` opens a browser control center: watch or replay a city, see every agent decision and tool
call, compare runs, read the team's collaboration, and launch runs and model evals.

Contents:

- [Open the control center](#open-the-control-center)
- [Panels: Turns, Team, Problems and Compare](#panels-turns-team-problems-and-compare)

## Open the control center

```bash
civitas serve            # then open http://127.0.0.1:8765
```

Everything happens in the browser from here. You can:

- **Watch** the city in the isometric pixel view (the default) or in 3D (toggle with `X`
  or the `PX`/`3D` button). The pixel view is a 2:1 isometric pixel-art map: drag to
  scroll, zoom in steps with the wheel, rotate in quarter turns with `Q`/`E`. Buildings go
  up behind scaffolding, cars, buses, trains, boats and planes move around, cars queue at
  traffic lights on junctions, planes land, taxi to the terminal and take off (in region runs
  they fly to the other cities' airports), container ships berth under the harbour crane, and night lights the windows. Around the city a distant view
  fades into haze: fields, the river and highways running on, villages where roads leave the
  map and, in region runs, the neighbouring cities with the rivers flowing between them and
  their highway and rail links routed from connector to connector (`T` zooms out over all of it). Both views
  show the river (a little murkier where it is polluted), bridges,
  houses, row houses and villas that become apartments and towers, offices, shop-houses,
  factories, farms, sawmills, mines and oil fields, power plants (nuclear too), pumps, sewage works,
  police and fire stations, clinics, 2×2 hospitals and universities, schools, landfills,
  crematoria, higher-tier buildings drawn bigger or with annexes,
  bus and metro lines, casinos, airports, levees, landmarks, floods, historic plaques,
  snow in winter and orange trees in autumn. Smoke drifts with the wind.
- **Replay or follow live.** Use the timeline to play, step or scrub. Marks on the
  timeline show council messages (purple), failed actions (amber), money trouble (red),
  disasters and offers (blue) and milestones (gold). A run in progress is followed
  automatically.
- **See why things happen.** The Agents tab shows each month's decisions with ✓/✗
  results. Click an action to fly the camera to where it happened. For LLM agents you
  can expand the model's full reasoning, its raw reply and the exact prompt it saw. A fleet
  turn also has a **Tool calls** panel: every call in order with its arguments, outcome (ok,
  failed, refused or repeat), cost and result, one line each (click one to expand it), under
  a header counting calls, actions, failures and how the turn ended. The
  Events tab starts with what residents complain about this month.
- **Inspect** any tile by clicking it: stats, services, wealth or industry type, its
  building level over time, and which agent zoned or built it.
- **Switch map layers** (press `V`, or `1`–`9`): pollution, land value, happiness, noise,
  power, water, sewage, river pollution, traffic, bus coverage, police/fire/clinic/school
  cover, crime, fire risk, health, garbage, soil pollution, historic buildings, business
  profit, affordability, natural resources, flood risk and districts.
- **Launch runs** with **＋ New run**: pick the map (one city, or two / three cities on one river), the
  seed, the length, how often agents act and the team: an AI fleet on one model, the scripted baselines,
  or a mix per role (fleet, scripted or the single-call `llm` agent, each on its own model). The run uses
  `configs/fleet_trinity.toml`'s fleet settings and playbooks. Other setups (remote agents, warm-ups,
  eval suites) stay on the command line.
- **Watch a run live.** While a game plays, a feed over the city shows whose turn it is and what that agent is
  doing: each model step, look-up, build and message with its outcome and cost, as it happens. The map and the
  left dock update after each build, not just once a month. Click a build to fly there; fold the feed from its
  header. Runs write it to `live.jsonl` (and the city mid-month to `live_frame.json`); `bench` runs don't.
- **Pause, resume or stop a live run** from the top bar. A pause lands before the next agent's turn
  (a fleet turn in progress finishes first), the game then plays on exactly as it would have, and the
  wait doesn't count in its wall time. Stop ends the game where it is, with no final score. The Runs tab
  also stops runs and holds the leaderboard.
- **Browse the Charts tab**: population, scores, money, happiness and approval, demand,
  jobs, pollution, commute and traffic, trade, service coverage, crime and health,
  education.
- **Region runs** get a city switch in the top bar and a **Region** tab: a map of the
  cities with their links and the goods, commuters and tourists flowing between them,
  plus each city's score and specialisations.

Press `?` in the app for all keyboard shortcuts. `civitas run ... --serve` also opens the
interface while a run from the terminal is playing. The old 2D viewer is still available
at `/classic.html`.

## Panels: Turns, Team, Problems and Compare

**Right panel** (viewer). Four tabs and a menu:
- **Turns:** a compact row per department for the quarter, with what it did, spent, ✓ / ✗, messages, refusals and a bad turn end. A row opens into its card: reasoning, actions with 📍, the messages it sent with whether they were answered, and its tool calls. Open rows stay open across quarters.
- **Team.**
- **Issues:** *Agent mistakes* (the Problems analysis) or *City events* (disasters, milestones, offers, and what residents complain about).
- **Money.**
- **⋯:** Charts, Region, All messages (the full council log), Runs, **Compare runs** and **Model evals** (see [Evaluating a model](EVALS.md)).

**Compare view** (viewer: ⋯ → Compare runs, the Runs tab's checkboxes and **Compare (n)**, or `C`). Up to 8 runs side by
side, e.g. the fleet's long runs, a region run, a late smoke and the scripted fleet. A table with the runs as columns
(best value per row in green): kind (LLM fleet, scripted fleet, scripted, LLM; region, late with its warm-up), model,
months played, the final and average city score (and after a warm-up the average since, and the score and population
change since hand-over), population, each department's final score, the Teamwork score with its four components as bars
and its requests, and from `fleet_report.json` the fleet calls per turn (failed, refused, repeats, tool errors) and
overrides. Above it, overlaid line charts of the city score, population and Teamwork per month, each run's scripted
warm-up shaded and its line faded there. A warm-up run with a scripted yardstick (see [Late-game smokes](FLEETS.md#late-game-smokes-and-the-yardstick)) shows it
beside its own numbers in the "since hand-over" and "average after warm-up" rows (`+2.3 · scripted +1.4`) and as a
dashed line of its colour from the hand-over in the city score and population charts (the legend's checkbox hides
it). A region run's own column holds the region's city score and population and its
departments' and Teamwork's mean over its cities; "Cities of region runs" adds a column (and a dashed line) per city.
The picked runs stay picked (localStorage); click a run's name to open it. Backend: `GET /api/compare?runs=a,b,c`
(`bench/compare.py:run_compare`, cached per run until its files change). The Runs / Evals switch above the table shows
model evals instead, one column per eval, and "Load eval…" adds an eval's runs (see [Evaluating a model](EVALS.md)).

**Team tab** (viewer). The departments as a live network (`viewer/collab.js`): the mayor in the centre, the departments on a ring, each node sized by its funds with an arc for its open requests. Edges glow with the last 12 months of messages and money. As the replay steps, each turn's events play out:
- requests fly as comets with the amount asked, and reports as small pulses;
- the mayor's money is a stream of gold coins, with "+$2,500" at the receiver;
- builds burst on the builder;
- a request's edge flashes green when it was answered, amber when partly, and red ✕ when nobody answered.

**Hand-off beams** (toolbar ⇢, key <kbd>B</kbd>, on by default) draw the land planning handed over on the city itself, in both the pixel and the 3D view. A hand-off is another department's build within 2 tiles of an earlier planning road, dezone or clearance, inside the window. When one happens, planning's road or lot glows, a light arcs from it to the new building, and the building bursts with "road → school". The hand-offs of the last window stay as arcs while you scrub, and the Team tab lists them (📍 flies there). The v23 seed-1 run has 72, most of them a road and then a school, power plant, pump or police station.

**Teamwork score.** A strip at the top of the tab shows the collaboration score (the fleet report's "Collaboration score", see [FLEETS.md](FLEETS.md#reports-and-the-collaboration-score)) as the run stood at the playhead: the overall score, the four components as small bars (n/a greyed out under 3 cases; hover one for its definition and counts, e.g. "95 of 165 requests answered"; money aim's says whether it is needs-based or message-based) and a sparkline over the months. A request counts once it is answered or its window has passed, so the score at month m is the one a live run showed then (one gap: a money request partly funded by then and met later counts as met). Each node reads its department's responsiveness as the one asked ("answers 59%", or "answers 2 of 2" under 3 cases), the left panel's Departments list ends with a Teamwork row (click it for the Team tab), and in the Region view a strip gives each city's score and, between the cities, the asks answered and deal offers accepted, with each city's score in the city table. The collab response carries it as `score` (`fleet/report.py:collab_score`: the report's numbers plus `series`, one entry per frame month; `region_score` for `?region=1`).

Hover a node or edge for its numbers, and click one to filter the thread list under the graph. Each thread is a request, its response (money, a reply, or an action near the tiles it named) and its outcome (the requester's next build). Click a thread's month to jump there, or its 📍 to fly to the building. ⤢ Expand opens the same view full-screen.

The data comes from `GET /api/runs/<id>/collab` (`bench/collab.py`), built from `frames.jsonl` alone, so it works for scripted, LLM and fleet runs and for runs already stored. A thread's window is 2 × `act_every` months. On the v23 120-month seed-1 run, 325 threads: 159 met, 63 partial, 50 replied, 33 unanswered, 12 acted, 8 open. A money request to the mayor is met or partly met by its top-ups to the sender, else replied to when the mayor writes back.

**Region runs** get a city / Region switch at the top of the Team tab. A city's view is its own team: a message to another city isn't one of its requests. Below the requests, a "With other cities" section lists the city's deals with its neighbours and its latest messages to other cities. The **Region** view draws the cities at their place in the region, sized by population, with the links they built drawn faintly underneath. Each mayor deal is an edge labelled with amount, resource and price (power yellow, water blue). An active deal is solid, with light flowing from seller to buyer and its payment travelling back as coins each month. An offer is dashed, and an ended deal stays faint for a year. As the replay steps, messages between cities fly as comets and an offer flies as a contract. An acceptance flashes the edge green and a cancellation flashes it red ✕. Below the graph:
- each city's tally: messages out and in, asks answered, deals proposed and accepted, sold and bought, and its teamwork score at the playhead;
- the deals, with their state at the playhead and what was paid;
- the threads between cities, newest first. A deal offer is accepted, declined or withdrawn. Another ask is acted on when the other mayor offers or accepts a deal with the asker inside the window, and replied to when the other city writes back.

Double-click a city, or click its row, to open its team. The data is `GET /api/runs/<id>/collab?region=1` (`bench/collab.py:region_collab`), built from each frame's `deals` snapshot, the deal actions in the turns and the council. `collab?city=C` adds that city's `deals`, `external` events and `external_threads`.

**Problems tab** (viewer). `GET /api/runs/<id>/problems` (`bench/problems.py`) lists every failed or refused tool call and every turn that ended badly (step limit, no tool call, token budget, unreadable reply), each with what the agent did next:
- **fixed:** a later call of the same tool worked in the same turn (the same arguments count too, after another action changed what failed, e.g. the dezone a road needed);
- **escalated:** it gave up but messaged another role, e.g. asked the mayor for money;
- **dropped:** it gave up;
- **repeated:** it sent the same failing call again;
- **heeded or overridden:** for a guard refusal, it changed course, or repeated the call with `confirm`;
- **recurred:** the same role hit the same cause on its next turn.

Hitting the per-turn action cap doesn't count. The tab shows these rates for the run and per role, the fleet report's coaching hints, and a filterable list; clicking an item opens that quarter's card with the call outlined. The timeline marks quarters with problems, red when one went unsolved. On the 120-month seed-0 run: 208 problems, 33% fixed in turn, 43% escalated, 24% unsolved, 58 recurred; planning had most of them.
