# Collaboration view: design

Date: 2026-09-30. Goal: make the agents' collaboration **visible and fun to watch** while a run replays, with the
detail behind it one click away. Step 1 is a live department network (this doc). Step 2, later, draws the land and road
hand-offs as beams on the city map.

## What we show

Three kinds of interaction, all derived from what every run already records in `frames.jsonl` (council messages, and
each turn's action results with their arguments). No new runs and no sim changes; it works for scripted, LLM and fleet
runs alike.

- **Messages**: requests and reports between departments (`frame.council`).
- **Money**: the mayor's top-ups and take-backs (`allocate_budget` results in `frame.turns`).
- **Builds**: a department's successful build, road, zone or upgrade (its outcome).

A **thread** links them: a request, then a response, then an outcome. For example: health asks the mayor for $2,000
for a landfill (month 9), the mayor funds $2,000 (month 12), and health builds the landfill at (21,30) (month 12).

## Backend: `civitas/bench/collab.py`, `GET /api/runs/<id>/collab`

`run_collab(frames_path) -> dict`, cached on the file's size like `/problems`:

- `events`: `{id, month, kind: message|money|build, from, to, text, amount, tile, thread, ask}` in month order.
- `threads`: `{id, month, from, to, kind: money|land|other, amount, text, status, response, outcome}`.
- `roles`, `act_every`, `last_month`, and per-role `stats` (asked, answered, asked of, answered by, money received).

How threads are matched. The window is `2 × act_every` months after the request; a request sent at month m can only be
answered after m, because the mayor acts first in each round.

- **A request** is a message to one role whose text asks for something ($ amount, need, request, please, can you, ...).
  Messages to `all` are broadcasts and start no thread.
- **Money request** (to the mayor, naming dollars or funds): answered by the mayor's positive `allocate_budget` to the
  sender inside the window. Status: `met` when the allocations reach 80% of the amount asked (any amount when none is
  named), `partial` below that, `unanswered` when nothing came, and `open` while the window hasn't passed yet (a live
  run).
- **Other requests** (to planning for a lot or road, or between departments): `acted` when the recipient's successful
  action falls within 3 tiles of a tile named in the request, `replied` when the recipient messages the sender back
  inside the window, else `unanswered` or `open`. `land` when the request names tiles or roads, lots or zones.
- **Outcome**: the requester's first successful build after the response and inside the window.

## Frontend: `viewer/collab.js` (`CollabGraph`), a **Team** tab and a full-screen **Team view**

- **Layout**: the mayor in the centre and the seven departments on a ring, each a glowing disc in its role colour (the
  viewer's `ROLE_C`). The disc's size follows the department's funds, a label shows its funds, and an arc round it
  fills with its open requests.
- **Edges**: a curved edge for each pair that interacted. Width and brightness follow recent activity (messages plus
  money, decaying month by month), with a gradient from sender to receiver.
- **Animation**: stepping or playing into a month launches that month's events:
  - a request is a comet with a trail from sender to receiver;
  - a report is a smaller, dimmer pulse;
  - money is a stream of gold coins, sized by amount, with a floating "+$2,500" at the receiver;
  - a build is a burst ring on the builder, with a short label ("built clinic");
  - a thread closing flashes its edge: green for met or acted, amber for partial, a red fade for unanswered.

  Durations scale with the playback speed. The loop runs only while something moves.
- **Paused or scrubbing**: a static picture of the last 12 months (edges, open-request arcs), so any month reads at a
  glance.
- **Interaction**: hovering a node or edge shows its numbers. Clicking a node or edge filters the thread list to it.
  Clicking a thread jumps to its month and flies the camera to its outcome tile. The existing role chips filter the
  graph too.
- **Team tab**: the graph on top, then a line for the quarter (requests, answered share, money moved), then the
  quarter's threads with status chips. The ⛶ button opens the same graph large in a modal, like the Region map.

## Testing

- `tests/test_collab.py`: synthetic frames for each status (met, partial, unanswered, open, replied, acted) and for
  outcomes, a short real scripted episode run through `run_collab` (valid JSON, events in month order), and the
  server endpoint.
- The view: a Playwright screenshot of the Team tab on a stored run, served on a spare port.

## Later (not in step 1)

Map beams for land hand-offs (road or lot, then the building), region runs (per-city threads and deals), and a
collaboration score in the fleet report.
