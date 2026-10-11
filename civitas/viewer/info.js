// Info bubbles: a small (i) button beside a metric opens a popover saying what it is, how it's computed and how to
// read it. INFO is the one source of those texts for the Evals page (evals.js) and the Compare view (compare.js), so
// a metric reads the same everywhere. infoBtn(key) renders the button; initInfo() (once, app.js) wires a single
// popover: hover with a mouse, focus with the keyboard, tap / click to pin it, Esc or a click elsewhere to close.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// key -> [title, text]: 1-3 plain sentences each (what it is, how it's computed, how to read it)
export const INFO = {
  // ------------------------------------------------ the headline numbers
  vs_scripted: ["vs scripted", "The run's city-score change since the hand-over (month 96) minus the change the scripted agents make from the same hand-over. Above 0: the model did better than the scripted baseline would have from that exact city. Subtracting scripted's change removes the seed's difficulty, which explains about 70% of the score's variance."],
  interval: ["95% interval", "The 95% t-interval of the mean vs scripted over the units (8 seeds on the v2 suites). An interval wholly above 0 beats scripted beyond seed luck; one that straddles 0 can't yet be told apart from it. The bar draws it, the dot is the mean and the thin line is 0."],
  units: ["Units", "A unit is one config and seed: one hand-over city. A unit's repeats are averaged first, so each seed counts once in the mean and its interval (a region run counts once, by its region score)."],
  pop_quality: ["Population / quality", "vs scripted split in two. Population: the gain in the population term, 50·√(population / target), minus scripted's gain. Quality: the rest (happiness, finance, utility services, safety, health, environment, employment, education, and any penalty). The two add up to vs scripted."],
  pop_term: ["Population term", "The part of vs scripted that came from population: the run's change in 50·√(population / target) minus the scripted agents' change from the same hand-over."],
  quality: ["Quality terms", "The part of vs scripted that didn't come from population: happiness, finance, utility services, safety, health, environment, employment and education (and any penalty), against scripted's change."],
  city_score: ["City score", "The city's score (0-100) at the end of the run, mean over units with its 95% interval. Half is population, 50·√(population / 30,000); half is quality (happiness, finance, services, environment, employment, safety, health, education), scaled by a population gate; bankruptcy halves it. It depends a lot on the seed, so models are ranked by vs scripted."],
  teamwork: ["Teamwork", "The report-only collaboration score (0-100): responsiveness to requests, follow-through on what was asked, hand-offs between departments (planning's lots used by the builders) and money aim (top-ups where they were needed). Mean over the runs; it doesn't enter the city score."],
  failed_turn: ["Failed actions per turn", "Game actions the game rejected per model turn: not enough money, a taken tile, no road access, a locked building. The agent reads the error in the result; lower is better."],
  refused_turn: ["Refused per turn", "Calls the tool box's guards refused per model turn before they reached the game: a top-up below the treasury reserve, a loan with money in the bank, a duplicate allocation, a take-back below a department's keep. The refusal says why; confirm: true overrides only on an exact repeat."],
  failed_refused: ["Failed / refused per turn", "Failed: game actions the game rejected (no money, a taken tile, no road access), per model turn. Refused: calls the tool box's guards stopped before the game (reserve, loan, duplicate, take-back), per model turn. Lower is better for both."],
  tokens: ["Tokens", "Prompt + completion tokens over every model call of the eval's runs (the warm-up is played by scripted agents and costs none). Each tool call re-sends the whole conversation as a new prompt, so prompt tokens dominate."],
  cost: ["Cost", "The eval's tokens priced by the suite's [eval] prices for this model (USD per million prompt and completion tokens); — when the suite has no price for it."],
  verdict: ["vs baseline", "Paired with the suite's baseline model on the same seeds: per seed, this model's vs scripted minus the baseline's, then the mean and its 95% interval. BETTER only when the whole interval is above 0, WORSE when it is below; otherwise there is no clear difference."],
  departments: ["Departments", "Each department's own final score (0-100) minus the scripted yardstick's from the same hand-over, mean over units: where a model gains or loses (e.g. a strong mayor, a weak planner)."],
  noop: ["Do-nothing change", "The city's score change since the hand-over when every model role does nothing (the noop baseline, from the same hand-over): what the city does on its own. Often below 0."],
  vs_noop: ["Points over doing nothing", "The model's city-score change since the hand-over minus the change when every model role does nothing from the same hand-over, with its 95% interval over the seeds. An absolute anchor: 0 = no better than idling. It ranks models exactly as vs scripted does (the two differ by the scripted agents' own gain over idling)."],
  normalized: ["Normalized score", "How much the model adds over doing nothing, as a multiple of what the scripted agents add: (model change − noop change) / (scripted change − noop change). 1.0× = as good as scripted, 0 = no better than idling, above 1 = better than scripted. Shown as the pooled ratio of the suite's means: per-seed ratios swing because scripted beats idling by only ~1 point in a 12-month window."],
  latest_eval: ["Latest eval", "This model's leaderboard row on the current environment version: vs scripted, mean over units with its 95% interval."],
  rank: ["Rank", "Models ordered by mean vs scripted on this environment version. Neighbours whose intervals overlap may swap places with more seeds."],
  // ------------------------------------------------ environment versions
  env_version: ["Environment version", "A frozen game: config, prompts, tools and the scripted fleet's play, recorded under a name (configs/envs/<name>.json and a git tag env/<name>). Only evals of the same version are ranked together."],
  fingerprint: ["Fingerprint", "A hash of what this code plays: config, prompts, tools and the scripted fleet's briefings, results and metrics on every suite seed. It matches the frozen version's while the game is unchanged."],
  drift: ["Drift", "Whether this code still plays the frozen version's game. Changed: evals run now are recorded as drifted and never ranked; freeze a new version to rank them (and re-run the models on it)."],
  tracks: ["Guided and open tracks", "Guided (civitas-v2): briefings name the next build, its exact call and the money to ask for. Open (civitas-v2-open): the same facts, prices and estimates, never a chosen action, so the model decides. Same seeds, game, tools and yardstick; each track has its own leaderboard."],
  // ------------------------------------------------ efficiency
  tokens_turn: ["Tokens per model turn", "Prompt + completion tokens per model turn (one department's quarter). A turn is a loop of model calls that each re-send the whole conversation, so tokens per turn ≈ model calls × context per call."],
  model_calls_turn: ["Model calls per turn", "API replies per model turn. Each call re-sends the whole conversation (system prompt, briefing, every tool result so far), so prompt tokens ≈ model calls × context: fewer calls is the main way to spend fewer tokens."],
  context_call: ["Context per call", "Prompt tokens per model call: how long the conversation is, on average, when the model is asked. It is similar for every model (about 6-7.6k), so tokens per turn differ mostly by the number of calls."],
  tool_calls_turn: ["Tool calls per turn", "Tool calls per model turn: game actions, look-ups and the turn controls (end_turn, send_message, update_memory, update_plan)."],
  actions_turn: ["Actions per turn", "Game actions per model turn (build, zone, allocate_budget, ...), whether they succeeded or not."],
  lookups_turn: ["Look-ups per turn", "Read-only tool calls per model turn (map_overview, find_sites, inspect, ...): every tool but the game actions and the turn controls. The briefing already holds most facts, so many look-ups mostly cost tokens."],
  capped: ["Step cap", "The share of model turns that used every step the environment allows (max_steps, 20): those turns ended at the cap rather than with the model's own end_turn."],
  per_10m: ["vs scripted per 10M tokens", "vs scripted divided by the tokens one run's model turns use, per 10 million tokens: score per token. A model can sit lower on vs scripted and higher here."],
  scatter: ["Cost against score", "One dot per model: tokens per model turn (x) against vs scripted (y), the vertical bar its 95% interval. Up and to the left is better and cheaper; the dashed line joins the Pareto front, the models no other model beats on both."],
  // ------------------------------------------------ eval scorecards and the Compare view
  beats_scripted: ["Beats scripted", "Units whose mean vs scripted is above 0, of all units."],
  seed_spread: ["Seed spread", "The standard deviation of vs scripted across units: how much the result moves from one seed to another."],
  replay_noise: ["Replay noise", "How much vs scripted varies between runs of the same model on the same seed (pooled standard deviation within units). Needs 2+ repeats per seed."],
  mde: ["Detectable difference", "The smallest gap to another eval on these units that a paired comparison finds 80% of the time, from the replay noise alone (a lower bound)."],
  city_change: ["City change since hand-over", "The model's own city-score change from the hand-over to the end of the run (mean): before subtracting scripted's."],
  scripted_change: ["Scripted change", "The scripted agents' city-score change from the same hand-over (the run's yardstick.json, mean): what vs scripted subtracts."],
  population_vs: ["Population vs scripted", "The run's population gain since the hand-over minus the scripted agents' gain from the same hand-over, mean over units with its 95% interval."],
  runs_scored: ["Runs scored", "Runs finished with a scripted yardstick to compare with, of the eval's planned runs."],
  units_repeats: ["Units × repeats", "Units (config + seed) and how many runs each has. Repeats replay the same hand-over and measure replay noise; more seeds narrow the interval more than repeats."],
  paired_first: ["Paired with the first column", "This eval's vs scripted against the first column's, paired by config and seed: the mean difference with its 95% interval over the shared units. Better / worse only when the interval misses 0."],
  model_turns: ["Model turns", "Department turns played by the model after the hand-over (32 per single-city run on the v2 suites: 8 departments × 4 quarters)."],
  run_time: ["Run time", "Wall-clock time of the runs, summed; they run several at a time, so the eval took less."],
  handover_city: ["City at hand-over", "The city score when the scripted warm-up ended and the model took over (month 96). It is the same for every model on a seed."],
  final_city: ["Final city score", "The city score (0-100) in the last month played."],
  avg_city: ["Average city score", "The mean of the monthly city score over every month played, warm-up included."],
  avg_after_warmup: ["Average after warm-up", "The mean city score over the configured agents' months, after the hand-over. Scripted: the yardstick's over the same months."],
  city_since_handover: ["City score since hand-over", "Final city score minus the score when the warm-up ended. Scripted: the yardstick's change from the same hand-over; the difference is vs scripted."],
  population: ["Population", "Residents in the last month played."],
  population_since_handover: ["Population since hand-over", "Final population minus the population at the hand-over. Scripted: the yardstick's gain from the same start."],
  warmup: ["Scripted warm-up", "Months 0 to N-1 are played by the scripted fleet before the configured agents take over the grown city."],
  bankrupt: ["Bankrupt", "Whether the treasury went below zero: it halves the city score and ends the game."],
  requests: ["Requests", "Requests between departments (the Team tab's threads)."],
  answers: ["Answers", "Teamwork's responsiveness: the share of requests answered (a top-up or a reply) within two turns."],
  follow_through: ["Follow-through", "Teamwork's follow-through: the share of promised or funded builds that were actually built."],
  handoffs: ["Hand-offs", "Teamwork's hand-offs: the share of planning's roads and lots that another department went on to build on."],
  money_aim: ["Money aim", "Teamwork's money aim: the share of the mayor's top-ups that went to a first building, supply or growth, or answered a money request."],
  asks_between: ["Asks between cities", "Share of asks to another city answered (n/a under the minimum cases)."],
  deals_accepted: ["Deals accepted", "Share of power / water deal offers between cities accepted."],
  fleet_turns: ["Turns", "Fleet turns played by model-backed agents."],
  repeats_turn: ["Repeats per turn", "A failed call sent again unchanged, refused by the tool box, per turn."],
  tool_errors_turn: ["Tool errors per turn", "Calls with bad arguments or an unknown tool, per turn."],
  overrides: ["Overrides", "Guard refusals overridden by repeating the exact call with confirm: true."],
  playing_time: ["Playing time", "Wall-clock time the run took."],
  // ------------------------------------------------ the Cities view
  cities: ["Cities", "Each model's city at the end of a run (month 108), drawn by the viewer's pixel renderer from the run's last frame. 'Best run per model' shows each model's run with the highest vs scripted, so seeds and terrain differ; pick a seed to see every model on the same hand-over city."],
  handover_pic: ["The hand-over city", "The city at month 96, built by the scripted fleet in the warm-up: every model on this seed started from it. The scripted yardstick plays on from here but isn't logged as frames, so its final city can't be drawn."],
};

export const infoText = (key) => (INFO[key] || [key, ""])[1];

// the (i) button: `key` names an INFO entry; a key not in INFO is shown as the text itself (with `title` as heading)
export function infoBtn(key, title = null) {
  if (!key) return "";
  const known = INFO[key];
  const head = title || (known ? known[0] : "About this");
  const attr = known ? `data-info="${esc(key)}"` : `data-info-text="${esc(key)}"`;
  return `<button type="button" class="info-i" ${attr} data-info-title="${esc(head)}" aria-label="What is ${esc(head)}?" aria-expanded="false">i</button>`;
}

// ------------------------------------------------ the popover (one for the page)
let pop = null, anchor = null, pinned = false, watch = null;
function place() {
  if (!pop || !anchor) return;
  const r = anchor.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight, m = 12;
  pop.style.maxWidth = `${Math.min(340, vw - 2 * m)}px`;
  pop.style.left = "0px"; pop.style.top = "0px";
  const w = pop.offsetWidth, h = pop.offsetHeight;
  const left = Math.max(m, Math.min(r.left + r.width / 2 - w / 2, vw - w - m));
  const below = r.bottom + 8, above = r.top - 8 - h;
  const top = below + h <= vh - m || above < m ? Math.min(below, vh - h - m) : above;
  pop.style.left = `${Math.round(left)}px`; pop.style.top = `${Math.round(Math.max(m, top))}px`;
  pop.classList.toggle("above", top < r.top);
}
export function showInfo(btn, pin = false) {
  if (!pop) return;
  if (anchor && anchor !== btn) anchor.setAttribute("aria-expanded", "false");
  anchor = btn; pinned = pin;
  const key = btn.dataset.info, entry = key ? INFO[key] : null;
  const title = btn.dataset.infoTitle || (entry ? entry[0] : ""), text = entry ? entry[1] : (btn.dataset.infoText || "");
  pop.innerHTML = `<b>${esc(title)}</b><div>${esc(text)}</div>`;
  pop.hidden = false;
  btn.setAttribute("aria-expanded", "true"); btn.setAttribute("aria-describedby", "infoPop");
  place();
  clearInterval(watch);
  watch = setInterval(() => { if (!anchor || !anchor.isConnected) hideInfo(); }, 400);   // its table was redrawn
}
export function hideInfo() {
  if (!pop) return;
  pop.hidden = true; pinned = false; clearInterval(watch);
  if (anchor) { anchor.setAttribute("aria-expanded", "false"); anchor.removeAttribute("aria-describedby"); }
  anchor = null;
}
export function initInfo() {
  if (pop) return;
  pop = document.createElement("div");
  pop.id = "infoPop"; pop.className = "info-pop"; pop.setAttribute("role", "tooltip"); pop.hidden = true;
  document.body.appendChild(pop);
  const btnOf = (e) => e.target && e.target.closest ? e.target.closest(".info-i") : null;
  document.addEventListener("pointerover", (e) => { const b = btnOf(e); if (b && e.pointerType === "mouse" && !pinned) showInfo(b); });
  document.addEventListener("pointerout", (e) => {
    const b = btnOf(e);
    if (b && b === anchor && e.pointerType === "mouse" && !pinned && document.activeElement !== b) hideInfo();
  });
  document.addEventListener("click", (e) => {
    const b = btnOf(e);
    if (b) { e.preventDefault(); e.stopPropagation(); if (b === anchor && pinned) hideInfo(); else showInfo(b, true); return; }
    if (anchor && !pop.contains(e.target)) hideInfo();
  }, true);
  document.addEventListener("focusin", (e) => { const b = btnOf(e); if (b && b !== anchor) showInfo(b); });
  document.addEventListener("focusout", (e) => { const b = btnOf(e); if (b && b === anchor && !pinned) hideInfo(); });
  document.addEventListener("keydown", (e) => {   // before the modals' own Esc: close the bubble, keep the modal
    if (e.key === "Escape" && anchor) { e.stopPropagation(); e.preventDefault(); hideInfo(); }
  }, true);
  window.addEventListener("resize", () => anchor && place());
  document.addEventListener("scroll", () => { if (anchor) { if (pinned) place(); else hideInfo(); } }, true);
}
