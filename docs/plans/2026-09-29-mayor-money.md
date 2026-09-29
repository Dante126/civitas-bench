# Mayor money: implementation plan (base: "Money picture first", with grafts)

Base commit is main at 963b91d. It contains fd5112f, and the fleet money code is unchanged since then. Every change is in `civitas/fleet/`, the two fleet configs, README and tests. Nothing in `civitas/core`, `civitas/sim`, `civitas/agents` or `bench/scoring` changes.

## 1. Root causes

- **The to-do line is built from plan targets, not needs.** `funding_gaps()` counts only target gaps, and the month-0 targets stay: education was $0 in 15 of 15 plans, and utilities stayed at ≥$7,000 on 40 of 45 m3/6/9 turns.
  - First buildings at a $0 target were funded 16/36 times, against 36/40 with a target.
  - 24 service first-building department-turns ended unfunded (14 turns). 20 of them were at m3, with $8.0–17.4k above the reserve.
  - Utilities got $113k of its $135k top-ups beyond what its needs line asked.
- **The mayor can't see running costs or its own cash flow.** Needs lines give the purchase price only. The Finances treasury change mixes in the mayor's own transfers: 30 of 81 turns showed a "loss" while operations ran a surplus. Interest never appears in dollars.
  - "Another" top-ups had a median of -$50 above the reserve at the call, and 32 of 48 left the treasury under it.
  - In the long run, service upkeep was a median 34% of income. At m48 the city had 22 service buildings against the scripted fleet's 7, and 13 of 31 builds added ≤3 coverage points.
- **Block order becomes funding order, because the step-0 plan is lost.** `agent.py` replays content and tool calls to later steps, not reasoning, and 234 of 338 allocation messages had under 40 characters of content.
  - 32 of 54 turns followed strict block order (concordance 0.77), and education was topped up last in 11 turns.
  - The treasury ran dry mid-turn in 10 of 56 smoke turns. 8 of the 25 failures were blind, drained by an earlier call in the same batch.
  - At v19 s0 m9, planning's $315 road came tenth.
- **The mayor triggers its own loans and never repays.**
  - 10 of 15 smoke loans ($34.8k of $60.0k) came in turns that started at ≥$3,000, and 14 of 15 were taken after the mayor's own top-ups had emptied the treasury.
  - There were 0 repay_loan calls in 96 turns.
  - In the long run, 11 loans ($50k) came, all in hold turns. Debt sat at the cap from m45 to m120, and interest came to $23.9k.
  - "Hold" showed in 35 of 40 turns: the reserve (3–6× expenses) was never reachable and the line gives no exit. $108.7k was still allocated in hold turns.
  - The reserve doubled on a $0.70 deficit (v16 s1 m6).
- **`confirm: true` is a free pass, and the keep is invisible.**
  - All 3 v19 take-back refusals were overridden at once, and 3 of 7 duplicate refusals were overridden.
  - The duplicate guard ignores negative amounts, so v19 s1 m6 repeated its take-back pairs.
  - 12 take-backs went below keep ($17.7k), and planning was emptied in 7 turns.
  - In the long run, planning was at ≤$100 at 33 of 40 turns, and 23 of its builds failed on budget.

## 2. Changes

**Merge order:** A first (define its API before starting B, C and D), then B, C and D in parallel, then E. Unit ownership is by function, so no two units edit the same function.

### API contract (Unit A publishes; B, C and D code against it)

```python
Price.upkeep: float; Need.upkeep: float
DeptMoney.kind -> "first/supply"|"growth"|"another"|"blocked"|"nothing"|"idle"
DeptMoney.top_up: int; DeptMoney.upkeep: float; DeptMoney.state: str; DeptMoney.keep_why: str
Funding.order: tuple[str]; Funding.to_fund: list[FundItem(dept, kind, amount, upkeep, what)]
Funding.spendable: float; Funding.unclaimed: dict; Funding.firsts_at_zero: tuple[str]
reserve(w); reserve_why(w) -> str; cash_flow(w) -> CashFlow
spend_lines(w, f) -> [to_spend_line, to_fund_line]; still_to_fund(f, n=4) -> str
```

### Unit A: money model (`civitas/fleet/tools.py`, module level, no ToolBox methods)

**A1. Upkeep on prices and needs** (root cause 2)
- Add `Price.upkeep` and a new `upkeep_of(w, dept, name)`. It reads values at call time, so `TIERS.*` overrides reach it:
  - tier spec `C.TIER_LINES[line][tier-1]["upkeep"]`;
  - off-tier specs: their spec's `"upkeep"`;
  - `bus_line`: `C.BUS_LINE_UPKEEP`;
  - `levee_tile`: `CATALOG["levee"]["upkeep"]` per tile;
  - `avenue_tile`: the `ROAD_UPKEEP` difference per tile.
- Multiply by the same funding factor `World.dept_upkeep` applies.
- `site_price()` fills `Price.upkeep`, and `first_build()` fills `Need.upkeep`. The road Need gets 0.
- `Need.text()` puts the upkeep inside the parentheses:
  - "its first police station (5 sites, $55/month)"
  - "another clinic (coverage 21%, $40/month)"
  - The road text is unchanged, and an upkeep of 0 adds nothing.

**A2. DeptMoney properties** (root causes 1 and 5)
- `kind`:
  - `"idle"` if the department has idle money;
  - otherwise, if it has a needs line: `"growth"` for planning; `"first/supply"` if any named Need is `.first`, or it is utilities (first_build names a utilities building only while `_wanted`), or it is a levee tile; `"another"` otherwise;
  - otherwise `"blocked"` if it is blocked or in `unclaimed`;
  - otherwise `"nothing"`.
- `top_up = ceil(needs.cost - funds)`.
- `upkeep` = the largest `Need.upkeep` among the named needs.
- `state`, used only for a line with nothing else to show, and never for planning:
  - "funds cover its builds" when funds ≥ bound;
  - "funds cover its next build" when funds ≥ the build's cost;
  - otherwise "nothing ready to buy".
- `keep_why`: whichever of "2 x its $658 spending", "its dearest first building" or "the $1,000 minimum" set `keep`.

**A3. Funding fields** (root causes 1 and 3)
- `funding_gaps()` also computes:
  - `unclaimed` (moved here from the briefing, so the order is the same everywhere);
  - `order`, sorted by kind rank (first/supply 0, growth 1, another 2, blocked 3, nothing 4, idle 5) with turn order breaking ties;
  - `to_fund`, the departments with `top_up > 0` in that order;
  - `spendable = spare + Σidle`;
  - `firsts_at_zero`, the first/supply departments whose plan target is 0 or missing.
- `gaps` and `total` keep their meaning (the "<" marker and update_plan's `below_target`).
- `hold` becomes `spendable <= 0`.
- `unclaimed_firsts` keeps turn order, so site claims don't change.

**A4. Reserve and cash flow** (root causes 2 and 4)
- `reserve()` tests for a deficit on rounded dollars: `round(income) >= round(expenses)` gives 3×. This is the graft; `agents/scripted.py` is untouched.
- `reserve_why(w)` returns "" at the minimum, " (3 x $1,237 expenses)", or " (6 x $676 expenses: last month's deficit was $2)".
- `cash_flow(w)` covers the same window as `overview()`'s trend (entries after `past`). It returns:
  - `ops = Σ(income - expenses)`, `top_ups = Σ dept_allocated`, `loans = Δdebt`;
  - `other = Δtreasury - ops + top_ups - loans`, which is $0 without rewards or offer bonuses;
  - ops per month;
  - last month's income and expenses, buildings (`expense_breakdown ∩ dept_upkeep`), city services (the rest), and interest (`loan_interest`).

**A5. `spend_lines(w, f)`** (root causes 1, 3 and 4)

The `to spend` line:
- Above the reserve: "- to spend: $2,698 above the $3,000 reserve[ + $1,508 idle to take back = $4,206]; operations +$233/month"
- Positive only because of idle money: "- to spend: $1,200: $1,508 idle to take back, less the $308 the treasury lacks for the $3,711 reserve (3 x $1,237 expenses); operations …"
- Hold: "- to spend: none (hold): the treasury is $1,723 under the $3,711 reserve (3 x $1,237 expenses); operations +$381/month close that in 5 months; first/supply and growth may use $X (treasury above $3,000 + idle)". The last clause reads "nothing above $3,000" when X = 0. The "close that" clause is left out when operations are not positive.

The `to fund` line:
- "- to fund: first/supply health $2,000, education $2,500 ($4,500, +$85/month) | growth planning $315 | another police $2,500, fire $2,500 ($5,000, +$110/month): $4,206 covers health, planning; the other $9,500 would go under the reserve"
- The cover clause is **greedy** in list order: an item that doesn't fit is skipped, and later cheaper items can still fit. In a hold it covers only first/supply and growth, against X.
- With nothing listed: "- to fund: nothing ready".
- `still_to_fund(f)` gives "education $2,500 (first/supply), fire $2,500 (another)", at most 4 entries, then "+N more", or "none".
- `HOLD` is removed. Its users switch to `spend_lines`.

### Unit B: briefing (`civitas/fleet/briefing.py`: `_mayor`, `_dept_line`, constants)

**B1. Finances block** (root causes 2 and 4)
- "- treasury $7,922 (-$12.3k since month 3: operations +$977, top-ups -$13.3k), debt $0". Zero parts are left out, and "loans +$X" and "other +$5,000" appear only when nonzero.
- "- last month: income $597, expenses $561: buildings $312 (utilities $82, roads $75, police $55, 2 more $100), city services $157, interest $0"
- When debt > 0: "- debt $18.8k: $94/month interest, $31.2k more can be borrowed; each $1,000 repaid saves $5/month". The first line then drops ", debt $X".
- The WARNING line is unchanged.

**B2. Departments block** (root causes 1, 3 and 5)
- Header: "## Departments (most urgent first; spent = last 3 months[; cut reports: history kind=report])".
- Lines follow `f.order`. Each line's suffix is one of:
  - "; idle $1,508 (keeps $1,000)"
  - the needs suffix, using A1's text
  - "; blocked: …"
  - "; " + `state`
- Drop the "; priority: …" suffix. It costs about 385 characters, and the size bound can't carry it.
- The City plan header becomes "(v2, month 3; priorities and full text: get_plan)".
- Mayor report previews are cut at `MAYOR_REPORT_CHARS = 180`.
- `BLOCKED_NOTE = "- blocked for lack of land: money buys nothing there until planning frees lots (ask planning)"`.
- Remove `_firsts_line`, `FIRSTS_LINE_CHARS`, the "N below target" line and the idle-note line. Append `spend_lines(w, f)` in their place.
- Example v19 s1 m3 (today: no service funded, $8.3k left): "first/supply police $2,500, fire $2,500, health $2,000, education $2,500 ($9,500, +$195/month): $17.4k covers police, fire, health, education".

**B3. System prompt text.** `BUDGET["mayor"]`: "Stay solvent: upkeep is paid from the treasury every month and every building adds upkeep for good; keep the reserve when you move money to departments (allocate_budget) or cut taxes."

### Unit C: guards (`tools.py`: `guarded`, `CONFIRM`, `ToolBox.__init__`, `_act`, `_duplicate_refusal`, `_allocation_refusal`, `_take_back_refusal`, new `_loan_refusal`; plus `scripted.py._actions`, `report.py`, `agent._slim`)

**C1. Earned confirm** (root cause 5)
- `__init__` sets `self._refused = set()` and `self.start_treasury = w.treasury`.
- In `_act`, for the mayor's `allocate_budget` and `take_loan`, the guard always runs. The key is `(name, dept, amount)`.
- A refusal is overridden only when `confirm` is true **and** the key is already in `_refused`. Otherwise the key is added and the call is refused. If confirm was sent, the refusal adds " (confirm: true counts only when repeating this exact call after it was refused)".
- An override runs and returns `"override": kind`.
- The bulldoze/demolish guards are unchanged.
- New `CONFIRM` texts:
  - allocate_budget: "true only to repeat an allocate_budget refused this turn with the same department and amount, once you are sure; on any other call it doesn't count"
  - take_loan: the same wording.

**C2. Duplicate guard on take-backs** (graft; root cause 5). An identical negative allocation is also a duplicate: "you already took $1,140 back from planning this turn (it now has $1,240); take-backs add up". In `_allocation_refusal` the take-back guard runs first, then the duplicate guard.

**C3. Take-back refusal text.** "that takes planning below the $1,316 it keeps (2 x its $658 spending), so it can't build next quarter[; you already took back $1,140 from it this turn]; {free}: {todo}. To take it anyway, repeat this exact call with confirm: true."

**C4. Soft loan guard** (graft; root cause 4)
- `guarded("mayor", "take_loan")` returns True, so the schema gains a `confirm` parameter.
- `_loan_refusal`, kind `"loan"`, refuses in two cases:
  - (a) `w.treasury >= RESERVE_MIN`: "the treasury holds $3,598, over the $3,000 minimum: a $5,000 loan costs $25/month until repaid (debt would be $5,000 of $50,000); borrow only for a first building or supply need the treasury can't cover. To borrow anyway, repeat this exact call with confirm: true."
  - (b) `start_treasury >= RESERVE_MIN` and this turn's net successful allocations are > 0: "the treasury started this turn at $5,698 and this turn's top-ups ($5,698 net) took it under $3,000: take back a top-up that isn't a first building or supply (free) instead of a $3,000 loan at $15/month. To borrow anyway, repeat this exact call with confirm: true."
- Replayed on the recorded calls, it meets 10 of 15 smoke loans ($34.8k) and 1 of 11 long-run loans (m21).
- The scripted mayor borrows only at turn start with the treasury under $3,000, so it never meets this guard.

**C5. `scripted._actions`.** The confirm tuple becomes `("duplicate", "take_back", "loan")`, and the docstring is updated.

**C6. Report and trace fields.**
- `agent._slim` keeps `"override"`.
- `report.turn_row` adds `"overrides": marked("override")`, which is added to the role totals.
- fleet_report.md shows overrides next to refused.

**C7. Hook in `_act`.** For the mayor's allocate_budget, take_loan and repay_loan, append `self._money_note(name, args, res)` to the model message on success and on core failure, instead of `_explain`. Guard refusals get no note. `ToolBox.__init__` appends `MONEY_DOCS.get(a, "")` to the action descriptions. Until D lands, stub both.

### Unit D: result notes and look-ups (`tools.py`: new `ToolBox._money_note`, `MONEY_DOCS`, `update_plan`, `overview`)

**D1. `_money_note`** (root cause 3; model only, while `executed`, `fleet.log` and history keep the core text). It recomputes `funding_gaps`, which takes about 5 ms.

| Case | Note |
|---|---|
| top-up OK | " \| police: can pay for another police station (+$55/month)" or "still $1,000 short of its first police station: less buys nothing" or "had nothing ready to buy" or "blocked for lack of land: this buys nothing until planning frees a lot"; then "; treasury $2,788, $212 under the $3,000 reserve; still to fund: …" |
| take-back OK | " \| planning now $1,240, what it keeps (nothing idle)" or "$508 over what it keeps" or "$1,316 below what it keeps (2 x its $658 spending): it can't build next quarter"; then treasury and still-to-fund |
| take_loan OK | " \| interest now $15/month (+$15); $47k more can be borrowed; treasury …; still to fund …" |
| take_loan at the cap | " \| no more credit: repay_loan from money above the reserve cuts the $250/month interest" (graft) |
| repay_loan OK | " \| interest now $225/month (saves $25/month); treasury …" |
| "treasury only has" | " \| treasury $500, $2,500 under the $3,000 reserve; idle to take back: planning $1,508; still to fund: …" (graft: ledger on core failures) |

**D2. `MONEY_DOCS`.**
- take_loan: " Interest is $5 a month per $1,000 until repaid; the finance score reads net worth (treasury + department funds - debt)."
- repay_loan: " Each $1,000 repaid saves $5 a month in interest."

**D3. `update_plan` result.** It keeps `idle` and `below_target` and adds:
- `to_spend` (rounded `spendable`);
- `to_fund`, as a list of `{department, kind, amount, upkeep_per_month}`;
- `first_buildings`, from `firsts_at_zero` (graft).

`next` becomes `"; ".join(spend_lines)`, plus "first buildings with a $0 target: education" when there are any.

**D4. `overview()`.** Add `operations_3_months` (the `cash_flow` split) and `costs_last_month` (`expense_breakdown`, rounded).

### Unit E: playbook and README (`configs/fleet_smoke.toml`, `configs/fleet_trinity.toml`, which must stay identical; `README.md`)

**E1. Mayor playbook.** Replace the first-turn, every-turn, reserve and loans bullets, and add a refusal bullet:
- **First turn:** "… planning $3,000 (…) and utilities $7,000 for its first plants. From month 3, fund every first building a needs line names ("its first …"; police, fire, health, education: $2,000-2,500 each; less buys nothing), whatever its target; transport $0 until ~1,500 residents. After month 0 give utilities what its needs line asks, not its target. The rest stays in the treasury: every building adds upkeep for good (its needs line says how much)."
- **Every turn:** "Every turn: read the reports and the Departments block (most urgent first). First take back the idle money the 'to spend' line counts (at most the idle a line shows), before any top-up or loan; then fund the 'to fund' line in its order and stop where the money to spend runs out. A department with no needs line has nothing ready to buy, whatever its target. Allocate once per department per turn, one call at a time when money is short: each result shows the treasury and what is still to fund."
- **Reserve:** "Reserve: keep at least $3,000 or 3 months of expenses in the treasury (6 after a deficit month). In a hold fund only first buildings, supply and planning's road, from the treasury above $3,000 and idle money; the 'to spend' line says when operations close the gap."
- **Loans:** "Loans only for a first building or supply need the money to spend can't cover, and only if operations stay positive after its upkeep and the interest (Finances shows both); never for another building, and never after this turn's own top-ups emptied the treasury (take one back instead). With debt and no first building or supply need waiting, repay from the money above the reserve (each $1,000 repaid saves $5 a month)."
- **Refusals:** "A refused call says why and what to do instead: do that. confirm: true counts only on a repeat of the exact refused call, and only for a reason the refusal didn't weigh."

**E2. README "Agent fleets".** Describe:
- the Finances split, the need-ordered block, the to spend / to fund lines and the result notes;
- earned confirm, duplicate take-backs and the loan guard;
- the overrides column.

State that the scripted baseline (77.5 ± 1.5), fleet_scripted (68.8 ± 2.0, about 7.3k residents) and region numbers are unchanged. Add a smoke v20 paragraph after it is run.

## 3. What stays the mayor's decision

- **Money choices.** Which departments get money, how much, in what order, what to take back, whether to borrow or repay, taxes, funding levels, policies and plan targets. No amount is chosen for the mayor, nothing is auto-allocated, and neither the reserve nor the hold is enforced.
- **What the harness adds.** Only arithmetic already implied by the needs lines: the need class, the order, top-up gaps, upkeep, a greedy cover sum, cash-flow splits and interest. The class ranking is shown, not imposed. A mayor can fund a second station ahead of a first school, or drain the treasury, and it will score accordingly.
- **Guards.** They stay soft: take-back, duplicate and now loan. Every override is still possible; it only has to be earned by repeating the exact refused call. Every refusal and override is counted in the report.

## 4. Tests

**Unit A**
- `test_needs_lines_carry_upkeep`:
  - station $55, clinic $40, school $45, landfill $30;
  - an override of `TIERS.police_station.1.upkeep` flows through, and 50% funding halves the figure;
  - the road shows no upkeep.
- `test_need_kinds_and_order`: first_build is monkeypatched with utilities supply-busy, education first, police another, a planning road, fire blocked, health idle. The order is utilities, education, planning, police, fire, transport, health, with turn order within a kind.
- `test_to_fund_is_needs_based_not_target_based`:
  - education at target 0 with a first school is listed and appears in `firsts_at_zero`;
  - utilities with target $7,000, funds $800 and a $1,200 outlet is listed at $400.
- `test_cover_clause_is_greedy`: a $5,000 first high school that doesn't fit doesn't hide a $315 road after it. Also check the hold variant (first/supply and growth only, against treasury above $3,000 plus idle), and that "close that in N months" appears only when operations are positive.
- `test_reserve_rounds_the_deficit`:
  - (675.7, 676.4) gives 3×, and (675, 677) gives 6×;
  - `reserve_why` strings;
  - update `test_reserve_follows_the_mayor_playbook`.
- `test_cash_flow_identity`: over months with top-ups, a take-back, a loan and a repay, operations − top-ups + loans + other equals the change, with other = 0. A Town reward shows "other +$5,000".

**Unit B**
- `test_briefings_stay_bounded_at_the_limits`:
  - the bound stays at 7,700;
  - add a mix with debt $50,000, a 6× deficit, a 7-department cost breakdown and a partial greedy cover;
  - assert "- to spend:" and "- to fund:" are present and "; priority: ", "first buildings still unfunded" and "below target (" are absent;
  - replace the firsts-line and idle-note asserts with "(keeps $…)".
- Prototype sizes to beat: HEAD's worst mix 7,671 becomes 7,357, and the realistic worst is 7,557 (design/measure.py, rerun here). With the hold dip clause and the debt repay hint the budget is ≤7,650. If it goes over, name the top 2 departments in the cost breakdown instead of 3.
- `test_mayor_finances_lines`: the split, the costs line (top 3 then "N more") and the debt line appear only with debt, and interest is 0.5% of the debt.
- Rewrite `test_first_buildings_come_before_second_ones` and `test_mayor_briefing_marks_departments_below_their_target` against the to-fund line.

**Unit C**
- `test_confirm_counts_only_after_a_refusal`:
  - a below-keep take-back sent first with confirm is refused and carries the note;
  - its identical repeat with confirm passes with `override == "take_back"`;
  - a within-keep take-back with confirm passes at once.
- `test_identical_take_backs_are_duplicates`: update `test_other_allocations_pass_the_duplicate_guard`.
- `test_a_self_inflicted_loan_is_refused_until_confirmed`:
  - treasury ≥ $3,000 is refused;
  - start ≥ $3,000 with top-ups draining it is refused;
  - start < $3,000 passes;
  - the earned repeat passes.
- Update `test_a_repeated_allocation…`: the take_loan schema now has confirm.
- Update `test_a_take_back_below_what_a_department_keeps_is_refused_until_confirmed` for the new text.
- Report: an overrides column and totals.
- `tests/test_fleet_scripted.py`: add `test_a_loan_refusal_the_policy_means_is_confirmed` (monkeypatched).

**Unit D**
- `test_money_results_show_the_treasury_and_what_is_left`: the notes for each case in the D1 table appear, and `box.executed` and `fleet.log` keep the core message.
- `test_update_plan_reports_what_to_fund`: `to_spend`, `to_fund` and `first_buildings`, and `next` equals the spend lines.

**Unit E.** `test_example_fleet_configs` still asserts identical `[fleet]` and `[agents]` blocks.

**Scripted parity.**
1. `pytest -q tests/test_fleet_scripted.py` passes unchanged: turn-for-turn parity (seed 7, 24 months), determinism, and `refused == 0` in the short episode. No new kind fires for the policy. Its confirms follow refusals, so they are earned. Notes don't touch `executed`.
2. `civitas run -c configs/fleet_scripted.toml --seed 0` on main and on the branch: identical `summary.json` scores and identical trajectory `results`.
3. Full `pytest -q`, including `test_benchmark.py`.
4. `civitas prompt -c configs/fleet_smoke.toml --role mayor` to eyeball the new blocks. It makes no LLM call.

## 5. Live evaluation (smoke v20 on seeds 0 and 1, when live runs allow)

Baseline is v16–v19: 8 runs, 24 mayor turns at m3–9, computed from calls.json, turns.json and frames.

| Measure | v16–v19 baseline | v20 target |
|---|---|---|
| Service first buildings unfunded at turn end (dept-turns) | 14 in 10 of 24 turns (education 10) | ≤1 per run; 0 education where a site is ready |
| Education school by m12 | 4/8 runs (misses: 2 money, v16 s0 and v17 s1; 2 land) | 2/2 unless land-blocked |
| Education topped up by m3 | 2/8 | every run whose m3 needs line shows a school |
| Utilities top-ups beyond its needs-line gap (after m0) | $65.8k (~$8.2k/run) | ≤$2k/run |
| "Another" top-ups leaving the treasury under the reserve | 21 of 30 | ≤1/run |
| Non-hold turns ending under the reserve | 11/24 | ≤1/run |
| "treasury only has" failures | 20 (2.5/run) | ≤0.5/run |
| Loans / debt at m12 | 8 loans ($25.5k); debt in 5/8 runs | ≤1 loan across both seeds; each loan visible as refused, then overridden, or first/supply |
| Planning at ≤$100 at a mayor turn after m0 | 4 turns (all via take-backs) | 0 |
| Confirmed (overridden) allocations | 7 (habitual) | only earned; reported per run |
| City / education score | mean 45.5 / 45.8 | city ≥ 46.5, no education 33 without a land block |
| Mayor steps per turn | — | not up by >2; 0 max_steps hits |

**120-month fleet_trinity run** against fleet_trinity_s0_v17:

| Measure | v17 | Target |
|---|---|---|
| Debt at m120 | $50k (cap from m45) | < $15k, 0 months at the cap |
| Interest | $23.9k | < $6k |
| Hold-turn top-ups | $108.7k | < $25k |
| Service upkeep / income (median) | 34% | < 22% |
| Planning at ≤$100 | 33 of 40 turns | < 8 |
| Population / city score | 2,892 / 47.9 | > 4,000 / > 55 (ceiling: scripted fleet at 68.8 and about 7.3k residents) |

## 6. Rejected ideas

- **Hard floor, loan-allowance and "nothing waits" take-back refusals, and per-call priority/reserve refusals** (Money guard). All three judges marked these as harming validity: the harness would decide solvency and funding order, mayor and finance scores would compress, `waiting()` mislabels would block spending, and every refusal costs steps.
- **An end_turn refusal while something waits.** Two of three judges said it pushes spending.
- **A limit of 2 overrides per turn.** Earned confirm plus the report counts cover it. At v19's override habit, the limit would just be spent on the first refusals.
- **The `fund_plan` tool.** Counting a whole budget as one action quietly loosens the 8-action rule (all three judges). It adds a second money path, an enforced reserve and default amounts (automation), about 1 kB of schema per call, and a risk from arrays of objects. Grafted from it instead: first buildings at a $0 target (A3/D3), the dip advice in a hold (text only), repayment in the playbook and results, and the loan note at the cap.
- **A soft reserve refusal on allocate_budget.** At a 100% override habit it costs steps for little. The reserve state now comes back in every money result instead. Revisit if v20 still shows "another" top-ups going under the reserve.
- **A per-supply utilities sum.** It changes utilities' keep and idle arithmetic, and there is no evidence of utilities being under-funded. A second short supply shows up next quarter.
- **Deferred:**
  - replaying a cut of the step-0 reasoning in `agent.py` (it collides with turn_tokens);
  - marginal coverage gain per site;
  - a coverage preview for set_funding;
  - the core set_tax doc (it states a real mechanic and is shared with non-fleet agents);
  - blind parallel batches, beyond the playbook's "one call at a time" advice.

The three counts behind the new baseline and the loan-guard replay (read-only; everything else comes from the three analyses you passed in) are in `/private/tmp/claude-501/-Users-dantesimon-Desktop-civitas-bench/6b0fba63-53df-4475-81ba-43187cae37f1/scratchpad/mayor_money/plan/`: `baseline.py`, `firsts_left2.py` and `loan_guard.py`.
