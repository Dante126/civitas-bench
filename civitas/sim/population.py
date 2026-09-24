"""Population: an aggregate age x education vector, ageing, schooling and job matching.

The number of residents is still set by housing (developed residential
tiles x capacity). `cohorts[age, edu]` describes *who* those residents are:
migrants fill new housing with a fixed profile, people who leave are removed
proportionally, and every month a share of each cohort ages or graduates.
Graduation only happens for students who have a school seat, so schools
shape the workforce a few years later.
"""
from __future__ import annotations

import numpy as np

from . import catalog as C

AGES = ("child", "teen", "adult", "senior")
EDUS = ("none", "elementary", "high", "university")
CHILD, TEEN, ADULT, SENIOR = range(4)
NONE_E, ELEM, HIGH, UNIV = range(4)


def _migrant_profile() -> np.ndarray:
    p = np.zeros((4, 4))
    p[CHILD, NONE_E] = 0.17
    p[CHILD, ELEM] = 0.03
    p[TEEN, ELEM] = 0.06
    p[TEEN, NONE_E] = 0.02
    adult_edu = np.array(C.MIGRANT_ADULT_EDU)
    p[ADULT] = 0.62 * adult_edu / adult_edu.sum()
    p[SENIOR] = 0.10 * np.array([0.35, 0.4, 0.18, 0.07])
    return p / p.sum()


class PopulationMixin:
    def _init_population(self) -> None:
        self.cohorts = np.zeros((4, 4))
        self.enrollment = {"elementary": 0.0, "high": 0.0, "university": 0.0}
        self.deaths_month = 0.0

    # --------------------------------------------------------- migration
    def _rescale_population(self, pop: float) -> None:
        total = float(self.cohorts.sum())
        if pop > total:
            self.cohorts += (pop - total) * _migrant_profile()
        elif total > 0:
            self.cohorts *= pop / total

    # -------------------------------------------------- ageing / schooling
    def _step_cohorts(self) -> None:
        """One month of births, ageing, graduation and deaths (called once per tick)."""
        n = self.cohorts
        if n.sum() <= 0:
            self.deaths_month = 0.0
            return
        new = n.copy()
        # births
        new[CHILD, NONE_E] += n[ADULT].sum() * C.BIRTH_RATE
        # schooling: only students with a seat progress
        boost = 1.0 + C.LIBRARY_BOOST * self.stats.get("library_coverage", 0.0)
        boost *= C.POLICIES["education_boost"]["rate"] if self.policy_on("education_boost") else 1.0
        e = self.enrollment
        g_elem = n[CHILD, NONE_E] * C.GRAD_RATE["elementary"] * e["elementary"] * boost
        g_teen = n[TEEN, NONE_E] * C.GRAD_RATE["elementary"] * e["elementary"] * boost
        g_high = n[TEEN, ELEM] * C.GRAD_RATE["high"] * e["high"] * boost
        g_univ = n[ADULT, HIGH] * C.STUDENT_AGE_SHARE * C.GRAD_RATE["university"] * e["university"] * boost
        new[CHILD, NONE_E] -= g_elem
        new[CHILD, ELEM] += g_elem
        new[TEEN, NONE_E] -= g_teen
        new[TEEN, ELEM] += g_teen
        new[TEEN, ELEM] -= g_high
        new[TEEN, HIGH] += g_high
        new[ADULT, HIGH] -= g_univ
        new[ADULT, UNIV] += g_univ
        # ageing (education level carries over)
        for a, rate in ((CHILD, C.AGE_RATE["child"]), (TEEN, C.AGE_RATE["teen"]), (ADULT, C.AGE_RATE["adult"])):
            moved = n[a] * rate
            new[a] -= moved
            new[a + 1] += moved
        # deaths: seniors, more when the city is unhealthy
        sick = 1.0 - self.stats.get("health", 0.7)
        d = n[SENIOR] * C.SENIOR_DEATH_RATE * (1.0 + 1.5 * sick) + n[ADULT] * C.ADULT_DEATH_RATE * sick
        new[SENIOR] -= d
        self.deaths_month = float(d.sum())
        self.cohorts = np.maximum(new, 0.0)

    def _update_enrollment(self) -> None:
        n = self.cohorts
        st = self.stats
        children, teens = n[CHILD].sum(), n[TEEN].sum()
        students_u = n[ADULT, HIGH] * C.STUDENT_AGE_SHARE
        eff = self.funding_eff("education")

        def rate(students, seats, covered):
            if students <= 0:
                return 0.0
            return float(min(1.0, min(students * covered, seats * eff) / students))

        self.enrollment = {
            "elementary": rate(children, st.get("seats_elementary", 0), st.get("school_coverage", 0.0)),
            "high": rate(teens, st.get("seats_high", 0), st.get("high_coverage", 0.0)),
            "university": rate(students_u, st.get("seats_university", 0), 1.0 if st.get("seats_university", 0) else 0.0),
        }

    # ------------------------------------------------------ workforce
    def _match_jobs(self, jobs_by_sector: dict[str, float]) -> dict:
        """Fill jobs by required education. Over-educated workers take lower jobs."""
        n = self.cohorts
        sick_share = C.SICK_SHARE * (1.0 - self.stats.get("health", 0.75))
        workers_e = (n[ADULT] * C.LABOR_PARTICIPATION + n[TEEN] * C.TEEN_PARTICIPATION) * (1.0 - sick_share)
        need = {s: jobs * np.array(C.JOB_EDU[s]) for s, jobs in jobs_by_sector.items()}
        demand_e = sum(need.values()) if need else np.zeros(4)
        avail = workers_e.copy()
        filled_e = np.zeros(4)
        for e in (UNIV, HIGH, ELEM, NONE_E):
            want = float(demand_e[e])
            for src in range(e, 4):                    # exact level first, then over-educated
                take = min(want, avail[src])
                avail[src] -= take
                filled_e[e] += take
                want -= take
                if want <= 1e-9:
                    break
        ratio = np.divide(filled_e, demand_e, out=np.zeros(4), where=demand_e > 0)
        filled = {s: float((v * ratio).sum()) for s, v in need.items()}
        workers = float(workers_e.sum())
        unemployed = float(avail.sum())
        return {
            "workers": workers, "filled": filled,
            "unemployment": unemployed / workers if workers > 0 else 0.0,
            "unemployment_by_edu": {EDUS[e]: round(float(avail[e] / workers_e[e]), 4) if workers_e[e] > 0 else 0.0
                                    for e in range(4)},
            "skilled_fill": float(filled_e[HIGH:].sum() / demand_e[HIGH:].sum()) if demand_e[HIGH:].sum() > 0 else 1.0,
            "sick_workers": float((n[ADULT] * C.LABOR_PARTICIPATION).sum() * sick_share),
            "employed_by_edu": workers_e - avail,
        }

    def education_index(self) -> float:
        """Share of adults with at least a high-school education."""
        a = self.cohorts[ADULT]
        return float(a[HIGH:].sum() / a.sum()) if a.sum() > 0 else 0.0

    def cohort_summary(self) -> dict:
        n = self.cohorts
        adults = n[ADULT]
        return {
            "children": int(n[CHILD].sum()), "teens": int(n[TEEN].sum()),
            "adults": int(adults.sum()), "seniors": int(n[SENIOR].sum()),
            "adult_education_pct": {EDUS[e]: round(100 * float(adults[e] / adults.sum()), 1) if adults.sum() > 0 else 0.0
                                    for e in range(4)},
        }
