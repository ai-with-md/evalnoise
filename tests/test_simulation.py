"""The promises, checked against simulated evals where the true score is known.

Each simulated eval draws fresh questions, judges, prompt wordings and runs
from a known random-effects model, so the true score (the average over all
possible questions, judges, wordings and runs) is exactly known.
"""

import random
from statistics import median, pstdev

import pytest

import evalnoise as en

SIMS = 300
SPREAD = dict(item=0.15, judge=0.05, prompt=0.03, item_judge=0.05, item_prompt=0.03, generation=0.05, noise=0.10)


def simulate(rng, items, judges, prompts, runs, sd=SPREAD, truth=0.7, pass_fail=False, systems=None):
    """Rows for one simulated eval. With ``systems``, a dict of system -> (lift, sd of system effects)."""
    a = [rng.gauss(0, sd["item"]) for _ in range(items)]
    b = [rng.gauss(0, sd["judge"]) for _ in range(judges)]
    c = [rng.gauss(0, sd["prompt"]) for _ in range(prompts)]
    ab = [[rng.gauss(0, sd["item_judge"]) for _ in range(judges)] for _ in range(items)]
    ap = [[rng.gauss(0, sd["item_prompt"]) for _ in range(prompts)] for _ in range(items)]
    rows = []
    for system, (lift, by) in (systems or {None: (0.0, {})}).items():
        si = [rng.gauss(0, by.get("item", 0)) for _ in range(items)]
        sj = [rng.gauss(0, by.get("judge", 0)) for _ in range(judges)]
        sp = [rng.gauss(0, by.get("prompt", 0)) for _ in range(prompts)]
        for i in range(items):
            for p in range(prompts):
                for r in range(runs):
                    g = rng.gauss(0, sd["generation"])
                    for j in range(judges):
                        x = truth + lift + a[i] + b[j] + c[p] + ab[i][j] + ap[i][p] + g + si[i] + sj[j] + sp[p]
                        if pass_fail:
                            y = 1.0 if rng.random() < x else 0.0
                        else:
                            y = x + rng.gauss(0, sd["noise"])
                        row = {"item": i, "judge": j, "prompt": p, "run": r, "score": y}
                        if system is not None:
                            row["system"] = system
                        rows.append(row)
    return rows


def coverage(results, truth):
    honest = sum(s.interval()[0] <= truth <= s.interval()[1] for s in results) / len(results)
    usual = sum(s.naive_interval()[0] <= truth <= s.naive_interval()[1] for s in results) / len(results)
    return honest, usual


def test_interval_covers_the_true_score():
    rng = random.Random(1)
    results = [en.study(simulate(rng, 50, 3, 3, 2)) for _ in range(SIMS)]
    honest, usual = coverage(results, 0.7)
    assert honest >= 0.92
    assert usual <= 0.85  # the usual interval misses the judge and wording noise


def test_interval_covers_pass_fail_scores():
    rng = random.Random(2)
    sd = {**SPREAD, "generation": 0.0, "noise": 0.0}
    results = [en.study(simulate(rng, 60, 3, 2, 2, sd=sd, pass_fail=True)) for _ in range(SIMS)]
    honest, _ = coverage(results, 0.7)
    assert honest >= 0.93


def test_without_judge_or_wording_noise_the_interval_is_not_much_wider():
    rng = random.Random(3)
    sd = {**SPREAD, "judge": 0.0, "prompt": 0.0}
    results = [en.study(simulate(rng, 50, 3, 3, 2, sd=sd)) for _ in range(SIMS)]
    honest, usual = coverage(results, 0.7)
    assert honest >= 0.93 and usual >= 0.92
    widths = [(s.interval()[1] - s.interval()[0]) / (s.naive_interval()[1] - s.naive_interval()[0]) for s in results]
    assert median(widths) < 1.3


def test_comparisons_with_biased_judges_rarely_call_noise_a_difference():
    rng = random.Random(4)
    systems = {
        "old": (0.0, {"item": 0.08, "judge": 0.03, "prompt": 0.02}),
        "new": (0.0, {"item": 0.08, "judge": 0.03, "prompt": 0.02}),
    }
    results = [en.compare(simulate(rng, 60, 3, 3, 2, systems=systems)) for _ in range(SIMS)]
    honest = sum(s.interval()[0] > 0 or s.interval()[1] < 0 for s in results) / SIMS
    usual = sum(s.naive_interval()[0] > 0 or s.naive_interval()[1] < 0 for s in results) / SIMS
    assert honest <= 0.08
    assert usual >= 0.2


def test_comparisons_still_find_real_differences():
    rng = random.Random(5)
    systems = {"old": (0.0, {"item": 0.08}), "new": (0.06, {"item": 0.08})}
    results = [en.compare(simulate(rng, 80, 3, 2, 2, systems=systems)) for _ in range(SIMS)]
    found = sum(s.interval()[0] > 0 for s in results) / SIMS
    covered = sum(s.interval()[0] <= 0.06 <= s.interval()[1] for s in results) / SIMS
    assert found >= 0.8
    assert covered >= 0.93


def test_judge_spread_is_estimated_without_bias():
    rng = random.Random(6)
    estimates = [
        next(c.raw_estimate for c in en.study(simulate(rng, 40, 6, 2, 1)).components if c.name == "judge")
        for _ in range(SIMS)
    ]
    assert sum(estimates) / SIMS == pytest.approx(SPREAD["judge"] ** 2, rel=0.2)


def test_projection_predicts_the_spread_of_scores_at_a_new_design():
    s = en.study(simulate(random.Random(7), 20, 3, 3, 2))
    truth = {
        "question": SPREAD["item"] ** 2, "judge": SPREAD["judge"] ** 2, "prompt": SPREAD["prompt"] ** 2,
        "question x judge": SPREAD["item_judge"] ** 2, "question x prompt": SPREAD["item_prompt"] ** 2,
        "question x prompt x run": SPREAD["generation"] ** 2, "question x judge x prompt x run": SPREAD["noise"] ** 2,
    }
    labels = ["question", "judge", "prompt", "run"]
    for mask in s._masks:
        s._var[mask] = truth.get(" x ".join(labels[f] for f in range(4) if mask >> f & 1), 0.0)
    predicted = s.project(items=100, judges=2, prompts=2, runs=1).se
    rng = random.Random(8)
    scores = [en.study(simulate(rng, 100, 2, 2, 1)).mean for _ in range(SIMS)]
    assert pstdev(scores) == pytest.approx(predicted, rel=0.1)


def test_plan_from_a_pilot_beats_the_pilot_design_in_practice():
    pilot = en.study(simulate(random.Random(9), 60, 6, 6, 2))
    plan = pilot.plan(budget=30, generation_cost=0.01, judge_cost=0.005, limits={"judge": 3, "prompt": 3})
    pilot_per_item = 3 * 2 * (0.01 + 3 * 0.005)
    same_money_items = int(30 // pilot_per_item)
    rng = random.Random(10)
    sizes = plan.sizes
    assert plan.sizes["run"] == 1  # reruns add little here: spend on questions instead
    planned = [en.study(simulate(rng, sizes["item"], sizes["judge"], sizes["prompt"], sizes["run"])).mean for _ in range(100)]
    naive = [en.study(simulate(rng, same_money_items, 3, 3, 2)).mean for _ in range(100)]
    assert pstdev(planned) < pstdev(naive)
