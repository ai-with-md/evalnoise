import random
from statistics import mean, variance

import pytest

import evalnoise as en


def grid(rng, items=40, judges=3, prompts=2, runs=2, sd=None, mu=0.7):
    """Scores from a known random-effects model, with nested runs."""
    sd = sd or dict(item=0.15, judge=0.05, prompt=0.03, item_judge=0.05, generation=0.05, noise=0.1)
    a = [rng.gauss(0, sd["item"]) for _ in range(items)]
    b = [rng.gauss(0, sd["judge"]) for _ in range(judges)]
    c = [rng.gauss(0, sd["prompt"]) for _ in range(prompts)]
    ab = [[rng.gauss(0, sd["item_judge"]) for _ in range(judges)] for _ in range(items)]
    rows = []
    for i in range(items):
        for p in range(prompts):
            for r in range(runs):
                g = rng.gauss(0, sd["generation"])
                for j in range(judges):
                    y = mu + a[i] + b[j] + c[p] + ab[i][j] + g + rng.gauss(0, sd["noise"])
                    rows.append({"item": f"q{i}", "judge": f"j{j}", "prompt": f"p{p}", "run": r, "score": y})
    return rows


def test_two_way_matches_textbook_formulas():
    rng = random.Random(0)
    scores = [[rng.random() for _ in range(3)] for _ in range(8)]
    rows = [{"item": i, "judge": j, "score": scores[i][j]} for i in range(8) for j in range(3)]
    s = en.study(rows)
    n_i, n_j = 8, 3
    grand = mean(v for row in scores for v in row)
    row_means = [mean(r) for r in scores]
    col_means = [mean(scores[i][j] for i in range(n_i)) for j in range(n_j)]
    ms_i = n_j * sum((m - grand) ** 2 for m in row_means) / (n_i - 1)
    ms_j = n_i * sum((m - grand) ** 2 for m in col_means) / (n_j - 1)
    resid = sum(
        (scores[i][j] - row_means[i] - col_means[j] + grand) ** 2 for i in range(n_i) for j in range(n_j)
    ) / ((n_i - 1) * (n_j - 1))
    by_name = {c.name: c.raw_estimate for c in s.components}
    assert by_name["question x judge"] == pytest.approx(resid)
    assert by_name["question"] == pytest.approx((ms_i - resid) / n_j)
    assert by_name["judge"] == pytest.approx((ms_j - resid) / n_i)
    assert s.naive_se == pytest.approx((variance(row_means) / n_i) ** 0.5)


def test_components_recover_a_known_model():
    rows = grid(random.Random(1), items=300, judges=8, prompts=6, runs=3)
    s = en.study(rows)
    found = {c.name: c.variance for c in s.components}
    assert found["question"] == pytest.approx(0.15 ** 2, rel=0.25)
    assert found["question x judge"] == pytest.approx(0.05 ** 2, rel=0.3)
    assert found["question x prompt x run"] == pytest.approx(0.05 ** 2, rel=0.3)
    assert found["question x judge x prompt x run"] == pytest.approx(0.1 ** 2, rel=0.1)


def test_run_labels_mean_nothing_across_questions():
    rows = grid(random.Random(2))
    rng = random.Random(3)
    relabel = {}
    for r in rows:
        key = (r["item"], r["prompt"])
        relabel.setdefault(key, rng.sample([0, 1], 2))
    shuffled = [{**r, "run": relabel[(r["item"], r["prompt"])][r["run"]]} for r in rows]
    a, b = en.study(rows), en.study(shuffled)
    assert a.se == pytest.approx(b.se)
    assert [c.name for c in a.components] == [c.name for c in b.components]
    assert not any(c.name in ("run", "judge x run", "prompt x run") for c in a.components)


def test_usual_interval_ignores_judges_and_wordings():
    s = en.study(grid(random.Random(4), sd=dict(item=0.15, judge=0.1, prompt=0.05, item_judge=0.02, generation=0.02, noise=0.05)))
    assert s.se > 1.5 * s.naive_se
    assert s.components[0].name == "judge"
    assert "By judge:" in s.report()


def test_fixed_judges_drop_their_own_effect():
    rows = grid(random.Random(5), sd=dict(item=0.15, judge=0.1, prompt=0.0, item_judge=0.03, generation=0.03, noise=0.05))
    random_judges = en.study(rows)
    fixed_judges = en.study(rows, fixed=["judge"])
    assert fixed_judges.se < random_judges.se
    assert all(c.share == 0 for c in fixed_judges.components if c.name == "judge")
    with pytest.raises(ValueError, match="fixed"):
        fixed_judges.project(judges=1)


def test_projection_at_the_pilot_design_matches_the_study():
    s = en.study(grid(random.Random(6)))
    assert s.project().se == pytest.approx(s.se)
    assert s.project(items=400).se < s.se
    assert s.project(questions=40, judges=1).se > s.se
    with pytest.raises(ValueError, match="unknown factor"):
        s.project(graders=2)


def test_plan_for_a_budget_stays_within_it_and_beats_the_alternatives():
    s = en.study(grid(random.Random(7)))
    plan = s.plan(budget=50, generation_cost=0.01, judge_cost=0.005, limits={"judge": 3})
    assert plan.cost <= 50
    assert plan.sizes["judge"] <= 3
    for judges in (1, 2, 3):
        for prompts in range(1, 6):
            for runs in range(1, 6):
                per_item = prompts * runs * (0.01 + judges * 0.005)
                items = int(50 // per_item)
                alt = s.project(items=items, judges=judges, prompts=prompts, runs=runs).se
                assert plan.se <= alt + 1e-12
    assert "Best design for $50.00" in plan.summary()


def test_plan_for_a_target_reaches_it_cheaply():
    s = en.study(grid(random.Random(8)))
    plan = s.plan(target=0.08, generation_cost=0.01, judge_cost=0.005)
    assert plan.half_width <= 0.08
    assert s.plan(target=0.08, generation_cost=0.01, judge_cost=0.005, max_items=10 ** 6).cost == pytest.approx(plan.cost)
    with pytest.raises(ValueError, match="no design"):
        s.plan(target=0.0001, generation_cost=0.01)
    capped = s.plan(budget=1000, generation_cost=0.01, max_items=50)
    assert capped.sizes["item"] == 50


def test_compare_cancels_noise_shared_by_both_systems():
    rng = random.Random(9)
    strictness = {"j0": 0.0, "j1": -0.3, "j2": 0.25}
    rows = []
    for system, lift in (("old", 0.0), ("new", 0.05)):
        for i in range(60):
            base = random.Random(i).gauss(0, 0.2)
            for j, strict in strictness.items():
                rows.append({"system": system, "item": i, "judge": j, "score": 0.6 + base + strict + lift + rng.gauss(0, 0.05)})
    single = en.study([r for r in rows if r["system"] == "new"], facets=["judge"])
    diff = en.compare(rows)
    assert single.components[0].name == "judge"
    assert diff.mean == pytest.approx(0.05, abs=0.02)
    assert diff.se < single.se / 3
    assert diff.interval()[0] > 0
    assert "new is reliably better than old" in diff.report()


def test_compare_finds_new_and_old_or_asks():
    rows = [{"model": m, "item": i, "judge": j, "score": (i + j) % 2} for m in ("a1", "b1") for i in range(4) for j in range(2)]
    with pytest.raises(ValueError, match="say which system is new"):
        en.compare(rows, system="model")
    s = en.compare(rows, system="model", new="b1")
    assert s.compared == ("b1", "a1")


def test_missing_scores_drop_the_question():
    rows = grid(random.Random(10), items=10)
    rows = [r for r in rows if not (r["item"] == "q3" and r["judge"] == "j1" and r["run"] == 0)]
    s = en.study(rows)
    assert s.sizes["item"] == 9
    assert s.dropped == ["q3"]
    assert "left out 1 questions" in s.report()


def test_one_judge_is_noted_not_analysed():
    rows = [{"item": i, "judge": "only", "prompt": p, "score": (i * p) % 3} for i in range(5) for p in range(2)]
    s = en.study(rows)
    assert s.factors == ("item", "prompt")
    assert "only one judge" in s.report()


@pytest.mark.parametrize(
    "rows, kwargs, message",
    [
        ([{"item": 1, "judge": "a", "score": 1}, {"item": 1, "judge": "a", "score": 0}], {}, "add a run column"),
        ([{"item": 1, "judge": "a", "score": 1}, {"item": 1, "judge": "b", "score": 0}], {}, "fewer than 2 questions"),
        ([{"item": 1, "judge": "a", "score": "maybe"}], {}, "cannot read 'maybe'"),
        ([{"question": 1, "score": 1}], {}, "no 'item' column"),
        ([], {}, "no rows"),
    ],
)
def test_clear_errors(rows, kwargs, message):
    with pytest.raises(ValueError, match=message):
        en.study(rows, **kwargs)


def test_plan_needs_budget_or_target():
    s = en.study(grid(random.Random(11), items=10))
    with pytest.raises(ValueError, match="either a budget or a target"):
        s.plan()
    with pytest.raises(ValueError, match="does not cover"):
        s.plan(budget=0.001, generation_cost=1)
