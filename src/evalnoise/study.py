"""Split the noise in an eval score by where it comes from, and plan where to spend the next calls.

An eval score moves when you change the questions, but also when you change the
judge, the wording of the prompt, or simply run it again. The usual error bar
counts only the questions. This module estimates how much each source adds,
using generalizability theory: a random-effects analysis of variance on a
balanced pilot, where every question is scored under every combination of
judges, prompt wordings and runs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import product
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ._stats import t_quantile

JUDGE_WORDS = ("judge", "grader", "rater", "annotator", "evaluator", "scorer", "labeler", "labeller")
PROMPT_WORDS = ("prompt", "template", "wording", "paraphrase", "instruction", "phrasing", "variant", "format")
RUN_WORDS = ("run", "seed", "sample", "attempt", "trial", "repeat", "replicate", "repetition", "generation")

TRUE_WORDS = {"1", "true", "yes", "y", "pass", "passed", "correct", "right", "success", "ok", "good"}
FALSE_WORDS = {"0", "false", "no", "n", "fail", "failed", "incorrect", "wrong", "failure", "bad", "error"}


def _kind(name: str) -> str:
    """'judge', 'prompt', 'run' or 'other', from a facet's name."""
    words = re.split(r"[^a-z]+", name.lower())
    for kind, vocab in (("judge", JUDGE_WORDS), ("prompt", PROMPT_WORDS), ("run", RUN_WORDS)):
        if any(w.startswith(v) for w in words for v in vocab if w):
            return kind
    return "other"


def to_score(value) -> float:
    """Read a score: a number, a boolean, or a word like pass, fail, yes, no.

    >>> to_score("pass"), to_score(True), to_score("0.75"), to_score(4)
    (1.0, 1.0, 0.75, 4.0)
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isnan(value):
            raise ValueError("score is missing (NaN)")
        return float(value)
    text = str(value).strip().lower()
    if text in TRUE_WORDS:
        return 1.0
    if text in FALSE_WORDS:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"cannot read {value!r} as a score") from None
    if math.isnan(number):
        raise ValueError("score is missing (NaN)")
    return number


_DESCRIPTIONS = {
    ("item",): "some questions are harder than others",
    ("judge",): "some judges are stricter than others across the board",
    ("prompt",): "some prompt wordings score higher across the board",
    ("run",): "some runs score higher across the board",
    ("item", "judge"): "judges disagree about particular questions",
    ("item", "prompt"): "a wording helps on some questions and hurts on others",
    ("item", "run"): "answers change from run to run",
    ("judge", "prompt"): "judges react differently to the wordings",
    ("judge", "run"): "judges react differently to the runs",
    ("prompt", "run"): "runs differ by wording",
}

_DIFFERENCE_DESCRIPTIONS = {
    ("item",): "the new version is better on some questions and worse on others",
    ("judge",): "judges disagree about which version is better overall (for example, a judge that favours one model's style)",
    ("prompt",): "the gap between the versions depends on the wording",
    ("run",): "some runs favour one version",
    ("item", "judge"): "judges disagree about which version is better on particular questions",
    ("item", "prompt"): "on some questions the gap depends on the wording",
    ("item", "run"): "answers change from run to run",
}


@dataclass(frozen=True)
class Component:
    """One source of noise.

    ``variance`` is its estimated variance on the score scale (negative
    estimates are set to zero), ``share`` is its part of the noise in the
    overall score as the study was run, and ``share_of_single_score`` is its
    part of the variance of one individual score.
    """

    name: str
    factors: Tuple[str, ...]
    variance: float
    raw_estimate: float
    df: int
    share: float
    share_of_single_score: float
    description: str


@dataclass(frozen=True)
class Projection:
    """The expected noise in the overall score for a given design."""

    sizes: Dict[str, int]
    se: float
    level: float
    fixed: Tuple[str, ...] = ()
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def half_width(self) -> float:
        """Approximate half width of the interval, using the normal distribution."""
        return NormalDist().inv_cdf(0.5 + self.level / 2) * self.se

    def summary(self) -> str:
        design = " x ".join(f"{n:,} {_plural(self.labels.get(name, name), n)}" for name, n in self.sizes.items())
        return f"{design}: the score would be good to about +-{self.half_width:.3g} ({self.level:.0%})."

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class Plan:
    """A recommended design: how many questions, judges, wordings and runs."""

    sizes: Dict[str, int]
    se: float
    level: float
    cost: float
    generation_calls: int
    judge_calls: int
    current: Projection
    current_cost: float
    goal: str
    what: str = "score"
    labels: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    @property
    def half_width(self) -> float:
        return NormalDist().inv_cdf(0.5 + self.level / 2) * self.se

    def summary(self) -> str:
        design = ", ".join(f"{n:,} {_plural(self.labels.get(name, name), n)}" for name, n in self.sizes.items())
        lines = [
            f"Best design {self.goal}: {design}.",
            f"  {self.generation_calls:,} model calls and {self.judge_calls:,} judge calls, costing {_money(self.cost)}.",
            f"  The {self.what} would be good to about +-{self.half_width:.3g} ({self.level:.0%}).",
            f"  On the same basis, the pilot's own design ({_money(self.current_cost)} at these prices) "
            f"gives about +-{self.current.half_width:.3g}.",
        ]
        lines.extend(f"  Careful: {w}" for w in self.warnings)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _money(value: float) -> str:
    return f"${value:,.2f}" if value >= 0.995 else f"${value:.4g}"


def _plural(name: str, n: int) -> str:
    if n == 1:
        return name
    if name.endswith("s"):
        return name
    if name.endswith("y") and not name.endswith(("ay", "ey", "oy", "uy")):
        return name[:-1] + "ies"
    return name + "s"


def _bits(mask: int, k: int) -> List[int]:
    return [f for f in range(k) if mask >> f & 1]


class Study:
    """A generalizability study of one eval score.

    Build it with :func:`study` (rows in memory) or ``read_scores(path).study()``.

    Attributes:
        mean: the overall score (average of all scores).
        se: its standard error, counting every source of noise that varies.
        naive_se: the usual standard error, from per-question averages only.
        df: degrees of freedom of ``se`` (Satterthwaite). Small when the
            estimate rests on few judges or wordings.
        components: the sources of noise, largest share first.
        sizes: how many questions, judges, wordings and runs the pilot used.
    """

    def __init__(
        self,
        cells: Mapping[Tuple[int, ...], float],
        factors: Sequence[str],
        levels: Sequence[Sequence[str]],
        fixed: Iterable[str] = (),
        judging: Optional[Iterable[str]] = None,
        notes: Sequence[str] = (),
        dropped: Sequence[str] = (),
        compared: Optional[Tuple[str, str]] = None,
        nested: Optional[Iterable[str]] = None,
    ) -> None:
        self.factors = tuple(factors)
        self.levels = [list(l) for l in levels]
        self.sizes = {name: len(l) for name, l in zip(self.factors, self.levels)}
        self.fixed = tuple(fixed)
        unknown = [f for f in self.fixed if f not in self.factors[1:]]
        if unknown:
            raise ValueError(f"fixed facets {unknown} are not in the data (facets: {list(self.factors[1:])})")
        if judging is None:
            judging = [f for f in self.factors[1:] if _kind(f) == "judge"]
        self.judging = tuple(judging)
        if nested is None:
            nested = [f for f in self.factors[1:] if _kind(f) == "run" and f not in self.judging and f not in self.fixed]
        self.nested = tuple(nested)
        bad = [f for f in self.nested if f not in self.factors[1:]]
        if bad:
            raise ValueError(f"nested facets {bad} are not in the data")
        self.labels = {name: name for name in self.factors}
        self.labels[self.factors[0]] = "question"
        self.notes = list(notes)
        self.dropped = list(dropped)
        self.compared = compared
        self._analyse(cells)

    # Analysis of variance for a balanced, fully crossed random design with one score per cell.
    def _analyse(self, cells: Mapping[Tuple[int, ...], float]) -> None:
        k = len(self.factors)
        n = [len(l) for l in self.levels]
        total = math.prod(n)
        if len(cells) != total:
            raise ValueError("the design is not balanced")
        values = list(cells.values())
        self.mean = sum(values) / total
        self.count = total
        centred = {key: y - self.mean for key, y in cells.items()}
        self.means_by: Dict[str, Dict[str, float]] = {}

        squares: Dict[int, float] = {}
        for mask in range(1 << k):
            idx = _bits(mask, k)
            per_cell = total // math.prod(n[f] for f in idx) if idx else total
            sums: Dict[Tuple[int, ...], float] = {}
            for key, y in centred.items():
                sub = tuple(key[f] for f in idx)
                sums[sub] = sums.get(sub, 0.0) + y
            squares[mask] = sum(s * s for s in sums.values()) / per_cell
            if len(idx) == 1 and idx[0] > 0:
                f = idx[0]
                self.means_by[self.factors[f]] = {
                    self.levels[f][key[0]]: self.mean + s / per_cell for key, s in sorted(sums.items())
                }

        masks = [m for m in range(1, 1 << k)]
        sums_sq: Dict[int, float] = {}
        self._ms: Dict[int, float] = {}
        self._df: Dict[int, int] = {}
        for mask in masks:
            ss = 0.0
            sub = mask
            while True:
                sign = -1 if (bin(mask).count("1") - bin(sub).count("1")) % 2 else 1
                ss += sign * squares[sub]
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            self._df[mask] = math.prod(n[f] - 1 for f in _bits(mask, k))
            sums_sq[mask] = ss

        # Runs are nested: run 1 of one question has nothing to do with run 1 of another,
        # so a run facet only appears together with the question and the wordings it was generated under.
        # Components without them are pooled into the ones with them.
        self._parents: Dict[int, int] = {}
        for name in self.nested:
            bit = 1 << self.factors.index(name)
            parents = 1 | sum(
                1 << i for i, f in enumerate(self.factors[1:], start=1)
                if f not in self.judging and f not in self.nested
            )
            self._parents[bit] = parents
            for mask in list(masks):
                if mask & bit and mask & parents != parents:
                    target = mask | parents
                    sums_sq[target] += sums_sq.pop(mask)
                    self._df[target] += self._df.pop(mask)
                    masks.remove(mask)
        for mask in masks:
            self._ms[mask] = max(sums_sq[mask], 0.0) / self._df[mask]

        # Expected mean squares: E[MS_S] = sum over T containing S of (product of n outside T) * var_T.
        # Solve from the highest interaction down, keeping each estimate as a combination of mean squares.
        outside = {m: math.prod(n[f] for f in range(k) if not m >> f & 1) for m in masks}
        coef: Dict[int, Dict[int, float]] = {}
        for mask in sorted(masks, key=lambda m: -bin(m).count("1")):
            c = {mask: 1.0 / outside[mask]}
            for sup in masks:
                if sup != mask and sup & mask == mask:
                    scale = outside[sup] / outside[mask]
                    for t, v in coef[sup].items():
                        c[t] = c.get(t, 0.0) - scale * v
            coef[mask] = c
        self._coef = coef
        self._raw = {m: sum(v * self._ms[t] for t, v in coef[m].items()) for m in masks}
        self._var = {m: max(0.0, v) for m, v in self._raw.items()}
        self._masks = masks
        self._n = n

        item_means: Dict[int, float] = {}
        for key, y in cells.items():
            item_means[key[0]] = item_means.get(key[0], 0.0) + y
        per_item = total / n[0]
        means = [s / per_item for s in item_means.values()]
        m = sum(means) / len(means)
        self.naive_se = math.sqrt(sum((x - m) ** 2 for x in means) / (len(means) - 1) / len(means))

        weights = self._weights(dict(zip(self.factors, n)))
        self.variance = sum(weights[mask] * self._var[mask] for mask in masks)
        self.se = math.sqrt(self.variance)
        self.df = self._satterthwaite(weights)

        single = sum(self._var.values())
        components = []
        for mask in masks:
            names = tuple(self.factors[f] for f in _bits(mask, k))
            components.append(
                Component(
                    name=" x ".join(self.labels[f] for f in names),
                    factors=names,
                    variance=self._var[mask],
                    raw_estimate=self._raw[mask],
                    df=self._df[mask],
                    share=weights[mask] * self._var[mask] / self.variance if self.variance > 0 else 0.0,
                    share_of_single_score=self._var[mask] / single if single > 0 else 0.0,
                    description=self._describe(names, mask == (1 << k) - 1, mask),
                )
            )
        self.components = sorted(components, key=lambda c: (-c.share, -c.share_of_single_score))

    def _describe(self, names: Tuple[str, ...], highest: bool, mask: int = 0) -> str:
        for bit, parents in self._parents.items():
            if mask == bit | parents and not highest:
                return "answers change from run to run"
            if mask == bit | parents and highest:
                return "answers change from run to run, plus one-off noise in single scores"
        kinds = tuple("item" if i == 0 and name == self.factors[0] else _kind(name) for i, name in enumerate(names))
        text = None
        if len(set(kinds)) == len(kinds) and "other" not in kinds:
            table = _DIFFERENCE_DESCRIPTIONS if self.compared else _DESCRIPTIONS
            text = table.get(tuple(sorted(kinds, key=["item", "judge", "prompt", "run"].index)))
        if highest and len(names) > 1:
            return f"{text}, plus one-off noise in single scores" if text else "everything else, including one-off noise in single scores"
        if text:
            return text
        shown = [self.labels[n] for n in names]
        if len(shown) > 1:
            return "an interaction between " + ", ".join(shown[:-1]) + " and " + shown[-1]
        return f"some {_plural(shown[0], 2)} score higher than others"

    def _weights(self, sizes: Mapping[str, int]) -> Dict[int, float]:
        """How much each variance component counts in the variance of the overall average."""
        k = len(self.factors)
        weights = {}
        for mask in self._masks:
            names = [self.factors[f] for f in _bits(mask, k)]
            if all(name in self.fixed for name in names):
                weights[mask] = 0.0
            else:
                weights[mask] = 1.0 / math.prod(sizes[name] for name in names)
        return weights

    def _satterthwaite(self, weights: Mapping[int, float]) -> float:
        combo: Dict[int, float] = {}
        for mask in self._masks:
            if self._raw[mask] > 0:
                for t, v in self._coef[mask].items():
                    combo[t] = combo.get(t, 0.0) + weights[mask] * v
        denominator = sum((b * self._ms[t]) ** 2 / self._df[t] for t, b in combo.items())
        if denominator <= 0 or self.variance <= 0:
            return math.inf
        return max(1.0, self.variance ** 2 / denominator)

    def interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Interval for the score that counts every source of noise (t with Satterthwaite degrees of freedom)."""
        half = self._multiplier(level) * self.se
        return self.mean - half, self.mean + half

    def naive_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """The usual interval, from per-question averages only."""
        half = t_quantile(0.5 + level / 2, self.sizes[self.factors[0]] - 1) * self.naive_se
        return self.mean - half, self.mean + half

    def _multiplier(self, level: float) -> float:
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        if math.isinf(self.df):
            return NormalDist().inv_cdf(0.5 + level / 2)
        return t_quantile(0.5 + level / 2, self.df)

    def project(self, level: float = 0.95, **sizes: int) -> Projection:
        """Expected noise in the score for another design, for example ``project(items=500, judges=1)``.

        Name each factor as it appears in the data (``item``, ``judge`` and so
        on, singular or plural). Factors left out keep the pilot's size.
        """
        design = self._design(sizes)
        weights = self._weights(design)
        variance = sum(weights[mask] * self._var[mask] for mask in self._masks)
        return Projection(sizes=design, se=math.sqrt(variance), level=level, fixed=self.fixed, labels=self.labels)

    def _name(self, key: str) -> str:
        lookup = {}
        for name in self.factors:
            lookup[name] = name
            lookup[_plural(name, 2)] = name
            lookup[name.lower()] = name
            lookup[_plural(name, 2).lower()] = name
        for alias in ("item", "items", "question", "questions"):
            lookup.setdefault(alias, self.factors[0])
        name = lookup.get(key) or lookup.get(key.lower())
        if name is None:
            raise ValueError(f"unknown factor {key!r}; the data has {list(self.factors)}")
        return name

    def _design(self, sizes: Mapping[str, int]) -> Dict[str, int]:
        design = dict(self.sizes)
        for key, value in sizes.items():
            name = self._name(key)
            if name in self.fixed and value != self.sizes[name]:
                raise ValueError(f"{name} is fixed: always the same {self.sizes[name]}, so its count cannot change")
            if int(value) != value or value < 1:
                raise ValueError(f"{key} must be a whole number of at least 1")
            design[name] = int(value)
        return design

    def _calls(self, design: Mapping[str, int]) -> Tuple[int, int]:
        generation = design[self.factors[0]] * math.prod(
            design[f] for f in self.factors[1:] if f not in self.judging
        )
        judge = generation * math.prod(design[f] for f in self.judging)
        return generation, judge

    def plan(
        self,
        budget: Optional[float] = None,
        target: Optional[float] = None,
        generation_cost: float = 1.0,
        judge_cost: float = 0.0,
        max_items: Optional[int] = None,
        max_per_facet: int = 5,
        limits: Optional[Mapping[str, int]] = None,
        level: float = 0.95,
    ) -> Plan:
        """The best design for a budget, or the cheapest one that reaches a target.

        Args:
            budget: total to spend. The plan gives the narrowest interval it can buy.
            target: wanted half width of the interval, on the score scale (for
                example 0.02 for +-2 points on a 0 to 1 score). The plan gives the
                cheapest design that reaches it.
            generation_cost: cost of one call to the model being tested.
            judge_cost: cost of one call to a judge (0 for exact match or code checks).
            max_items: how many questions exist; the plan never asks for more.
            max_per_facet: the most judges, wordings or runs to consider (default 5).
            limits: per-facet maximums that override ``max_per_facet``, for
                example ``{"judge": 3}`` when only three judges are available.
            level: confidence level for the half width.
        """
        if (budget is None) == (target is None):
            raise ValueError("give either a budget or a target, not both")
        if generation_cost < 0 or judge_cost < 0 or generation_cost + judge_cost <= 0:
            raise ValueError("costs must not be negative, and at least one must be positive")
        z = NormalDist().inv_cdf(0.5 + level / 2)
        item = self.factors[0]
        free = [f for f in self.factors[1:] if f not in self.fixed]
        best = None
        caps = {f: max_per_facet for f in free}
        for key, value in (limits or {}).items():
            name = self._name(key)
            if name == item:
                max_items = value if max_items is None else min(max_items, value)
            elif name in caps:
                caps[name] = int(value)
        for combo in product(*(range(1, caps[f] + 1) for f in free)):
            design = dict(self.sizes)
            design.update(zip(free, combo))
            design[item] = 1
            generation, judges = self._calls(design)
            per_item = generation * generation_cost + judges * judge_cost
            weights = self._weights(design)
            with_item = sum(weights[m] * self._var[m] for m in self._masks if m & 1)
            without = sum(weights[m] * self._var[m] for m in self._masks if not m & 1)
            if budget is not None:
                items = int(budget // per_item)
                if max_items is not None:
                    items = min(items, max_items)
                if items < 2:
                    continue
            else:
                goal = (target / z) ** 2
                if without >= goal:
                    continue
                items = max(2, math.ceil(with_item / (goal - without) - 1e-9))
                if max_items is not None and items > max_items:
                    continue
            variance = without + with_item / items
            cost = items * per_item
            key = (variance, cost) if budget is not None else (cost, variance)
            if best is None or key < best[0]:
                design[item] = items
                best = (key, design, variance, cost)
        if best is None:
            if budget is not None:
                raise ValueError("the budget does not cover even 2 questions at these prices")
            raise ValueError(
                "no design within the limits reaches that target: the judges, wordings or runs themselves "
                "add more noise than the target allows (allow more of them, or loosen the target)"
            )
        _, design, variance, cost = best
        generation, judges = self._calls(design)
        current_generation, current_judges = self._calls(self.sizes)
        goal = f"for {_money(budget)}" if budget is not None else f"for +-{target:g}"
        warnings = []
        for name in free:
            if design[name] < self.sizes[name] and self.sizes[name] <= 5 and name not in self.nested:
                label = self.labels[name]
                warnings.append(
                    f"the pilot had only {self.sizes[name]} {_plural(label, 2)}, so how much {_plural(label, 2)} differ "
                    f"is a rough estimate; dropping to {design[name]} is a bet that it stays small."
                )
        return Plan(
            sizes=design,
            se=math.sqrt(variance),
            level=level,
            cost=cost,
            generation_calls=generation,
            judge_calls=judges,
            current=self.project(level=level),
            current_cost=current_generation * generation_cost + current_judges * judge_cost,
            goal=goal,
            what="difference" if self.compared else "score",
            labels=self.labels,
            warnings=tuple(warnings),
        )

    def report(self, level: float = 0.95, top: int = 6) -> str:
        """A plain-language summary."""
        design = " x ".join(f"{n} {_plural(self.labels[name], n)}" for name, n in self.sizes.items())
        low, high = self.interval(level)
        naive_low, naive_high = self.naive_interval(level)
        if self.compared:
            new, old = self.compared
            head = f"Difference, {new} minus {old}: {self.mean:+.3g} from {design} ({self.count:,} paired scores)."
        else:
            head = f"Score {self.mean:.3g} from {design} ({self.count:,} scores)."
        lines = [
            head,
            f"  Interval counting every source of noise ({level:.0%}): {low:.3g} to {high:.3g} (+-{(high - low) / 2:.3g}).",
            f"  The usual interval, counting only which questions were picked: {naive_low:.3g} to {naive_high:.3g} "
            f"(+-{(naive_high - naive_low) / 2:.3g}).",
        ]
        if self.compared:
            new, old = self.compared
            if low > 0:
                lines.append(f"  {new} is reliably better than {old}, even counting every source of noise.")
            elif high < 0:
                lines.append(f"  {new} is reliably worse than {old}, even counting every source of noise.")
            elif naive_low > 0 or naive_high < 0:
                lines.append(
                    "  The usual interval would call this a real difference, but with the judge, wording and run "
                    "noise counted it could be noise."
                )
            else:
                lines.append("  The difference could be noise.")
        if self.df < 10:
            lines.append(
                f"  The noise estimate itself rests on few judges, wordings or runs (about {self.df:.0f} degrees "
                "of freedom), which widens the interval. More of them in the pilot would firm it up."
            )
        if self.fixed:
            lines.append(f"  Treated as fixed (always the same, not a sample): {', '.join(self.fixed)}.")
        lines.append(f"Where the noise in this {'difference' if self.compared else 'score'} comes from:")
        width = max(len(c.name) for c in self.components)
        for c in self.components[:top]:
            if c.share < 0.005 and c is not self.components[0]:
                continue
            lines.append(f"  {c.name:<{width}}  {c.share:5.0%}  {c.description}")
        for c in self.components[:2]:
            if len(c.factors) == 1 and c.factors[0] in self.means_by and c.share >= 0.2:
                sign = "+" if self.compared else ""
                shown = ", ".join(f"{level} {value:{sign}.3g}" for level, value in self.means_by[c.factors[0]].items())
                lines.append(f"  By {c.factors[0]}: {shown}")
        lines.extend(f"Note: {note}" for note in self.notes)
        if self.dropped:
            shown = ", ".join(self.dropped[:5]) + (" and others" if len(self.dropped) > 5 else "")
            lines.append(f"Note: left out {len(self.dropped)} questions with missing scores ({shown}).")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        what = f"difference {self.compared[0]} - {self.compared[1]}" if self.compared else "score"
        return f"<Study {what} {self.mean:.3g} +-{self.se:.2g} (usual +-{self.naive_se:.2g}), sizes {self.sizes}>"


def study(
    rows,
    item: str = "item",
    score: str = "score",
    facets: Optional[Sequence[str]] = None,
    fixed: Iterable[str] = (),
    judging: Optional[Iterable[str]] = None,
    nested: Optional[Iterable[str]] = None,
) -> Study:
    """Analyse a pilot where every question was scored under every combination of facets.

    Args:
        rows: one row per score, as dicts or a pandas DataFrame, with a
            column for the question, one for each facet (judge, prompt
            wording, run, ...) and one for the score.
        item: column naming the question.
        score: column with the score: numbers, booleans, or words like pass and fail.
        facets: columns to treat as sources of noise. By default every other column.
        fixed: facets that are not a sample, for example a judge you will always
            use. Their own effect is then part of what you are measuring.
        judging: facets that cost a judge call rather than a model call, for
            budget planning. Found by name (judge, grader, rater...) if left out.
        nested: facets whose labels mean nothing across questions, like run 1
            and run 2. Found by name (run, seed, sample, attempt...) if left out.
            Pass ``[]`` if run 1 really is the same for every question (for
            example, all run-1 answers came from one batch on one day).

    Questions missing any combination are left out and reported.

    >>> rows = [{"item": i, "judge": j, "score": (i + j) % 3 > 0} for i in range(6) for j in range(2)]
    >>> round(study(rows).mean, 3)
    0.667
    """
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    columns = list(rows[0].keys())
    for needed in (item, score):
        if needed not in columns:
            raise ValueError(f"no {needed!r} column; the columns are {columns}")
    if facets is None:
        facets = [c for c in columns if c not in (item, score)]
    facets = list(facets)
    return _build(
        ((r[item], tuple(r[f] for f in facets), to_score(r[score])) for r in rows),
        item, facets, fixed, judging, nested=nested,
    )


PAIRS = (("new", "old"), ("candidate", "baseline"), ("treatment", "control"), ("after", "before"), ("b", "a"))


def pick_pair(names: Sequence[str], new: Optional[str], old: Optional[str]) -> Tuple[str, str]:
    """Work out which system is new and which is old."""
    names = [str(n) for n in names]
    if new is not None and old is not None:
        for label in (new, old):
            if str(label) not in names:
                raise ValueError(f"no scores for {label!r}; the systems are {names}")
        return str(new), str(old)
    if len(names) == 2:
        lowered = {n.lower(): n for n in names}
        for a, b in PAIRS:
            if a in lowered and b in lowered:
                return lowered[a], lowered[b]
        if new is not None or old is not None:
            given = str(new if new is not None else old)
            other = next(n for n in names if n != given)
            return (given, other) if new is not None else (other, given)
    raise ValueError(f"say which system is new and which is old; the systems are {names}")


def compare(
    rows,
    system: str = "system",
    new: Optional[str] = None,
    old: Optional[str] = None,
    item: str = "item",
    score: str = "score",
    facets: Optional[Sequence[str]] = None,
    fixed: Iterable[str] = (),
    judging: Optional[Iterable[str]] = None,
    nested: Optional[Iterable[str]] = None,
) -> Study:
    """Compare two systems (models or prompts) scored on the same questions, judges, wordings and runs.

    Returns a :class:`Study` of the difference, new minus old. Noise that hits
    both systems alike, such as a strict judge, cancels out; what is left is
    noise in the gap itself, such as a judge that favours one system's style.
    Runs are paired by their label, so run 1 of one system is compared with
    run 1 of the other.

    >>> rows = [{"system": s, "item": i, "judge": j, "score": (i + j + (s == "new")) % 3 > 0}
    ...         for s in ("new", "old") for i in range(6) for j in range(2)]
    >>> round(compare(rows).mean, 3)
    0.0
    """
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    columns = list(rows[0].keys())
    for needed in (system, item, score):
        if needed not in columns:
            raise ValueError(f"no {needed!r} column; the columns are {columns}")
    if facets is None:
        facets = [c for c in columns if c not in (system, item, score)]
    facets = list(facets)
    systems = list(dict.fromkeys(str(r[system]) for r in rows))
    new, old = pick_pair(systems, new, old)
    return _build_difference(
        ((str(r[system]), r[item], tuple(r[f] for f in facets), to_score(r[score])) for r in rows),
        new, old, item, facets, fixed, judging, nested,
    )


def _build_difference(records, new, old, item, facets, fixed=(), judging=None, nested=None) -> Study:
    """records: (system, question, facet levels, score)."""
    scores: Dict[str, Dict[Tuple, float]] = {new: {}, old: {}}
    order: List[Tuple] = []
    for name, question, facet_levels, value in records:
        if name not in scores:
            continue
        key = (str(question),) + tuple(str(v) for v in facet_levels)
        if key in scores[name]:
            label = ", ".join(f"{n} {v}" for n, v in zip([item] + list(facets), key))
            raise ValueError(f"two scores for {name}, {label}. If these are repeated runs, add a run column.")
        scores[name][key] = value
        if key not in scores[old if name == new else new]:
            order.append(key)
    pairs = [(k[0], k[1:], scores[new][k] - scores[old][k]) for k in order if k in scores[new] and k in scores[old]]
    unpaired = {k[0] for k in order if not (k in scores[new] and k in scores[old])}
    study_ = _build(pairs, item, facets, fixed, judging, compared=(new, old), nested=nested)
    if unpaired:
        study_.dropped = sorted(set(study_.dropped) | unpaired)
    return study_


def _build(records, item, facets, fixed=(), judging=None, compared=None, nested=None) -> Study:
    """records: (question, facet levels, score) triples."""
    levels: List[Dict[str, int]] = [{} for _ in range(len(facets) + 1)]
    seen: Dict[Tuple, float] = {}
    for question, facet_levels, value in records:
        key = (str(question),) + tuple(str(v) for v in facet_levels)
        if key in seen:
            label = ", ".join(f"{name} {v}" for name, v in zip([item] + list(facets), key))
            hint = "" if any(_kind(f) == "run" for f in facets) else " If these are repeated runs, add a run column."
            raise ValueError(f"two scores for {label}.{hint}")
        seen[key] = value
        for i, v in enumerate(key):
            levels[i].setdefault(v, len(levels[i]))

    notes = []
    keep = [0]
    for i, name in enumerate(facets, start=1):
        if len(levels[i]) < 2:
            only = next(iter(levels[i]), "?")
            notes.append(
                f"only one {name} ({only}), so noise from changing the {name} cannot be measured and is not counted."
            )
        else:
            keep.append(i)
    fixed = [f for f in fixed if f in [facets[i - 1] for i in keep]]

    grid = [list(levels[i]) for i in keep]
    expected = math.prod(len(g) for g in grid[1:])
    by_item: Dict[str, Dict[Tuple[str, ...], float]] = {}
    for key, value in seen.items():
        by_item.setdefault(key[0], {})[tuple(key[i] for i in keep[1:])] = value
    complete = [q for q in grid[0] if len(by_item.get(q, {})) == expected]
    dropped = [q for q in grid[0] if q not in complete]
    if len(complete) < 2:
        raise ValueError(
            "fewer than 2 questions have a score for every combination of "
            + ", ".join(facets[i - 1] for i in keep[1:])
            + "; the pilot must score each question under every combination"
        )
    used_levels = [complete] + [
        [lv for lv in grid[j] if any(k[j - 1] == lv for q in complete for k in by_item[q])] for j in range(1, len(keep))
    ]
    index = [{lv: n for n, lv in enumerate(lvls)} for lvls in used_levels]
    cells = {}
    for q in complete:
        for combo, value in by_item[q].items():
            cells[(index[0][q],) + tuple(index[j + 1][c] for j, c in enumerate(combo))] = value
    names = [item] + [facets[i - 1] for i in keep[1:]]
    judging_names = None if judging is None else [j for j in judging if j in names]
    nested_names = None if nested is None else [f for f in nested if f in names]
    return Study(
        cells, names, used_levels, fixed=fixed, judging=judging_names, notes=notes, dropped=dropped,
        compared=compared, nested=nested_names,
    )
