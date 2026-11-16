"""Read eval results from CSV, JSON or JSON Lines, finding the columns automatically."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .study import Study, _build, _build_difference, _kind, pick_pair, to_score

ITEM_NAMES = (
    "item", "itemid", "question", "questionid", "example", "exampleid", "case", "caseid", "testcase",
    "testcaseid", "task", "taskid", "problem", "problemid", "qid", "id", "inputid", "datasetitemid",
)
SCORE_NAMES = (
    "score", "judgescore", "grade", "rating", "verdict", "judgeverdict", "passed", "pass", "correct",
    "iscorrect", "result", "outcome", "label", "success", "value",
)
SYSTEM_NAMES = ("system", "model", "modelname", "version", "arm", "group", "candidate", "setup")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _flatten(record: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def _read_records(path: str) -> List[dict]:
    path = os.fspath(path)
    lower = path.lower()
    with open(path, encoding="utf-8-sig") as handle:
        if lower.endswith(".csv"):
            return list(csv.DictReader(handle))
        if lower.endswith(".tsv"):
            return list(csv.DictReader(handle, delimiter="\t"))
        text = handle.read()
    stripped = text.lstrip()
    if lower.endswith((".jsonl", ".ndjson")) or (stripped.startswith("{") and "\n{" in stripped):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        records = data
    if not all(isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: expected one object per row")
    return [_flatten(r) for r in records]


def _find(columns: Sequence[str], names: Sequence[str], taken: Iterable[str] = ()) -> Optional[str]:
    taken = set(taken)
    by_norm: Dict[str, str] = {}
    for c in columns:
        if c not in taken:
            by_norm.setdefault(_norm(c.split(".")[-1]), c)
            by_norm.setdefault(_norm(c), c)
    for name in names:
        if name in by_norm:
            return by_norm[name]
    return None


@dataclass
class ScoreTable:
    """Scores read from a file, with the columns that were used.

    Use :meth:`study` for one system, or :meth:`compare` for two.
    """

    rows: List[dict]
    item: str
    score: str
    facets: List[str]
    system: Optional[str] = None
    ignored: List[str] = field(default_factory=list)
    path: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def systems(self) -> List[str]:
        if self.system is None:
            return []
        return list(dict.fromkeys(str(r[self.system]) for r in self.rows))

    def report(self) -> str:
        parts = [f"question from {self.item!r}", f"score from {self.score!r}"]
        parts += [f"{_kind(f) if _kind(f) != 'other' else 'facet'} from {f!r}" for f in self.facets]
        if self.system:
            parts.append(f"system from {self.system!r} ({', '.join(self.systems)})")
        lines = [f"Read {len(self.rows):,} rows: " + ", ".join(parts) + "."]
        if self.ignored:
            lines.append(f"Not used: {', '.join(self.ignored)}. Name them with facets=[...] to count them as sources of noise.")
        lines.extend(f"Note: {n}." for n in self.notes)
        return "\n".join(lines)

    def _records(self, rows):
        for line, r in enumerate(rows, start=2):
            try:
                value = to_score(r[self.score])
            except ValueError as exc:
                raise ValueError(f"{self.path}, row {line}: {exc}") from None
            yield r, value

    def study(
        self,
        fixed: Iterable[str] = (),
        judging: Optional[Iterable[str]] = None,
        system: Optional[str] = None,
        nested: Optional[Iterable[str]] = None,
    ) -> Study:
        """Analyse one system. If the file has several, name one with ``system``."""
        rows = self.rows
        if self.system is not None and len(self.systems) > 1:
            if system is None:
                raise ValueError(
                    f"the file has several systems ({', '.join(self.systems)}): pick one with system=..., "
                    "or compare two with compare()"
                )
            rows = [r for r in rows if str(r[self.system]) == str(system)]
            if not rows:
                raise ValueError(f"no rows for system {system!r}; the systems are {self.systems}")
        return _build(
            ((r[self.item], tuple(r[f] for f in self.facets), v) for r, v in self._records(rows)),
            self.item, self.facets, fixed, judging, nested=nested,
        )

    def compare(
        self,
        new: Optional[str] = None,
        old: Optional[str] = None,
        fixed: Iterable[str] = (),
        judging: Optional[Iterable[str]] = None,
        nested: Optional[Iterable[str]] = None,
    ) -> Study:
        """Analyse the difference between two systems scored on the same questions, new minus old."""
        if self.system is None or len(self.systems) < 2:
            raise ValueError("comparing needs a column naming the system (model or version) with at least two values")
        new, old = pick_pair(self.systems, new, old)
        return _build_difference(
            ((str(r[self.system]), r[self.item], tuple(r[f] for f in self.facets), v) for r, v in self._records(self.rows)),
            new, old, self.item, self.facets, fixed, judging, nested,
        )


def read_scores(
    path,
    item: Optional[str] = None,
    score: Optional[str] = None,
    facets: Optional[Sequence[str]] = None,
    system: Optional[str] = None,
) -> ScoreTable:
    """Read eval scores from a CSV, JSON or JSON Lines file.

    One row per score, with columns for the question, the score, and the
    facets that varied: judge, prompt wording, run, and so on. Columns are
    found by their names when not given; nested JSON fields are named with
    dots, like ``metadata.judge``.

    Args:
        path: the file.
        item: column naming the question.
        score: column with the score (numbers, true/false, pass/fail...).
        facets: columns to count as sources of noise. By default, columns
            whose names say judge, grader, prompt, template, run, seed, sample
            and similar.
        system: column naming the model or version, for comparisons.
    """
    records = _read_records(path)
    if not records:
        raise ValueError(f"{path} has no rows")
    columns = list(dict.fromkeys(c for r in records for c in r))
    for given in (item, score, system, *(facets or [])):
        if given is not None and given not in columns:
            raise ValueError(f"{path} has no column {given!r}; the columns are {columns}")
    item = item or _find(columns, ITEM_NAMES)
    if item is None:
        raise ValueError(f"{path}: no column naming the question (like item, question or id); name it with item=...")
    score = score or _find(columns, SCORE_NAMES, taken=[item])
    if score is None:
        score = next((c for c in columns if _norm(c).endswith("score") and c != item), None)
    if score is None:
        raise ValueError(f"{path}: no score column (like score, verdict or passed); name it with score=...")
    if system is None:
        system = _find(columns, SYSTEM_NAMES, taken=[item, score])
        if system is not None and len({str(r.get(system)) for r in records}) < 2:
            system = None
    taken = {item, score, system}
    notes = []
    if facets is None:
        questions = len({str(r.get(item)) for r in records})
        facets = []
        for c in columns:
            if c in taken or _kind(c.split(".")[-1]) == "other":
                continue
            distinct = len({str(r.get(c)) for r in records})
            if distinct > 20 or (questions >= 10 and distinct >= questions):
                notes.append(f"{c!r} has {distinct:,} different values, so it looks like text or an id, not a facet")
            else:
                facets.append(c)
    facets = list(facets)
    ignored = [c for c in columns if c not in taken and c not in facets]
    needed = [c for c in [item, score, system, *facets] if c]
    for line, r in enumerate(records, start=2):
        missing = [c for c in needed if r.get(c) in (None, "")]
        if missing:
            raise ValueError(f"{path}, row {line}: no value for {', '.join(missing)}")
    return ScoreTable(records, item, score, facets, system, ignored, os.fspath(path), notes)
