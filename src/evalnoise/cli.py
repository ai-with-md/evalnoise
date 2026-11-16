"""Command line: ``evalnoise check``, ``evalnoise plan`` and ``evalnoise project``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .files import read_scores


def _study(args: argparse.Namespace):
    table = read_scores(args.file, item=args.item, score=args.score, facets=args.facets, system=args.system)
    if args.verbose:
        print(table.report(), file=sys.stderr)
    nested = [] if args.runs_crossed else None
    if table.system is not None and len(table.systems) > 1 and args.only is None:
        return table.compare(args.new, args.old, fixed=args.fixed, judging=args.judging, nested=nested)
    return table.study(fixed=args.fixed, judging=args.judging, system=args.only, nested=nested)


def _cmd_check(args: argparse.Namespace) -> str:
    return _study(args).report(level=args.level)


def _limits(pairs: List[str]) -> dict:
    limits = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        try:
            limits[name.strip()] = int(value)
        except ValueError:
            raise ValueError(f"--limit takes name=number, like judge=3 (got {pair!r})") from None
    return limits


def _cmd_plan(args: argparse.Namespace) -> str:
    if (args.budget is None) == (args.target is None):
        raise ValueError("give either --budget or --target")
    found = _study(args)
    plan = found.plan(
        budget=args.budget,
        target=args.target,
        generation_cost=args.model_cost,
        judge_cost=args.judge_cost,
        max_items=args.max_items,
        max_per_facet=args.max_per_facet,
        limits=_limits(args.limit),
        level=args.level,
    )
    return found.report(level=args.level) + "\n\n" + plan.summary()


def _cmd_project(args: argparse.Namespace) -> str:
    found = _study(args)
    sizes = _limits(args.design) if args.design else {}
    return found.project(level=args.level, **sizes).summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalnoise",
        description="Where does the noise in an eval score come from, and where should the next calls go?",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("file", help="CSV, JSON or JSON Lines with one row per score")
        p.add_argument("--item", help="column naming the question (found automatically if left out)")
        p.add_argument("--score", help="column with the score (found automatically if left out)")
        p.add_argument("--facets", nargs="+", help="columns to count as sources of noise, like judge prompt run")
        p.add_argument("--system", help="column naming the model or version, to compare two of them")
        p.add_argument("--new", help="the new system, when comparing")
        p.add_argument("--old", help="the old system, when comparing")
        p.add_argument("--only", help="analyse just this one system, when the file has several")
        p.add_argument("--fixed", nargs="+", default=(), help="facets that are not a sample, like a judge you will always use")
        p.add_argument("--judging", nargs="+", help="facets that cost a judge call (found by name if left out)")
        p.add_argument(
            "--runs-crossed", action="store_true",
            help="run 1 means the same thing for every question (for example one batch per run)",
        )
        p.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
        p.add_argument("-v", "--verbose", action="store_true", help="say which columns were used")

    p = sub.add_parser("check", help="split the noise in a score, or in the difference between two systems")
    common(p)
    p.set_defaults(run=_cmd_check)

    p = sub.add_parser("plan", help="the best design for a budget, or the cheapest for a target")
    common(p)
    p.add_argument("--budget", type=float, help="total to spend")
    p.add_argument("--target", type=float, help="wanted half width of the interval, like 0.02")
    p.add_argument("--model-cost", type=float, default=1.0, help="cost of one call to the model being tested")
    p.add_argument("--judge-cost", type=float, default=0.0, help="cost of one judge call (default 0)")
    p.add_argument("--max-items", type=int, help="how many questions exist")
    p.add_argument("--max-per-facet", type=int, default=5, help="most judges, wordings or runs to consider (default 5)")
    p.add_argument("--limit", nargs="+", default=[], metavar="NAME=N", help="per-facet maximums, like judge=3")
    p.set_defaults(run=_cmd_plan)

    p = sub.add_parser("project", help="expected noise for another design, like items=500 judges=1")
    common(p)
    p.add_argument("design", nargs="*", metavar="NAME=N", help="sizes to change, like items=500 judges=1")
    p.set_defaults(run=_cmd_project)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(args.run(args))
    except (ValueError, OSError) as exc:
        print(f"evalnoise: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
