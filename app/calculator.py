import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="evalnoise: where does the noise in my eval score come from?",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _(mo):
    import sys
    import tempfile
    from pathlib import Path

    if sys.platform == "emscripten":
        # In the browser, install the library from the wheel published with this page.
        # wheel.txt holds the wheel's file name, so a new version needs no change here.
        import micropip
        from pyodide.http import pyfetch

        _public = mo.notebook_location() / "public"
        _wheel = (await (await pyfetch(str(_public / "wheel.txt"))).string()).strip()
        await micropip.install(str(_public / _wheel))
        EXAMPLES = {
            _name: await (await pyfetch(str(_public / _name))).string()
            for _name in ("pilot.csv", "two_models.csv")
        }
    else:
        _folder = Path(mo.notebook_location()).parent / "examples"
        EXAMPLES = {_name: (_folder / _name).read_text() for _name in ("pilot.csv", "two_models.csv")}
    import evalnoise as en
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7, 3.2),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    INK, ACCENT, MUTED, BAD = "#1f2937", "#b45309", "#9ca3af", "#b91c1c"

    def read_text(text, suffix):
        """read_scores works on files, so write the text to a temporary one."""
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as handle:
            handle.write(text)
        return en.read_scores(handle.name)

    return ACCENT, EXAMPLES, MUTED, en, plt, read_text


@app.cell
def _(mo):
    mo.md("""
    # Where does the noise in my eval score come from?

    An eval score moves when you change the questions, but also when you swap the judge, reword the
    prompt, or simply run it again. The usual error bar only counts the questions. Give this page a
    small pilot, where every question was scored under every judge, wording and run, and it shows
    how much each source adds, an error bar that counts all of them, and where the next API calls
    do the most good. Everything runs in your browser, so nothing is sent anywhere. The first load
    takes a few seconds while Python starts up.

    1. [Where does the noise come from?](#noise)
    2. [Where should the next calls go?](#plan)
    3. [What would another design give?](#whatif)

    The same calculations are in the Python library: `pip install evalnoise`.
    """)
    return


@app.cell
def _(mo):
    source = mo.ui.dropdown(
        {
            "Example: one model, 3 judges, 3 wordings, 2 runs": "pilot.csv",
            "Example: two models compared by 4 judges": "two_models.csv",
            "My file": "upload",
        },
        value="Example: one model, 3 judges, 3 wordings, 2 runs",
        label="Scores",
    )
    upload = mo.ui.file(filetypes=[".csv", ".json", ".jsonl"], label="Upload a CSV, JSON or JSON Lines file")
    return source, upload


@app.cell
def _(EXAMPLES, mo, read_text, source, upload):
    table, problem = None, None
    try:
        if source.value == "upload":
            if upload.value:
                _file = upload.value[0]
                _suffix = "." + _file.name.rsplit(".", 1)[-1].lower()
                table = read_text(_file.contents.decode("utf-8-sig"), _suffix)
            else:
                problem = "Upload a file with one row per score: a question column, a score column, and a column for each judge, wording or run."
        else:
            table = read_text(EXAMPLES[source.value], ".csv")
    except (ValueError, UnicodeDecodeError) as _exc:
        problem = f"Could not read the file: {_exc}"
    facets_found = list(table.facets) if table is not None else []
    fixed_pick = mo.ui.multiselect(facets_found, label="Always the same, not a sample (fixed)")
    return fixed_pick, problem, table


@app.cell
def _(ACCENT, MUTED, fixed_pick, mo, plt, problem, source, table, upload):
    _header = mo.md(
        """
        <h2 id="noise">1. Where does the noise come from?</h2>

        One row per score, with a column for the question, the score (a number, or pass/fail), and each
        judge, wording or run. If the file has a column naming two models, they are compared.
        Mark a facet as fixed if you will always use exactly those, like the one judge your team grades with.
        """
    )
    result = None
    _parts = [_header, mo.hstack([source, upload] if source.value == "upload" else [source], justify="start")]
    if problem:
        _parts.append(mo.callout(mo.md(problem), kind="warn"))
    else:
        try:
            if table.system is not None and len(table.systems) > 1:
                result = table.compare(fixed=fixed_pick.value)
            else:
                result = table.study(fixed=fixed_pick.value)
        except ValueError as _exc:
            _parts.append(mo.callout(mo.md(f"Could not analyse the scores: {_exc}"), kind="warn"))
    if result is not None:
        _low, _high = result.interval()
        _nlow, _nhigh = result.naive_interval()
        _what = "Difference" if result.compared else "Score"
        _sign = "+" if result.compared else ""
        _design = " x ".join(f"{n:,} {result.labels[k]}{'s' if n > 1 and not result.labels[k].endswith('s') else ''}" for k, n in result.sizes.items())
        _cards = mo.hstack(
            [
                mo.stat(f"{result.mean:{_sign}.3g}", label=_what if not result.compared else f"{result.compared[0]} minus {result.compared[1]}", caption=_design, bordered=True),
                mo.stat(f"±{(_high - _low) / 2:.3g}", label="Counting every source", caption=f"{_low:{_sign}.3g} to {_high:{_sign}.3g}, 95%", bordered=True),
                mo.stat(f"±{(_nhigh - _nlow) / 2:.3g}", label="The usual error bar", caption="counts only the questions", bordered=True),
            ],
            justify="start",
            wrap=True,
        )
        _shown = [c for c in result.components if c.share >= 0.005][:7][::-1]
        _fig, _ax = plt.subplots(figsize=(7, 0.45 * len(_shown) + 0.8))
        _ax.barh([c.name for c in _shown], [c.share for c in _shown], color=[ACCENT if c is result.components[0] else MUTED for c in _shown])
        for _i, _c in enumerate(_shown):
            _ax.text(_c.share + 0.01, _i, f"{_c.share:.0%}", va="center", fontsize=9)
        _ax.set_xlim(0, max(c.share for c in _shown) * 1.18)
        _ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        _ax.set_xlabel(f"Share of the noise in the {_what.lower()}")
        _ax.grid(axis="y", visible=False)
        _fig.tight_layout()
        _top = result.components[0]
        _lines = [f"**Biggest source: {_top.name}.** {_top.description[0].upper() + _top.description[1:]}."]
        for _c in result.components[:2]:
            if len(_c.factors) == 1 and _c.factors[0] in result.means_by and _c.share >= 0.2:
                _lines.append(
                    f"By {_c.factors[0]}: "
                    + ", ".join(f"{lv} {v:{_sign}.3g}" for lv, v in result.means_by[_c.factors[0]].items())
                    + "."
                )
        if result.compared:
            _new, _old = result.compared
            if _low > 0 or _high < 0:
                _lines.append(f"**{_new} is reliably {'better' if _low > 0 else 'worse'} than {_old}**, even counting every source of noise.")
            elif _nlow > 0 or _nhigh < 0:
                _lines.append("**The usual error bar would call this a real difference.** With the judge, wording and run noise counted, it could be noise.")
            else:
                _lines.append("**The difference could be noise.**")
        if result.df < 10:
            _lines.append(f"The noise estimate rests on few judges, wordings or runs (about {result.df:.0f} degrees of freedom), which widens the interval.")
        _lines += [f"Note: {n}" for n in result.notes]
        if result.dropped:
            _lines.append(f"Left out {len(result.dropped)} questions with missing scores.")
        _parts += [
            _cards,
            fixed_pick if table.facets else mo.md(""),
            _fig,
            mo.md("\n\n".join(_lines)),
            mo.accordion({"Full report": mo.md(f"```text\n{table.report()}\n\n{result.report()}\n```")}),
        ]
    mo.vstack(_parts)
    return (result,)


@app.cell
def _(mo):
    pl_budget = mo.ui.number(1, 1_000_000, step=1, value=40, label="Budget ($)")
    pl_model = mo.ui.number(0, 10, step=0.0005, value=0.01, label="Cost per call to the model being tested ($)")
    pl_judge = mo.ui.number(0, 10, step=0.0005, value=0.004, label="Cost per judge call ($)")
    pl_max = mo.ui.slider(1, 10, step=1, value=3, label="Most judges, wordings or runs to use", show_value=True)
    pl_items = mo.ui.number(0, 10_000_000, step=10, value=0, label="Questions available (0 for no limit)")
    return pl_budget, pl_items, pl_judge, pl_max, pl_model


@app.cell
def _(ACCENT, MUTED, mo, pl_budget, pl_items, pl_judge, pl_max, pl_model, plt, result):
    _header = mo.md(
        """
        <h2 id="plan">2. Where should the next calls go?</h2>

        Using the noise found above, the best mix of questions, judges, wordings and runs for a budget.
        A judge call is charged for every answer each judge grades; a model call for every answer generated.
        """
    )
    _inputs = mo.vstack([mo.hstack([pl_budget, pl_model, pl_judge], wrap=True), mo.hstack([pl_max, pl_items], wrap=True)])
    if result is None:
        _body = mo.md("Load scores in section 1 first.")
    else:
        _kw = dict(
            generation_cost=pl_model.value, judge_cost=pl_judge.value, max_per_facet=pl_max.value,
            max_items=pl_items.value or None,
        )
        try:
            _plan = result.plan(budget=pl_budget.value, **_kw)
            _design = ", ".join(
                f"{n:,} {result.labels[k]}{'s' if n > 1 and not result.labels[k].endswith('s') else ''}"
                for k, n in _plan.sizes.items()
            )
            _cards = mo.hstack(
                [
                    mo.stat(f"±{_plan.half_width:.3g}", label="Best you can buy", caption=f"95%, for ${pl_budget.value:,.0f}", bordered=True),
                    mo.stat(f"±{_plan.current.half_width:.3g}", label="The pilot's own design", caption=f"costs ${_plan.current_cost:,.2f}", bordered=True),
                    mo.stat(f"{_plan.generation_calls:,} + {_plan.judge_calls:,}", label="Model + judge calls", caption=f"${_plan.cost:,.2f} in all", bordered=True),
                ],
                justify="start",
                wrap=True,
            )
            _budgets = [pl_budget.value * f for f in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4)]
            _best, _more_q = [], []
            _pilot_gen, _pilot_judge = result._calls({**result.sizes, result.factors[0]: 1})
            _per_q = _pilot_gen * pl_model.value + _pilot_judge * pl_judge.value
            for _b in _budgets:
                try:
                    _best.append(result.plan(budget=_b, **_kw).half_width)
                except ValueError:
                    _best.append(float("nan"))
                _n = int(_b // _per_q) if _per_q > 0 else 0
                if pl_items.value:
                    _n = min(_n, pl_items.value)
                _more_q.append(result.project(**{result.factors[0]: _n}).half_width if _n >= 2 else float("nan"))
            _fig, _ax = plt.subplots()
            _ax.plot(_budgets, _best, color=ACCENT, marker="o", label="Best mix for each budget")
            _ax.plot(_budgets, _more_q, color=MUTED, marker="o", label="The pilot's mix, just more questions")
            _ax.set_xlabel("Budget ($)")
            _ax.set_ylabel("Error bar (±, 95%)")
            _ax.set_ylim(bottom=0)
            _ax.legend(frameon=False)
            _fig.tight_layout()
            _notes = [f"**Best design: {_design}.**"] + [f"Careful: {w}" for w in _plan.warnings]
            _bigger = result.plan(budget=pl_budget.value * 10, **_kw)
            if _bigger.half_width > 0.85 * _plan.half_width:
                _capped = [
                    result.labels[f] for f in result.factors[1:]
                    if f not in result.fixed and _plan.sizes[f] == pl_max.value
                ]
                _which = " and ".join(f"{c}s" for c in _capped) or "judges, wordings or runs"
                _notes.append(
                    f"More money barely helps: ten times the budget only gets to ±{_bigger.half_width:.3g}. "
                    f"What is left comes from how much the {_which} differ, and adding questions cannot "
                    f"average that out. Allow more {_which}, or mark them as fixed in section 1 if you will "
                    "always use the same ones."
                )
            _body = mo.vstack([_cards, mo.md("\n\n".join(_notes)), _fig])
        except ValueError as _exc:
            _body = mo.callout(mo.md(str(_exc)), kind="warn")
    mo.vstack([_header, _inputs, _body])
    return


@app.cell
def _(mo, result):
    if result is None:
        wi = None
    else:
        _sliders = {}
        for _name in result.factors:
            _label = result.labels[_name]
            if _name == result.factors[0]:
                _sliders[_name] = mo.ui.slider(10, 3000, step=10, value=min(3000, max(10, round(result.sizes[_name], -1))), label="Questions", show_value=True)
            elif _name not in result.fixed:
                _sliders[_name] = mo.ui.slider(1, 10, step=1, value=result.sizes[_name], label=_label[:1].upper() + _label[1:] + "s", show_value=True)
        wi = mo.ui.dictionary(_sliders)
    return (wi,)


@app.cell
def _(mo, result, wi):
    _header = mo.md(
        """
        <h2 id="whatif">3. What would another design give?</h2>
        """
    )
    if result is None or wi is None:
        _body = mo.md("Load scores in section 1 first.")
    else:
        _p = result.project(**wi.value)
        _pilot = result.project()
        _body = mo.vstack(
            [
                mo.hstack(list(wi.values()), justify="start", wrap=True),
                mo.hstack(
                    [
                        mo.stat(f"±{_p.half_width:.3g}", label="This design", caption="95% error bar", bordered=True),
                        mo.stat(f"±{_pilot.half_width:.3g}", label="The pilot's design", caption="on the same basis", bordered=True),
                    ],
                    justify="start",
                ),
            ]
        )
    mo.vstack([_header, _body])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    Built with [evalnoise](https://github.com/ai-with-md/evalnoise), MIT licensed. The method is
    generalizability theory: variance components from a balanced random-effects analysis of variance,
    with runs nested within question and wording, and an interval with Satterthwaite degrees of freedom.
    Projections and plans use a normal approximation. Every promise is checked against simulation in the test suite.
    """)
    return


if __name__ == "__main__":
    app.run()
