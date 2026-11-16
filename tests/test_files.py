import csv
import json

import pytest

import evalnoise as en


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_example_pilot_is_read_with_every_column_found():
    table = en.read_scores("examples/pilot.csv")
    assert (table.item, table.score, table.facets, table.system) == ("question_id", "verdict", ["judge", "prompt", "run"], None)
    s = table.study()
    assert s.sizes == {"question_id": 60, "judge": 3, "prompt": 3, "run": 2}
    assert s.mean == pytest.approx(sum(r["verdict"] == "pass" for r in csv.DictReader(open("examples/pilot.csv"))) / 1080)


def test_example_comparison_finds_the_systems():
    table = en.read_scores("examples/two_models.csv")
    assert table.system == "model" and table.systems == ["baseline", "candidate"]
    diff = table.compare()
    assert diff.compared == ("candidate", "baseline")
    with pytest.raises(ValueError, match="several systems"):
        table.study()
    assert table.study(system="candidate").sizes["judge"] == 4


def test_jsonl_with_nested_fields(tmp_path):
    rows = [
        {"id": f"q{i}", "metadata": {"grader": g, "seed": s}, "result": {"passed": (i + s) % 3 > 0}}
        for i in range(6) for g in ("g1", "g2") for s in (1, 2)
    ]
    path = write(tmp_path, "log.jsonl", "\n".join(json.dumps(r) for r in rows))
    table = en.read_scores(path)
    assert table.item == "id"
    assert table.score == "result.passed"
    assert table.facets == ["metadata.grader", "metadata.seed"]
    s = table.study()
    assert s.judging == ("metadata.grader",)
    assert s.nested == ("metadata.seed",)


def test_json_list_and_text_columns_are_not_facets(tmp_path):
    rows = [
        {"question": f"q{i}", "prompt": f"What is {i} + {i}?", "judge": j, "score": (i + len(j)) % 2}
        for i in range(30) for j in ("a", "bb")
    ]
    path = write(tmp_path, "rows.json", json.dumps({"results": rows}))
    table = en.read_scores(path)
    assert table.facets == ["judge"]
    assert "looks like text or an id" in table.report()


@pytest.mark.parametrize(
    "content, kwargs, message",
    [
        ("judge,score\na,1\n", {}, "no column naming the question"),
        ("item,judge\n1,a\n", {}, "no score column"),
        ("item,judge,score\n1,a,1\n1,,0\n", {}, "row 3: no value for judge"),
        ("item,judge,score\n1,a,1\n", {"facets": ["grader"]}, "no column 'grader'"),
        ("item,judge,score\n1,a,1\n2,a,1\n", {}, "comparing needs"),
    ],
)
def test_clear_errors(tmp_path, content, kwargs, message):
    path = write(tmp_path, "scores.csv", content)
    with pytest.raises(ValueError, match=message):
        table = en.read_scores(path, **kwargs)
        table.compare()
