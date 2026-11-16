from evalnoise.cli import main


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_check_a_pilot(capsys):
    code, out, _ = run(capsys, "check", "examples/pilot.csv")
    assert code == 0
    assert out.startswith("Score 0.713 from 60 questions x 3 judges x 3 prompts x 2 runs")
    assert "judge" in out.splitlines()[5]


def test_check_compares_when_the_file_has_two_systems(capsys):
    code, out, err = run(capsys, "check", "examples/two_models.csv", "-v")
    assert code == 0
    assert "system from 'model'" in err
    assert "Difference, candidate minus baseline" in out
    assert "could be noise" in out
    code, out, _ = run(capsys, "check", "examples/two_models.csv", "--only", "baseline")
    assert out.startswith("Score")


def test_plan_and_project(capsys):
    code, out, _ = run(
        capsys, "plan", "examples/pilot.csv", "--budget", "40", "--model-cost", "0.01", "--judge-cost", "0.004",
        "--limit", "judge=3", "prompt=3",
    )
    assert code == 0
    assert "Best design for $40.00" in out
    assert "3 judges, 3 prompts" in out
    code, out, _ = run(capsys, "project", "examples/pilot.csv", "items=500", "judges=1")
    assert out.startswith("500 questions x 1 judge x 3 prompts x 2 runs")


def test_errors_are_short(capsys):
    code, _, err = run(capsys, "plan", "examples/pilot.csv")
    assert code == 2 and err.startswith("evalnoise: give either --budget or --target")
    code, _, err = run(capsys, "project", "examples/pilot.csv", "graders=2")
    assert code == 2 and "unknown factor" in err
    code, _, err = run(capsys, "check", "missing.csv")
    assert code == 2
