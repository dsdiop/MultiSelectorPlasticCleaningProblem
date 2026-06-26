from pathlib import Path

from Learning.ctde_ram_claude import experiment_io


def test_resolve_run_dir_adds_numeric_suffix_for_existing_names(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()

    first = experiment_io.resolve_run_dir(str(root), "my_run")
    assert Path(first) == root / "my_run"

    second = experiment_io.resolve_run_dir(str(root), "my_run")
    assert Path(second) == root / "my_run_1"

    (root / "my_run_1").mkdir()
    third = experiment_io.resolve_run_dir(str(root), "my_run")
    assert Path(third) == root / "my_run_2"


def test_probe_artifact_names_include_run_name_and_episode_count(tmp_path):
    run_dir = tmp_path / "outputs" / "my_run"
    run_dir.mkdir(parents=True)

    csv_path = experiment_io.build_probe_artifact_path(str(run_dir), "my_run", 5000, "csv")
    png_path = experiment_io.build_probe_artifact_path(str(run_dir), "my_run", 5000, "png")

    assert Path(csv_path) == run_dir / "my_run_episodes_5000_probe.csv"
    assert Path(png_path) == run_dir / "my_run_episodes_5000_probe.png"
