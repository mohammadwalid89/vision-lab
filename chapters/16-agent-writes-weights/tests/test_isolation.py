"""The agent's folder holds no test split and no path to one, and sits outside the work directory.

Runs the suite on synthetic features with the agent replaced by a stub that records what it could see.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ch16_suite", HERE / "suite.py")
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)


def _features(folder: Path):
    folder.mkdir()
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 120)
    X = rng.normal(size=(len(y), 64)) + 0.3 * y[:, None]  # the placebo edits 40 weights
    np.savez(folder / "cifar10.npz", X=X, y=y)


def test_agent_folder_has_no_test_split(tmp_path, monkeypatch):
    _features(tmp_path / "features")
    work = tmp_path / "work"
    agents = tmp_path / "agents"
    monkeypatch.setenv("WSP_AGENT_TMP", str(agents))
    s = suite.Suite(work, "v1")
    s.build(tmp_path / "features")
    assert list(work.glob("tasks/*/test.npz"))

    seen = []

    def stub_agent(cwd, prompt, tools, minutes):
        cwd = Path(cwd)
        files = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*"))
        text = "\n".join(p.read_text(errors="replace") for p in cwd.rglob("*")
                         if p.suffix in (".md", ".py", ".txt", ".json"))
        seen.append({"cwd": cwd, "files": files, "text": text})
        if "BRIEF.md" in prompt:
            np.savez(cwd / "final.npz", **dict(np.load(cwd / "start_head.npz")))
        return 0

    monkeypatch.setattr(s, "agent", stub_agent)
    s.run("lessons")

    order = json.loads((work / "order.json").read_text())
    assert len(seen) == 2 * len(order)  # task session and lesson session per task
    for view in seen:
        assert work.resolve() not in view["cwd"].resolve().parents
        assert not any(name.endswith("test.npz") or name == "meta.json" for name in view["files"])
        assert str(work) not in view["text"] and "tasks/" not in view["text"]
    assert {"fit.npz", "feedback.npz", "start_head.npz", "BRIEF.md"} <= set(seen[0]["files"])

    assert list(agents.iterdir()) == []  # agent folders are removed after copying back
    rows = [json.loads(line) for line in (work / "arm_lessons" / "results.jsonl").read_text().splitlines()]
    assert [r["task"] for r in rows] == order
    assert all("agent" in r["test"] for r in rows)
    assert all((work / "arm_lessons" / t / "final.npz").is_file() for t in order)


def test_research_session_runs_outside_the_work_directory(tmp_path, monkeypatch):
    _features(tmp_path / "features")
    work = tmp_path / "work"
    agents = tmp_path / "agents"
    monkeypatch.setenv("WSP_AGENT_TMP", str(agents))
    s = suite.Suite(work, "v2")
    s.build(tmp_path / "features")

    seen = []

    def stub_agent(cwd, prompt, tools, minutes):
        cwd = Path(cwd)
        seen.append({"cwd": cwd, "files": sorted(p.name for p in cwd.rglob("*"))})
        (cwd / "PLAYBOOK.md").write_text("# playbook\n")
        return 0

    monkeypatch.setattr(s, "agent", stub_agent)
    s.research()

    assert len(seen) == 1
    assert work.resolve() not in seen[0]["cwd"].resolve().parents
    assert seen[0]["files"] == []
    assert (work / "research" / "PLAYBOOK.md").read_text() == "# playbook\n"  # copied back
    assert list(agents.iterdir()) == []
