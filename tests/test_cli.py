"""CLI smoke test — offline mode runs end to end with no API key."""

import json
from pathlib import Path

from app.cli import main

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"


def test_cli_offline_json_run(capsys):
    files = [str(p) for p in sorted(CORPUS.glob("*.md"))]
    assert files, "expected an eval corpus to exist"
    code = main([*files, "--topic", "the water cycle", "--offline", "--json", "--num-questions", "3"])
    out = capsys.readouterr()
    assert code == 0
    data = json.loads(out.out)
    assert isinstance(data, list) and len(data) >= 1
    assert data[0]["question"]
    assert "[metrics]" in out.err                       # metrics line goes to stderr


def test_cli_persist_dir_smoke(tmp_path, capsys):
    files = [str(p) for p in sorted(CORPUS.glob("*.md"))][:1]
    code = main([*files, "--topic", "photosynthesis", "--offline",
                 "--persist-dir", str(tmp_path / "c"), "--num-questions", "1"])
    assert code == 0
    # the persist dir was actually created on disk
    assert (tmp_path / "c").exists()
