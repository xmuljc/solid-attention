from pathlib import Path


def test_primary_docs_use_clean_clone_commands_and_paths() -> None:
    paths = [
        Path("README.md"),
        Path("AGENTS.md"),
        Path("io/README.md"),
        Path("integrations/llama_cpp/README.md"),
        Path("docs/ROADMAP.md"),
        Path("docs/PHASE4_RUNTIME_INTEGRATION_MAP.md"),
        *Path("docs").glob("PHASE*_STATUS.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "/data/disk2/ljc" not in text
    assert ".deps/" not in text
    assert "PYTHONPATH=.deps/python" not in text
    assert "solid attention/" not in text
    assert "python -m tools.check fast" in Path("README.md").read_text(encoding="utf-8")
    assert ".\\.venv\\Scripts\\python.exe -m pip" in Path("README.md").read_text(encoding="utf-8")
    assert "python -m tools.check liburing" in Path("AGENTS.md").read_text(encoding="utf-8")


def test_harness_baseline_status_records_scope_boundary() -> None:
    text = Path("docs/HARNESS_BASELINE_STATUS.md").read_text(encoding="utf-8")

    assert "solidattention.run_manifest.v1" in text
    assert "load-before-attention" in text
    assert "not implemented in this baseline" in text
