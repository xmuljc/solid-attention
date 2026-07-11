from pathlib import Path


def test_ci_workflow_covers_supported_profiles_and_platforms() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    required_fragments = (
        "windows-latest",
        "ubuntu-latest",
        "branches: [main, harness-candidate]",
        'python: "3.11"',
        'python: "3.13"',
        "python -m tools.check fast",
        "python -m tools.check io",
        "python -m tools.check liburing",
        "python -m tools.check llama",
        "submodules: recursive",
        "liburing-dev",
        "if-no-files-found: ignore",
    )

    for fragment in required_fragments:
        assert fragment in workflow

    assert workflow.count("cache-dependency-path: pyproject.toml") == 4
    assert workflow.count("overwrite: true") == 4
