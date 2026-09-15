"""Publishing must build source, not repackage the interactive prebuilt image."""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_devcontainer_publishing_builds_the_repository():
    build = json.loads((ROOT / ".devcontainer/devcontainer.build.json").read_text())
    assert build["build"] == {
        "dockerfile": "../docker/pure_pip.Dockerfile.cuda_12_1",
        "context": "..",
    }
    assert "image" not in build
    assert "dockerComposeFile" not in build
    workflow = (ROOT / ".github/workflows/devcontainer.yml").read_text()
    assert "configFile: .devcontainer/devcontainer.build.json" in workflow


def test_devcontainer_build_and_interactive_features_match():
    source = (ROOT / ".devcontainer/devcontainer.json").read_text()
    # This configuration uses full-line JSONC comments.
    interactive = json.loads(re.sub(r"^\s*//.*$", "", source, flags=re.M))
    build = json.loads((ROOT / ".devcontainer/devcontainer.build.json").read_text())
    for key in ("features", "remoteUser", "containerUser"):
        assert build[key] == interactive[key]
