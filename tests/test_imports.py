"""Import smoke test: every module under ``supreme`` must import cleanly.

This is the cheapest, highest-coverage guard in the suite. Walking and importing
every submodule catches syntax errors, broken/circular imports, and
import-time crashes across the whole package - the class of regression a
lint-only pipeline silently lets through. ``import supreme`` itself is
torch-free by design (see ``src/supreme/registry.py``); the leaf modules pull in
torch/Lightning/transformers, but none download data or weights at import time.
"""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import supreme

MODULES = sorted(m.name for m in pkgutil.walk_packages(supreme.__path__, "supreme."))


def test_package_import_is_torch_free():
    # `import supreme` must stay light enough to register components without the
    # heavy PyTorch/Lightning stack loaded - the registry documents this. Run in
    # a fresh interpreter so torch already imported by other tests can't mask a
    # regression.
    code = "import supreme, sys; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert hasattr(supreme, "register_model")


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
