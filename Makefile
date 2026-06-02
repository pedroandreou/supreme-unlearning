# SUPREME developer convenience targets.
#
# The Makefile is the entry point for LOCAL, interactive dev work in this repo -
# creating the venv, installing/updating dependencies, the editable install, the
# git hook, and building/publishing the package. For local work, prefer it over
# running pip/venv/twine by hand.
#
# The automated environments deliberately do NOT call it (they install into the
# system / container Python, not a venv) and instead mirror these same recipes:
#   - Docker:  pure_pip.Dockerfile.cuda_12_1  (mirrors `cuda` + the editable install)
#   - CI lint: .github/workflows/ci.yml       (calls `make quality RUFF=ruff`)
#   - CI build/publish: ci.yml + publish.yml  (mirror the `build` recipe below)
# When you change a recipe here, update its mirror.
#
# First-time setup (creates and provisions the virtual environment):
#
#   make mps     # Apple Silicon (M1/M2/M3/M4) / CPU
#   make cuda    # NVIDIA / CUDA 12.1 hosts (Linux / WSL2)
#
# The environment is named `unlearning` by default (hardware-neutral). Override it
# anywhere with VENV=<name>, or export VENV once for your shell:
#   make mps VENV=my_env        # or:  export VENV=my_env
#
# These create the venv (prompting if it already exists), install the pinned deps
# + SUPREME (editable) into it, and enable the pre-commit hook. A Makefile cannot
# activate a venv for your interactive shell, so it installs via <venv>/bin/pip
# directly; activate it yourself afterwards for day-to-day use:
#
#   source unlearning/bin/activate      # or: source $(VENV)/bin/activate
#
# Non-interactive (notebooks / CI): pass ON_EXISTING=reuse or ON_EXISTING=recreate
#   make mps ON_EXISTING=reuse
#
# Override the interpreter used to BUILD the venv (must be Python 3.9). Defaults
# to `python3.9`, falling back to the newest installed pyenv 3.9.x:
#   make mps BASE_PYTHON=$HOME/.pyenv/versions/3.9.12/bin/python

.PHONY: help cuda mps dev hooks build publish publish-test clean quality style \
        test precommit venv _create_venv _ensure_venv

# Virtual environment location (override with VENV=<name>) and the tools inside it.
VENV   ?= unlearning
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
# ruff binary used by quality/style. Defaults to the one in the venv; CI overrides
# it (RUFF=ruff) to reuse these targets against a system-installed ruff.
RUFF   ?= $(VENV)/bin/ruff
# Interpreter used to CREATE the venv (3.9 required); see header for overrides.
BASE_PYTHON ?= python3.9
# Behaviour when the venv already exists: empty -> prompt; reuse | recreate.
ON_EXISTING ?=

# Directories/files that the lint + format targets operate on. Single source of
# truth: CI runs `make quality`, so it inherits this list (don't duplicate it there).
check_dirs := src tests setup.py

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- Environment setup ----------------------------------------------------

cuda: venv  ## Create/refresh the venv with CUDA 12.1 deps + editable package + git hook
	$(PIP) install -r requirements.cuda_12_1.txt
	@$(MAKE) --no-print-directory dev

mps: venv  ## Create/refresh the venv with Apple-Silicon (MPS)/CPU deps + editable package + git hook
	$(PIP) install -r requirements.mps.txt
	@$(MAKE) --no-print-directory dev

venv:  ## Create the venv (prompts to reuse/recreate if it exists; ON_EXISTING skips the prompt)
	@if [ -d "$(VENV)" ]; then \
		action="$(ON_EXISTING)"; \
		if [ -z "$$action" ]; then \
			printf "Virtual env '$(VENV)/' already exists. [r]euse & reinstall / [d]elete & recreate / [c]ancel? "; \
			read ans; \
			case "$$ans" in [dD]*) action=recreate ;; [rR]*) action=reuse ;; *) action=cancel ;; esac; \
		fi; \
		case "$$action" in \
			recreate) echo "Deleting and recreating $(VENV)/ ..."; rm -rf "$(VENV)"; $(MAKE) --no-print-directory _create_venv ;; \
			reuse)    echo "Reusing existing $(VENV)/ (reinstalling into it)." ;; \
			*)        echo "Cancelled."; exit 1 ;; \
		esac; \
	else \
		$(MAKE) --no-print-directory _create_venv; \
	fi

_create_venv:
	@echo "Creating virtual environment at $(VENV)/ ..."
	@PY="$(BASE_PYTHON)"; \
	if ! command -v "$$PY" >/dev/null 2>&1; then \
		PY="$$(ls -d $(HOME)/.pyenv/versions/3.9.*/bin/python 2>/dev/null | sort -V | tail -1)"; \
	fi; \
	if [ -z "$$PY" ] || ! "$$PY" --version >/dev/null 2>&1; then \
		echo "ERROR: no Python 3.9 interpreter found."; \
		echo "  Install one (e.g. 'pyenv install 3.9.12') or pass BASE_PYTHON=/path/to/python3.9"; \
		exit 1; \
	fi; \
	echo "Using $$PY ($$($$PY --version 2>&1))"; \
	"$$PY" -m venv "$(VENV)"

# Ensure the venv exists WITHOUT prompting (used by build/publish). Create if absent.
_ensure_venv:
	@[ -d "$(VENV)" ] || $(MAKE) --no-print-directory _create_venv

dev:  ## Editable install + enable the pre-commit git hook (into the venv)
	$(PIP) install -e .
	$(VENV)/bin/pre-commit install
	@echo ""
	@echo "Setup complete. Activate the environment for interactive use:"
	@echo "    source $(VENV)/bin/activate"

hooks:  ## (Re)install the pre-commit git hook only
	$(VENV)/bin/pre-commit install

# --- Packaging ------------------------------------------------------------

build: _ensure_venv clean  ## Build sdist + wheel into dist/ (distribution name: supreme-unlearning)
	$(PIP) install --quiet build
	$(PYTHON) -m build
	$(PIP) install --quiet twine
	$(PYTHON) -m twine check dist/*

publish-test: build  ## Upload to TestPyPI (needs a TestPyPI API token)
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build  ## Upload to PyPI - IRREVERSIBLE, a version can be uploaded only once
	@echo "About to upload dist/* to PyPI as 'supreme-unlearning'. This cannot be undone."
	$(PYTHON) -m twine upload dist/*

clean:  ## Remove build artifacts (dist/, build/, *.egg-info, __pycache__)
	rm -rf dist build supreme.egg-info supreme_unlearning.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --- Code quality ---------------------------------------------------------

quality:  ## Lint + format check without modifying files (CI-style)
	$(RUFF) check $(check_dirs)
	$(RUFF) format --check $(check_dirs)

style:  ## Auto-fix lint issues and format in place
	$(RUFF) check $(check_dirs) --fix
	$(RUFF) format $(check_dirs)

# --- Tests ----------------------------------------------------------------

test:  ## Run the CPU-only unit test suite (pytest). Mirrored by ci.yml's tests job.
	$(PIP) install --quiet pytest
	$(PYTHON) -m pytest

precommit:  ## Run every pre-commit hook against the whole tree
	$(VENV)/bin/pre-commit run --all-files
