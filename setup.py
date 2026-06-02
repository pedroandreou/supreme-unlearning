# Compatibility shim. All package metadata lives in pyproject.toml ([project]).
# This file only lets older pip/setuptools toolchains perform editable installs
# (pip < 21.3 requires a setuptools-based build entry); modern PEP 517/660
# front-ends use pyproject.toml directly and ignore this file.
from setuptools import setup

setup()
