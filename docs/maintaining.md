# CI and releases

CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) lints, format-checks,
and validates the package build on every push and PR. A version tag like `v0.1.0`
triggers [`.github/workflows/publish.yml`](../.github/workflows/publish.yml) to build
and publish the release to PyPI (a manual run targets TestPyPI as a dry-run). The
CUDA images are published to GHCR manually via [`.github/workflows/docker.yml`](../.github/workflows/docker.yml)
(runtime image) and [`.github/workflows/devcontainer.yml`](../.github/workflows/devcontainer.yml)
(prebuilt dev container). Notable changes per release are tracked in [`CHANGELOG.md`](../CHANGELOG.md).
