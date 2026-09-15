# CI and releases

CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) lints, format-checks,
and validates the package build on every push and PR. A version tag like `v0.1.0`
triggers [`.github/workflows/publish.yml`](../.github/workflows/publish.yml) to build
and publish the release to PyPI (a manual run targets TestPyPI as a dry-run). The
CUDA images are published to GHCR manually via [`.github/workflows/docker.yml`](../.github/workflows/docker.yml)
(runtime image) and [`.github/workflows/devcontainer.yml`](../.github/workflows/devcontainer.yml)
(prebuilt dev container). Notable changes per release are tracked in [`CHANGELOG.md`](../CHANGELOG.md).

## Publish a release

Release and container workflows run on GitHub-hosted Ubuntu runners. They do not
require an SSH session, a GPU server, datasets or experiment runs.

1. Choose an unused version and update `supreme.__version__`, the runtime image
   reference in `docker/docker-compose.yml`, and `CHANGELOG.md`.
2. Run `make quality`, `make test` and `make build`. Commit the release files,
   push `main`, and wait for its CI checks to pass.
3. Create and push the corresponding version tag on that tested commit. For
   example, a new `v0.1.5` tag triggers the production PyPI upload and GitHub
   Release. Do not move an existing release tag or reuse a published version.
4. Run both container workflows from a tested source tag. Publish the runtime
   image once for the version and once for `latest`; the development-container
   workflow accepts both tags in one run.

For `v0.1.5`, the container commands are:

```bash
gh workflow run docker.yml --ref v0.1.5 -f tag=0.1.5
gh workflow run docker.yml --ref v0.1.5 -f tag=latest
gh workflow run devcontainer.yml --ref v0.1.5-containers -f tag=0.1.5,latest
```

The `v0.1.5-containers` reference includes the development-container build
configuration fix without changing the published Python release. Subsequent
releases can use their release tag for both workflows.

The publishing workflow uses `.devcontainer/devcontainer.build.json` to build
the CUDA Dockerfile and development features from source. The interactive
`.devcontainer/devcontainer.json` configuration instead pulls the prebuilt image.
Keep their feature settings aligned.

Container builds are separate from the Python release and may take substantially
longer. Check all workflow outcomes and both image tags before announcing that
all packages are available. Rebuilding a container tag changes what that tag
resolves to; use an image digest when an immutable reference is needed.

The repository's `pypi` deployment environment records Python publication. The
[project website](https://pedroandreou.github.io/supreme-unlearning-page/) is
deployed separately by GitHub Pages from the `supreme-unlearning-page` repository.
A Python release does not require redeploying an unchanged website.
