<div align="center">

# SUPREME Community

</div>

This is where the community shares **unlearning methods, metrics, and results**
built on SUPREME. We encourage you to develop new methods, tune them for the
supported benchmarks, and compare against existing approaches.

## What lives here

| Path | Purpose |
|---|---|
| [`methods/`](methods/) | One folder per community-contributed unlearning method: a short README describing the method and a `run.sh` that reproduces its results. Start from [`methods/template/`](methods/template/). |
| [`leaderboard.md`](leaderboard.md) | Community results table. Submit your numbers via PR. |

## Two ways to contribute a method

SUPREME is registry-based and pip-installable, so you can share a component
**without forking the framework**:

1. **Ship it as your own plugin package.** Register your component from your own
   package (`supreme.register_*` or packaging entry points) so others
   `pip install` it and use it directly. See
   [`docs/extending.md`](../docs/extending.md) and
   [`notebooks/custom_components.ipynb`](../notebooks/custom_components.ipynb).
   Then add a folder here (and a leaderboard row) so people can find it and
   reproduce your numbers.

2. **Upstream it in-tree** via a pull request into `src/supreme/`, with a folder here
   documenting it.

Either way, copy [`methods/template/`](methods/template/) and fill it in. See the
[contributing guide](../docs/contributing.md) for the full workflow.

> [!NOTE]
> The reproducibility numbers in [`docs/reproducing_the_paper.md`](../docs/reproducing_the_paper.md)
> are fixed to the paper. The [leaderboard](leaderboard.md) is the living table:
> contribute improvements and new methods there.
