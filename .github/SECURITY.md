# Security Policy

## Supported versions

SUPREME is published to PyPI as
[`supreme-unlearning`](https://pypi.org/project/supreme-unlearning/). Security
fixes are applied to the latest released version on the `main` branch. Please
upgrade to the most recent release before reporting an issue.

| Version | Supported |
|---|---|
| Latest release | ✅ |
| Older releases | ❌ |

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/pedroandreou/supreme-unlearning/security/advisories/new)
(Security tab -> "Report a vulnerability").

When reporting, please include:

- a description of the issue and its potential impact,
- the exact command or `supreme.run_*` call that triggers it,
- steps to reproduce, and
- your environment (OS, install method, Python/PyTorch versions).

You can expect an initial acknowledgement within a few days. Once a fix is
ready, it will be released and the advisory published with credit to the
reporter unless anonymity is requested.

## Dependency pinning and known advisories

SUPREME pins its full dependency stack (PyTorch 2.1.0, Lightning 2.1.0,
Transformers 4.35.0, DeepSpeed 0.14.5, ...) to the exact environment used to
produce the results in the WIPE-OUT 2 (ECML-PKDD 2026) paper. These versions
are interdependent (e.g. DeepSpeed 0.14.x is the last line compatible with
PyTorch 2.1), so they are kept frozen for reproducibility rather than upgraded
as new dependency advisories appear.

Published advisories against these pinned versions (e.g. `torch.load` /
checkpoint deserialization issues) are exploitable only when loading
**untrusted** checkpoints, weights, or Hub models. SUPREME is a research
framework intended to train, unlearn, and evaluate models from sources you
control:

- Only load checkpoints and datasets you produced yourself or obtained from a
  source you trust.
- Do not point SUPREME at model files or Hub repositories from unknown parties.
- Run experiments in an isolated environment (venv/container) as an
  unprivileged user.

Dependabot alerts covered by this policy are dismissed as "tolerable risk"
with a pointer to this document. Vulnerabilities in SUPREME's own code are
always in scope - please report them as described above.
