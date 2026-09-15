"""Keep display equations protected from Markdown block parsing on GitHub."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_seed_variance_equation_uses_a_math_fence():
    source = (ROOT / "docs/seed_protocols.md").read_text()
    blocks = re.findall(r"^```math\n(.*?)^```", source, flags=re.M | re.S)
    assert len(blocks) == 1
    assert blocks[0].strip() == (
        r"\mathrm{Var}(\bar Z)"
        "\n"
        r"= \frac{\sigma_{\mathrm{train}}^2}{I}"
        "\n"
        r"+ \frac{\sigma_{\mathrm{unlearn}}^2}{IJ}."
    )


def test_documentation_uses_fenced_display_math():
    paths = [ROOT / "README.md"]
    for directory in ("docs", "src", "community", ".github"):
        paths.extend((ROOT / directory).rglob("*.md"))
    for path in paths:
        source = path.read_text()
        prose = re.sub(r"^```[^\n]*\n.*?^```", "", source, flags=re.M | re.S)
        assert "$$" not in prose, f"Use a fenced math block in {path}"


def test_inline_formulas_are_protected_from_markdown_escaping():
    for name in ("docs/notation.md", "docs/seed_protocols.md", "src/supreme/README.md"):
        source = (ROOT / name).read_text()
        prose = re.sub(r"^```[^\n]*\n.*?^```", "", source, flags=re.M | re.S)
        for expression in re.findall(r"\$([^$\n]+)\$", prose):
            assert expression.startswith("`") and expression.endswith("`"), name
