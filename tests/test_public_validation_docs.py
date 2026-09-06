"""Keep infrastructure identifiers out of publishable Markdown snapshots.

This is a regression guard, not a substitute for a full secret scanner or
review of commit history and raw log attachments.
"""

import ipaddress
from pathlib import Path
import re
import subprocess


def test_tracked_markdown_omits_private_infrastructure_details():
    root = Path(__file__).resolve().parents[1]
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "--", "*.md"], cwd=root)
        .decode()
        .split("\0")
    )
    forbidden = re.compile(
        r"\blnx-[\w.-]+|\b[\w.-]+\.lunet\.[\w.-]+|/(?:home|Users)/[^\s`]+",
        re.IGNORECASE,
    )
    findings = []
    for name in filter(None, tracked):
        path = root / name
        if not path.exists():
            continue
        content = path.read_text()
        if forbidden.search(content):
            findings.append(name)
        candidates = re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", content)
        candidates += re.findall(
            r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]*", content
        )
        for value in candidates:
            try:
                address = ipaddress.ip_address(value)
            except ValueError:
                continue
            # Loopback and wildcard examples identify no private host.
            if not address.is_loopback and not address.is_unspecified:
                findings.append(name)
    assert (
        not findings
    ), f"Review infrastructure identifiers in: {sorted(set(findings))}"
