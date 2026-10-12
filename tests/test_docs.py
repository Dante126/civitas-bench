"""README.md and docs/*.md: every relative link points at a file that exists, and every #anchor at a heading."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    h = heading.strip().lower().replace("`", "")
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


def _unfenced(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _prose(text: str) -> str:
    return re.sub(r"`[^`\n]*`", "", _unfenced(text))


def _anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for line in _unfenced(path.read_text()).split("\n"):
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            s = _slug(m.group(1))
            out.add(s if s not in seen else f"{s}-{seen[s]}")
            seen[s] = seen.get(s, 0) + 1
    return out


def test_relative_links_resolve():
    broken = []
    for f in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", _prose(f.read_text())):
            if re.match(r"^[a-z]+:", target):
                continue
            path, _, anchor = target.partition("#")
            dest = (f.parent / path).resolve() if path else f
            if not dest.exists():
                broken.append(f"{f.name}: {target} (no such file)")
            elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
                broken.append(f"{f.name}: {target} (no such heading)")
    assert not broken, "\n".join(broken)
