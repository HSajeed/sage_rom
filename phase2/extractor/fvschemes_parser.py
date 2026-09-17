"""
Minimal parser for OpenFOAM's `fvSchemes` dictionary format.

This is deliberately NOT a general OpenFOAM dictionary parser (that already
exists inside OpenFOAM itself, and reimplementing it fully is out of scope).
It handles exactly the structure fvSchemes files use: a banner comment, a
FoamFile header block, then a flat sequence of named blocks
(ddtSchemes { ... }, divSchemes { ... }, ...) each containing
`key value...;` entries. That's sufficient to answer the one question this
pipeline needs from it: "what discretization scheme did the case actually
use for this operator?" -- which is case configuration, not C++, and so
doesn't belong in the AST extractor at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SchemeEntry:
    block: str          # e.g. "divSchemes"
    key: str             # e.g. "div(phi,U)" or "default"
    scheme: str            # e.g. "Gauss linear"


_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*")
_FOAMFILE_BLOCK_RE = re.compile(r"FoamFile\s*\{[^}]*\}", re.DOTALL)
_BLOCK_RE = re.compile(r"(\w+)\s*\{([^}]*)\}", re.DOTALL)
# Key: any run of non-space chars that isn't `{`, `}` or `;` -- OpenFOAM
# scheme keys can nest arbitrarily deep parens/commas/operators, e.g.
# `div((nuEff*dev(T(grad(U)))))`, which `[\w().,]+` used to truncate.
_ENTRY_RE = re.compile(r"([^\s{};]+)\s+([^;]+);")

# Only these blocks describe *discretization operator* schemes; fvSchemes
# also has structural blocks (e.g. fluxRequired, wallDist) that aren't part
# of the operator ontology and are intentionally skipped.
OPERATOR_BLOCKS = {
    "ddtSchemes", "gradSchemes", "divSchemes",
    "laplacianSchemes", "interpolationSchemes", "snGradSchemes",
}


def parse_fvschemes(path: str) -> list[SchemeEntry]:
    with open(path, "r") as f:
        text = f.read()

    text = _COMMENT_BLOCK_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _FOAMFILE_BLOCK_RE.sub("", text)

    entries: list[SchemeEntry] = []
    for block_match in _BLOCK_RE.finditer(text):
        block_name, block_body = block_match.group(1), block_match.group(2)
        if block_name not in OPERATOR_BLOCKS:
            continue
        for entry_match in _ENTRY_RE.finditer(block_body):
            key, scheme = entry_match.group(1).strip(), entry_match.group(2).strip()
            entries.append(SchemeEntry(block=block_name, key=key, scheme=scheme))
    return entries


def scheme_for(entries: list[SchemeEntry], block: str, key: str) -> str | None:
    """
    Look up the scheme for a specific operator instance (e.g. block=
    'divSchemes', key='div(phi,U)'), falling back to that block's
    'default' entry if no exact match exists -- matching OpenFOAM's own
    fallback behaviour.
    """
    exact = [e for e in entries if e.block == block and e.key == key]
    if exact:
        return exact[0].scheme
    default = [e for e in entries if e.block == block and e.key == "default"]
    if default and default[0].scheme != "none":
        return default[0].scheme
    return None


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/fvSchemes"
    for e in parse_fvschemes(path):
        print(f"{e.block:<22} {e.key:<16} -> {e.scheme}")
