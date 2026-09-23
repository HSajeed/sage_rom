"""
Minimal OpenFOAM case-dictionary reader for Step 4a's term-family builder
(term_library.py). Answers exactly the questions the rule table needs to
decide which parsed UEqn terms are active for a given case: is the
simulation laminar (constant-coefficient diffusion assumption), what is
`nu`, are MRF zones or fvOptions active, what ddt scheme is used, and what
does the PIMPLE sub-dict say. No new dependency: a small hand-rolled
recursive-descent reader for OpenFOAM's brace/semicolon dictionary syntax,
in the same spirit as extractor/fvschemes_parser.py and
extractor/dispatch.py's parse_turbulence_properties (which this module
does NOT reuse, since the extractor's version only extracts two specific
keys -- this one needs a general nested dict for fvSolution's PIMPLE block).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_TOKEN_RE = re.compile(r'"[^"]*"|[{}();]|[^\s{}();]+')


def _strip_comments(text: str) -> str:
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    return text


def parse_foam_dict(text: str) -> dict:
    """
    Parses OpenFOAM's `key value;` / `key { ... }` dictionary syntax into a
    nested dict of str -> (str | dict). Deliberately minimal: no #include,
    no macro substitution ($var), no list/table parsing (a `key ( ... );`
    entry's value is kept as the raw joined token string, not exploded into
    a list) -- none of that is needed for the keys this module reads
    (simulationType, nu, ddtSchemes.default, PIMPLE.*, and MRF/fvOptions
    zone-name presence).
    """
    tokens = _TOKEN_RE.findall(_strip_comments(text))
    pos = [0]

    def parse_block() -> dict:
        d: dict = {}
        while pos[0] < len(tokens):
            tok = tokens[pos[0]]
            if tok == "}":
                pos[0] += 1
                return d
            key = tok
            pos[0] += 1
            if pos[0] < len(tokens) and tokens[pos[0]] == "{":
                pos[0] += 1
                d[key] = parse_block()
            else:
                values = []
                while pos[0] < len(tokens) and tokens[pos[0]] not in (";",):
                    if tokens[pos[0]] in ("(", ")"):
                        pos[0] += 1
                        continue
                    values.append(tokens[pos[0]])
                    pos[0] += 1
                if pos[0] < len(tokens) and tokens[pos[0]] == ";":
                    pos[0] += 1
                d[key] = " ".join(values)
        return d

    return parse_block()


@dataclass
class CaseActivity:
    simulation_type: str            # from constant/turbulenceProperties, e.g. "laminar"
    nu: float | None                # from constant/transportProperties
    mrf_active: bool                # constant/MRFProperties present with >=1 zone entry
    fvoptions_active: bool          # constant/ or system/fvOptions present with >=1 active entry
    ddt_scheme: str | None          # system/fvSchemes ddtSchemes.default
    pimple: dict = field(default_factory=dict)   # system/fvSolution PIMPLE sub-dict


def _read_dict(path: Path) -> dict:
    return parse_foam_dict(path.read_text())


def _fvoptions_active_at(path: Path) -> bool:
    """An fvOptions dict is active if it has at least one sub-dict entry
    (a named option) whose own `active` key, if present, isn't false/no/0."""
    d = _read_dict(path)
    for value in d.values():
        if isinstance(value, dict):
            flag = value.get("active", "true").strip().lower()
            if flag not in ("false", "no", "0"):
                return True
    return False


def read_case_activity(case_dir: str) -> CaseActivity:
    case = Path(case_dir)

    simulation_type = "laminar"
    turb_path = case / "constant" / "turbulenceProperties"
    if turb_path.exists():
        simulation_type = _read_dict(turb_path).get("simulationType", "laminar")

    nu = None
    transport_path = case / "constant" / "transportProperties"
    if transport_path.exists():
        raw_nu = _read_dict(transport_path).get("nu")
        if raw_nu is not None:
            nu = float(raw_nu)

    mrf_active = False
    mrf_path = case / "constant" / "MRFProperties"
    if mrf_path.exists():
        mrf_active = len(_read_dict(mrf_path)) > 0

    fvoptions_active = False
    for rel in ("constant/fvOptions", "system/fvOptions"):
        p = case / rel
        if p.exists() and _fvoptions_active_at(p):
            fvoptions_active = True

    ddt_scheme = None
    fvschemes_path = case / "system" / "fvSchemes"
    if fvschemes_path.exists():
        ddt_block = _read_dict(fvschemes_path).get("ddtSchemes", {})
        if isinstance(ddt_block, dict):
            ddt_scheme = ddt_block.get("default")

    pimple: dict = {}
    fvsolution_path = case / "system" / "fvSolution"
    if fvsolution_path.exists():
        pimple_block = _read_dict(fvsolution_path).get("PIMPLE", {})
        if isinstance(pimple_block, dict):
            pimple = pimple_block

    return CaseActivity(
        simulation_type=simulation_type, nu=nu, mrf_active=mrf_active,
        fvoptions_active=fvoptions_active, ddt_scheme=ddt_scheme, pimple=pimple,
    )


if __name__ == "__main__":
    import sys
    activity = read_case_activity(sys.argv[1] if len(sys.argv) > 1 else "../data/datasets_29_10_2021/datasets/of_cylinder2D_binary")
    print(activity)
