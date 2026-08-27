"""
LLM-assisted labeling -- NARROW scope by design (see project discussion).

For a standard solver, ast_parser.py + ontology.py already resolve most
fvm::/fvc::/fvi:: calls deterministically, because the DSL names the
physics in the function name. An LLM is NOT needed to tell you that
fvm::laplacian is diffusion.

This module exists for the cases the deterministic pass can't resolve:
  1. A call in a known namespace with NO ontology entry (e.g. a genuinely
     new fvi:: function, or fvm::Sp/SuSp source terms whose physical
     meaning depends on the surrounding case, not just the function name).
  2. A namespace the ontology has never seen (custom solver, hand-rolled
     physics not expressed through the fvm/fvc/fvi DSL at all).
  3. Anything flagged "low confidence" in ontology.py that a human hasn't
     reviewed yet -- see the fvi:: entries.

This is NOT wired to a live API call in this scaffold (no key configured
in this environment). It defines the interface and the constrained output
schema so the extraction pipeline has a clear plug-in point: swap
`stub_label` for a real Anthropic API call using this exact prompt
structure and schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ast_parser import ExtractedCall
from .ontology import ONTOLOGY

# The model must choose from this fixed set -- open-ended physical_type
# strings would defeat the point of having an ontology at all. If nothing
# fits, it must say so explicitly (UNRECOGNIZED) rather than inventing a
# category, so an UNRECOGNIZED result routes to human review instead of
# silently entering the graph as a confident label.
ALLOWED_PHYSICAL_TYPES = sorted({
    meaning.physical_type
    for ns in ONTOLOGY.values()
    for meaning in ns.values()
} | {"UNRECOGNIZED"})


LABEL_PROMPT_TEMPLATE = """\
You are labeling ONE OpenFOAM finite-volume DSL call with its physical
role in the discretized PDE. You are extracting, not inventing: base your
answer only on the call itself and the surrounding equation assembly
shown below. Do not use any physical_type outside this fixed list:

{allowed_types}

If none of these genuinely fit, answer UNRECOGNIZED -- that is a valid,
expected answer, not a failure. Do not guess.

Call: {qualified_name}({arguments})
Source: {source_file}, line {line_start}
Surrounding context (raw source, same equation assembly block):
---
{context}
---

Respond with exactly:
physical_type: <one of the allowed types above>
confidence: <high|medium|low>
justification: <one sentence, citing what in the call/context supports this>
"""


@dataclass
class LLMLabel:
    physical_type: str
    confidence: str
    justification: str
    call: ExtractedCall


def build_prompt(call: ExtractedCall, context: str) -> str:
    return LABEL_PROMPT_TEMPLATE.format(
        allowed_types="\n".join(f"  - {t}" for t in ALLOWED_PHYSICAL_TYPES),
        qualified_name=call.qualified_name,
        arguments=", ".join(call.arguments),
        source_file=call.source_file,
        line_start=call.line_start,
        context=context,
    )


def stub_label(call: ExtractedCall, context: str) -> LLMLabel:
    """
    Placeholder -- NOT a real model call. Replace with an Anthropic API
    call using `build_prompt(call, context)`, parse the three fields back
    out, and validate physical_type is in ALLOWED_PHYSICAL_TYPES before
    accepting the result (reject and flag for human review otherwise --
    never silently coerce an out-of-schema answer into the ontology).
    """
    raise NotImplementedError(
        "Wire this to a real API call before using it on anything outside "
        "the known fvm/fvc/fvi namespaces already covered by ontology.py."
    )
