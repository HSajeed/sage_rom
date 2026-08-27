"""
Diffs two extractions by call *identity* (namespace, function, arguments),
not by line number -- so inserting/removing lines elsewhere in the file
doesn't register as spurious changes. This is the actual tool for the
synthetic-modification test: "what did this change add, and did the
extractor notice it without being told where to look."

This is also just a generally useful capability beyond Gate 1: re-running
this against a solver after any change (deliberate test injection, or a
real upstream solver update) tells you exactly what moved in the operator
graph.
"""

from __future__ import annotations

from extractor.ast_parser import ExtractedCall, extract_calls
from extractor.ontology import lookup


def _identity(call: ExtractedCall) -> tuple:
    return (call.namespace, call.function, tuple(call.arguments))


def diff_extractions(source_a: str, source_b: str) -> tuple[list[ExtractedCall], list[ExtractedCall]]:
    """Returns (added, removed): calls present in b but not a, and vice versa.
    Matches by (namespace, function, arguments) identity -- immune to line drift."""
    calls_a = extract_calls(source_a, known_namespaces=None)
    calls_b = extract_calls(source_b, known_namespaces=None)

    ids_a = [_identity(c) for c in calls_a]
    ids_b = [_identity(c) for c in calls_b]

    remaining_a = list(ids_a)
    added, removed = [], []

    for call, ident in zip(calls_b, ids_b):
        if ident in remaining_a:
            remaining_a.remove(ident)
        else:
            added.append(call)

    remaining_b = list(ids_b)
    for call, ident in zip(calls_a, ids_a):
        if ident in remaining_b:
            remaining_b.remove(ident)
        else:
            removed.append(call)

    return added, removed


def report(source_a: str, source_b: str) -> str:
    added, removed = diff_extractions(source_a, source_b)
    lines = [f"Diff: {source_a}  ->  {source_b}", f"  {len(added)} added, {len(removed)} removed\n"]

    for call in added:
        meaning = lookup(call.namespace, call.function)
        ptype = meaning.physical_type if meaning else "UNRECOGNIZED"
        conf = "low" if (meaning and "LOW CONFIDENCE" in meaning.notes) else (
            "unlabeled -> route to LLM/human review" if meaning is None else "high"
        )
        lines.append(
            f"  + L{call.line_start:<4} {call.qualified_name}({', '.join(call.arguments)})"
            f"  -> assigned_to={call.assigned_to}, physical_type={ptype}, confidence={conf}"
        )
    for call in removed:
        lines.append(f"  - L{call.line_start:<4} {call.qualified_name}({', '.join(call.arguments)})")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "fixtures/icoFoam.C"
    b = sys.argv[2] if len(sys.argv) > 2 else "fixtures/icoFoam_modified.C"
    print(report(a, b))
