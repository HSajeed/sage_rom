"""
Gate 1: does the extractor's automated labeling match a hand-reviewed
answer key? Run this BEFORE building anything ROM-related on top of the
operator graph -- there's no point structuring a ROM around a graph that's
wrong.

Usage:
    python -m validation.gate1_check \
        --source fixtures/icoFoam.C \
        --ground-truth validation/ground_truth_icofoam.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from extractor.ast_parser import extract_calls
from extractor.ontology import resolve_label


def load_ground_truth(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_gate1(
    source_path: str,
    ground_truth_path: str,
    include_unqualified: bool = False,
    strict_labels: bool = False,
) -> int:
    gt = load_ground_truth(ground_truth_path)
    # Multi-file fixtures (e.g. pimpleFoam_v2006: UEqn.H, pEqn.H, ...) tag
    # each entry with its own `source_file`; score only the entries for the
    # file being checked. Entries without the field apply to --source
    # (single-file answer keys like ground_truth_icofoam.yaml are unchanged).
    gt_entries = [
        e for e in gt["entries"]
        if "source_file" not in e
        or os.path.normpath(e["source_file"]) == os.path.normpath(source_path)
    ]

    unreviewed = [e for e in gt_entries if not e.get("reviewed", False)]
    if unreviewed:
        print(
            f"WARNING: {len(unreviewed)}/{len(gt_entries)} ground-truth entries "
            f"are unreviewed (reviewed: false). This Gate 1 run is checking the "
            f"pipeline against a DRAFT answer key, not a validated one -- treat "
            f"the numbers below as a sanity check on the harness, not as a real "
            f"result, until someone reviews {ground_truth_path}.\n",
            file=sys.stderr,
        )

    # A source line can hold more than one call (e.g. fvc::interpolate(...)
    # and fvc::ddtCorr(...) both on the same line) -- match against the pool
    # of calls on that line, not a single {line: call} slot, and remove a
    # call from the pool once it's matched so two ground-truth entries can't
    # both claim the same extracted call.
    extracted = extract_calls(source_path, known_namespaces=None, include_unqualified=include_unqualified)
    pool_by_line: dict[int, list] = {}
    for c in extracted:
        pool_by_line.setdefault(c.line_start, []).append(c)

    tp_calls, tp_types, mismatches, missed = [], [], [], []
    tp_matches: list[tuple[dict, "ExtractedCall"]] = []  # (entry, matched call)

    for entry in gt_entries:
        line = entry["line"]
        candidates = pool_by_line.get(line, [])
        match = next(
            (c for c in candidates
             if c.namespace == entry["namespace"] and c.function == entry["function"]),
            None,
        )
        if match is not None:
            candidates.remove(match)
            tp_calls.append(entry)
            tp_matches.append((entry, match))
            if match.arguments == entry.get("arguments", []):
                tp_types.append(entry)
        elif candidates:
            # Something was extracted on this line, just not a matching
            # (namespace, function) -- a real mismatch, not a miss.
            mismatches.append((entry, candidates[0]))
        else:
            missed.append(entry)

    # Whatever's left in the per-line pools after matching is unclaimed by
    # any ground-truth entry -- either a genuinely new call to add to the
    # answer key, or a false positive.
    extra = [c for lst in pool_by_line.values() for c in lst]

    n_gt = len(gt_entries)
    recall = len(tp_calls) / n_gt if n_gt else float("nan")
    precision = len(tp_calls) / len(extracted) if extracted else float("nan")
    arg_exact_rate = len(tp_types) / len(tp_calls) if tp_calls else float("nan")

    # Label agreement: compare the extractor's operand-aware physical_type
    # and discretization_role against the ground truth, for matched calls
    # whose GT scope is an actual operator label (operator/operator_argument)
    # or has no scope field at all (single-file keys like icoFoam's, which
    # predate the scope field and are entirely operator-scope). This is the
    # check Gate 1 was missing: call-identity matching alone can't catch a
    # mislabeled physical_type/discretization_role.
    label_checked = [
        (entry, match) for entry, match in tp_matches
        if entry.get("scope") in (None, "operator", "operator_argument")
    ]
    type_mismatches: list[tuple[dict, str | None]] = []
    role_mismatches: list[tuple[dict, str | None]] = []
    for entry, match in label_checked:
        meaning = resolve_label(
            match.namespace, match.function, match.arguments,
            receiver=match.receiver, access=match.access,
        )
        got_type = meaning.physical_type if meaning else None
        got_role = meaning.discretization_role if meaning else None
        if "physical_type" in entry and got_type != entry["physical_type"]:
            type_mismatches.append((entry, got_type))
        if "discretization_role" in entry and got_role != entry["discretization_role"]:
            role_mismatches.append((entry, got_role))

    n_label_checked = len(label_checked)
    type_agree = n_label_checked - len(type_mismatches)
    role_checked = [e for e, _ in label_checked if "discretization_role" in e]
    role_agree = len(role_checked) - len(role_mismatches)
    type_rate = type_agree / n_label_checked if n_label_checked else float("nan")
    role_rate = role_agree / len(role_checked) if role_checked else float("nan")

    print(f"Source:        {source_path}")
    print(f"Ground truth:  {ground_truth_path} ({n_gt} entries, "
          f"{n_gt - len(unreviewed)} reviewed)")
    print(f"Extracted:     {len(extracted)} calls\n")
    print(f"Recall (call identity):       {recall:.2%}  ({len(tp_calls)}/{n_gt})")
    print(f"Precision (call identity):    {precision:.2%}  ({len(tp_calls)}/{len(extracted)})")
    print(f"Argument exact-match rate:    {arg_exact_rate:.2%}  "
          f"(among matched calls, {len(tp_types)}/{len(tp_calls)})")
    print(f"Label agreement (physical_type): {type_rate:.2%}  ({type_agree}/{n_label_checked})")
    print(f"Role agreement (discretization_role): {role_rate:.2%}  ({role_agree}/{len(role_checked)})")

    if type_mismatches:
        print(f"\nLABEL MISMATCHES ({len(type_mismatches)}):")
        for entry, got in type_mismatches:
            print(
                f"  line {entry['line']}: {entry['namespace']}::{entry['function']} "
                f"expected physical_type={entry['physical_type']!r}, got {got!r}"
            )
    if role_mismatches:
        print(f"\nROLE MISMATCHES ({len(role_mismatches)}):")
        for entry, got in role_mismatches:
            print(
                f"  line {entry['line']}: {entry['namespace']}::{entry['function']} "
                f"expected discretization_role={entry['discretization_role']!r}, got {got!r}"
            )

    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for entry, found in mismatches:
            print(f"  line {entry['line']}: expected {entry['namespace']}::{entry['function']}, "
                  f"got {found.namespace}::{found.function}")
    if missed:
        print(f"\nMISSED ({len(missed)}) -- in ground truth, not extracted:")
        for entry in missed:
            print(f"  line {entry['line']}: {entry['namespace']}::{entry['function']}")
    if extra:
        print(f"\nEXTRA ({len(extra)}) -- extracted, not in ground truth "
              f"(new calls to add to the answer key, or false positives):")
        for c in extra:
            print(f"  line {c.line_start}: {c.qualified_name}({', '.join(c.arguments)})")

    call_identity_failed = bool(mismatches or missed)
    label_failed = strict_labels and bool(type_mismatches or role_mismatches)
    return 0 if not (call_identity_failed or label_failed) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="fixtures/icoFoam.C")
    parser.add_argument("--ground-truth", default="validation/ground_truth_icofoam.yaml")
    parser.add_argument("--include-unqualified", action="store_true",
                         help="Also extract unqualified (member/free-function) calls, "
                              "for multi-file fixtures whose ground truth includes them.")
    parser.add_argument("--strict-labels", action="store_true",
                         help="Non-zero exit on physical_type/discretization_role "
                              "mismatches, not just call-identity mismatches/misses "
                              "(default: report only, don't fail the run).")
    args = parser.parse_args()
    sys.exit(run_gate1(
        args.source, args.ground_truth,
        include_unqualified=args.include_unqualified,
        strict_labels=args.strict_labels,
    ))
