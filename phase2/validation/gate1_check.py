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
import sys

import yaml

from extractor.ast_parser import extract_calls


def load_ground_truth(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_gate1(source_path: str, ground_truth_path: str) -> int:
    gt = load_ground_truth(ground_truth_path)
    gt_entries = gt["entries"]

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
    extracted = extract_calls(source_path, known_namespaces=None)
    pool_by_line: dict[int, list] = {}
    for c in extracted:
        pool_by_line.setdefault(c.line_start, []).append(c)

    tp_calls, tp_types, mismatches, missed = [], [], [], []

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

    print(f"Source:        {source_path}")
    print(f"Ground truth:  {ground_truth_path} ({n_gt} entries, "
          f"{n_gt - len(unreviewed)} reviewed)")
    print(f"Extracted:     {len(extracted)} calls\n")
    print(f"Recall (call identity):       {recall:.2%}  ({len(tp_calls)}/{n_gt})")
    print(f"Precision (call identity):    {precision:.2%}  ({len(tp_calls)}/{len(extracted)})")
    print(f"Argument exact-match rate:    {arg_exact_rate:.2%}  "
          f"(among matched calls, {len(tp_types)}/{len(tp_calls)})")

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

    return 0 if (not mismatches and not missed) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="fixtures/icoFoam.C")
    parser.add_argument("--ground-truth", default="validation/ground_truth_icofoam.yaml")
    args = parser.parse_args()
    sys.exit(run_gate1(args.source, args.ground_truth))
