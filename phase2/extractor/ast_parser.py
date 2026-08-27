"""
AST-level extraction of fvm::/fvc::/fvi:: (and other namespaced) call
expressions from OpenFOAM solver source, using a real C++ AST (tree-sitter)
rather than regex.

Scope decision (see project discussion): we deliberately do NOT attempt a
general call-graph/data-flow analysis of the whole OpenFOAM codebase. We
target the files where the finite-volume DSL assembles the discretized
equations, because the DSL already names the physics operator in the
function name -- e.g. `fvm::laplacian(nu, U)` IS the diffusion term. This
converts "understand a large C++ codebase" into "pattern-match a small,
stylized DSL over a real AST", which is a much smaller and much more
checkable problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tree_sitter import Language, Parser, Node
import tree_sitter_cpp as tscpp

_CPP_LANGUAGE = Language(tscpp.language())


@dataclass
class ExtractedCall:
    namespace: str            # e.g. "fvm", "fvc", "fvi", or "" if unqualified
    function: str             # e.g. "ddt", "div", "laplacian"
    arguments: list[str]      # raw source text of each top-level argument
    raw_text: str              # full call expression source text
    source_file: str
    line_start: int            # 1-indexed
    line_end: int               # 1-indexed
    assigned_to: str | None = None  # LHS variable name, if this call is the
                                     # direct RHS of a declaration/assignment
                                     # (used later for data-flow edges)

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.function}" if self.namespace else self.function


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _split_top_level_args(arg_list_node: Node, src: bytes) -> list[str]:
    """Argument list children, skipping punctuation tokens like ',' '(' ')'."""
    args = []
    for child in arg_list_node.children:
        if child.type in ("(", ")", ","):
            continue
        args.append(_node_text(child, src).strip())
    return args


def _find_enclosing_lhs(call_node: Node, src: bytes) -> str | None:
    """
    If this call is the direct initializer of a declaration or the RHS of
    an assignment, return the LHS variable name. This is a shallow,
    syntactic check (not full data-flow) -- good enough to seed data-flow
    edges between operator-graph nodes; not a claim of complete alias
    analysis.
    """
    # Walk up through node types that are purely compositional -- i.e. that
    # a call can be arbitrarily deeply nested inside without changing which
    # declaration/assignment it ultimately belongs to (a chain of N summed
    # terms is N-1 nested binary_expressions, and OpenFOAM equation
    # assemblies commonly have 3-6+ terms, so this must not be a small
    # fixed hop count -- an earlier version of this function used one and
    # silently lost assigned_to on the first two terms of a 4-term sum;
    # caught by the synthetic-modification test, see validation/ notes).
    # Stop climbing the moment we leave compositional territory (hit a
    # statement/block boundary) -- that's a real "no assignment" case, not
    # a depth-limit artifact.
    _COMPOSITIONAL = {
        "binary_expression", "unary_expression", "parenthesized_expression",
        "argument_list", "field_expression", "call_expression",
    }
    node = call_node.parent
    while node is not None:
        if node.type == "init_declarator":
            declarator = node.child_by_field_name("declarator")
            if declarator is not None:
                return _node_text(declarator, src).strip()
            return None
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            if left is not None:
                return _node_text(left, src).strip()
            return None
        if node.type not in _COMPOSITIONAL:
            return None  # left compositional territory without finding a target
        node = node.parent
    return None


def extract_calls(
    source_path: str,
    known_namespaces: set[str] | None = None,
) -> list[ExtractedCall]:
    """
    Parse `source_path` and return every namespaced call expression found
    (fvm::/fvc::/fvi:: by default, or any namespace in `known_namespaces`
    if given -- pass None to capture ALL qualified calls, useful for
    discovering namespaces the ontology doesn't know about yet).
    """
    parser = Parser(_CPP_LANGUAGE)
    with open(source_path, "rb") as f:
        src = f.read()
    tree = parser.parse(src)

    results: list[ExtractedCall] = []

    def visit(node: Node):
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            args_node = node.child_by_field_name("arguments")
            if func_node is not None and func_node.type == "qualified_identifier":
                # qualified_identifier children: scope (namespace_identifier), name
                scope_node = func_node.child_by_field_name("scope")
                name_node = func_node.child_by_field_name("name")
                namespace = _node_text(scope_node, src) if scope_node else ""
                function = _node_text(name_node, src) if name_node else _node_text(func_node, src)

                if known_namespaces is None or namespace in known_namespaces:
                    arguments = _split_top_level_args(args_node, src) if args_node else []
                    results.append(ExtractedCall(
                        namespace=namespace,
                        function=function,
                        arguments=arguments,
                        raw_text=_node_text(node, src),
                        source_file=source_path,
                        line_start=node.start_point.row + 1,
                        line_end=node.end_point.row + 1,
                        assigned_to=_find_enclosing_lhs(node, src),
                    ))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return results


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/icoFoam.C"
    calls = extract_calls(path, known_namespaces=None)
    for c in calls:
        print(f"L{c.line_start:>4}  {c.qualified_name:<18} args={c.arguments}"
              f"{'  -> ' + c.assigned_to if c.assigned_to else ''}")
    print(f"\n{len(calls)} DSL calls extracted from {path}")
