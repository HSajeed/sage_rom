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
    receiver: str | None = None     # e.g. "turbulence", "MRF", "UEqn", "this"
                                     # -- only set for unqualified member/free
                                     # calls (call_form field_expression),
                                     # None for qualified (fvm::/fvc::) calls
    access: str | None = None       # "." / "->" for member calls, else None
    sign: int = 1               # +1/-1: unary minus and binary minus on the
                                 # path from this call up to its enclosing
                                 # sum, syntactic only (see extract_calls doc)
    side: str | None = None     # "lhs"/"rhs" of the nearest enclosing `==`
                                 # (fvMatrix equation split), else None
    start_byte: int = 0          # byte offset of the full call_expression in
                                  # source_file -- used (with end_byte) for
                                  # position-based nesting checks, since two
                                  # textually identical calls on different
                                  # lines must not be confused by raw-text
                                  # containment (see operator_graph.py)
    end_byte: int = 0
    args_start_byte: int | None = None  # byte offsets of this call's own
    args_end_byte: int | None = None    # argument_list (including the
                                         # parens), so a *different* call can
                                         # check "am I nested inside THIS
                                         # call's arguments"; None if there's
                                         # no argument_list (shouldn't happen
                                         # for a call, kept Optional for safety)

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


# Node types a call can be nested inside without leaving "one sum" or "one
# equation" -- used by both _find_enclosing_lhs (already existed) and
# _find_sign_and_side (new). Kept as the same set as _COMPOSITIONAL in
# _find_enclosing_lhs; duplicated locally there to keep that function
# self-contained.
_ARITHMETIC_COMPOSITIONAL = {
    "binary_expression", "unary_expression", "parenthesized_expression",
}


def _find_sign_and_side(call_node: Node, src: bytes) -> tuple[int, str | None]:
    """
    Syntactic sign/side of a call within its enclosing fvMatrix expression.

    Walks strictly upward through unary_expression/binary_expression/
    parenthesized_expression nodes (arithmetic composition only -- this does
    NOT cross into a different call's argument_list, so a nested call's sign
    is local to whatever expression it's actually written in, matching how
    fvMatrix operator+/operator-/operator== actually combine terms):
      - a unary '-' flips the running sign.
      - for a binary '-', the node on the right side gets its sign flipped
        (a - b == a + (-b)); the left side is unaffected.
      - binary '+' has no effect.
      - the first binary_expression whose operator is '==' fixes `side`
        ("lhs"/"rhs" of that '==') and further sign changes above the '=='
        node still accumulate (e.g. a `-` in front of the whole RHS), but
        `side` itself is only set once, at the nearest enclosing '=='.

    This is deliberately shallow: it does not evaluate `*`/`/` (a term
    multiplied by a negative *coefficient*, e.g. `-1.0*fvm::Sp(...)`, is not
    tracked -- only `+`/`-`/unary '-' composition of terms is), and it stops
    at the first non-arithmetic ancestor (e.g. an argument_list boundary),
    so a call passed as an argument to another call (fvOptions(U) inside a
    sum) is scored within ITS enclosing sum, not the outer one.
    """
    sign = 1
    side: str | None = None
    node = call_node
    parent = node.parent
    while parent is not None and parent.type in _ARITHMETIC_COMPOSITIONAL:
        if parent.type == "unary_expression":
            op = parent.child_by_field_name("operator")
            if op is not None and _node_text(op, src) == "-":
                sign *= -1
        elif parent.type == "binary_expression":
            op = parent.child_by_field_name("operator")
            op_text = _node_text(op, src) if op is not None else ""
            right = parent.child_by_field_name("right")
            is_right_child = right is not None and right.id == node.id
            if op_text == "==":
                if side is None:
                    side = "rhs" if is_right_child else "lhs"
            elif op_text == "-" and is_right_child:
                sign *= -1
            # '+' and left-hand '-' leave sign unchanged
        node = parent
        parent = parent.parent
    return sign, side


def extract_calls(
    source_path: str,
    known_namespaces: set[str] | None = None,
    include_unqualified: bool = False,
) -> list[ExtractedCall]:
    """
    Parse `source_path` and return every namespaced call expression found
    (fvm::/fvc::/fvi:: by default, or any namespace in `known_namespaces`
    if given -- pass None to capture ALL qualified calls, useful for
    discovering namespaces the ontology doesn't know about yet).

    If `include_unqualified` is True (default OFF, so icoFoam behaviour and
    Gate 1 are unchanged), also capture unqualified calls: plain identifier
    calls (`solve(...)`, `dev2(...)`), member calls via `.`/`->` including
    `this->` (`turbulence->divDevReff(U)`, `MRF.correctBoundaryVelocity(U)`),
    and template_function calls. These get namespace="" and, where the call
    is a member call, `receiver`/`access` set from the field_expression.
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
            namespace = None
            function = None
            receiver = None
            access = None

            if func_node is not None and func_node.type == "qualified_identifier":
                # qualified_identifier children: scope (namespace_identifier), name
                scope_node = func_node.child_by_field_name("scope")
                name_node = func_node.child_by_field_name("name")
                namespace = _node_text(scope_node, src) if scope_node else ""
                function = _node_text(name_node, src) if name_node else _node_text(func_node, src)
                if known_namespaces is not None and namespace not in known_namespaces:
                    namespace = None  # filtered out below
            elif include_unqualified and func_node is not None:
                if func_node.type == "identifier":
                    namespace = ""
                    function = _node_text(func_node, src)
                elif func_node.type == "field_expression":
                    obj_node = func_node.child_by_field_name("argument")
                    field_node = func_node.child_by_field_name("field")
                    op_node = func_node.child_by_field_name("operator")
                    namespace = ""
                    function = _node_text(field_node, src) if field_node else _node_text(func_node, src)
                    receiver = _node_text(obj_node, src) if obj_node else None
                    access = _node_text(op_node, src) if op_node else None
                elif func_node.type == "template_function":
                    namespace = ""
                    name_node = func_node.child_by_field_name("name")
                    function = _node_text(name_node, src) if name_node else _node_text(func_node, src)

            if namespace is not None and function is not None:
                arguments = _split_top_level_args(args_node, src) if args_node else []
                sign, side = _find_sign_and_side(node, src)
                results.append(ExtractedCall(
                    namespace=namespace,
                    function=function,
                    arguments=arguments,
                    raw_text=_node_text(node, src),
                    source_file=source_path,
                    line_start=node.start_point.row + 1,
                    line_end=node.end_point.row + 1,
                    assigned_to=_find_enclosing_lhs(node, src),
                    receiver=receiver,
                    access=access,
                    sign=sign,
                    side=side,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    args_start_byte=args_node.start_byte if args_node else None,
                    args_end_byte=args_node.end_byte if args_node else None,
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
