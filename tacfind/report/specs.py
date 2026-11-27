"""Module explanations of the report's architecture tab: load `modules.yaml`, evaluate its symbol table, expand steps.

`docs/report/src/symbols.js` implements the same grammar for the page; tests compare the two on every preset and
check every step's shape against the real tensors of `TacFind`.
"""

import ast
from pathlib import Path

import yaml

SPEC_FILE = Path(__file__).with_name("modules.yaml")
FUNCS = {"min": min, "max": max, "len": len,
         "at": lambda seq, i, default: seq[i] if 0 <= i < len(seq) else default}


def load_specs() -> dict:
    """The parsed `modules.yaml` (keys symbols, modules, losses, obs_groups)."""
    return yaml.safe_load(SPEC_FILE.read_text())


def lookup(cfg: dict, path: str):
    """Value at a dotted path of a plain-dict config; KeyError when absent."""
    node = cfg
    for key in path.split("."):
        node = node[key]
    return node


def evaluate(expr, cfg: dict, symbols: dict, skip: str | None = None):
    """Evaluate one expression of the spec grammar.

    Args:
        expr: int (returned as is) or string expression.
        cfg: resolved config as plain dicts. symbols: known symbol values. skip: the symbol being defined.
    Returns:
        the value (int or list).
    """
    if not isinstance(expr, str):
        return expr

    def name(node) -> str:  # dotted name of a Name / Attribute chain
        return node.id if isinstance(node, ast.Name) else f"{name(node.value)}.{node.attr}"

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            key = name(node)
            return symbols[key] if key in symbols and key != skip else lookup(cfg, key)
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.FloorDiv: lambda: a // b}
            return ops[type(node.op)]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
            return FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression: {expr}")

    return ev(ast.parse(expr, mode="eval").body)


def symbol_values(cfg: dict, choices: dict, specs: dict | None = None) -> dict:
    """Numbers of every symbol that applies to this config (B stays symbolic and is left out).

    Args:
        cfg: resolved config (plain dicts). choices: selected option per config group.
    """
    specs = specs or load_specs()
    values = {}
    for key, sym in specs["symbols"].items():
        expr = sym["expr"]
        if isinstance(expr, dict):
            expr = expr.get(choices[expr["by"]])
        if expr is not None:
            values[key] = evaluate(expr, cfg, values, skip=key)
    return values


def active_modules(choices: dict, specs: dict | None = None) -> dict:
    """Module specs that apply to the selected group options, keyed by spec id."""
    specs = specs or load_specs()
    return {k: m for k, m in specs["modules"].items()
            if all(choices.get(g) == o for g, o in m.get("when", {}).items())}


def expand_steps(module: dict, cfg: dict, symbols: dict) -> list[dict]:
    """Flat step list: `foreach` blocks repeated per list entry, with `{i}` / `{n}` filled and local symbols bound.

    Returns:
        steps, each with `symbols` = the symbol values its shape uses.
    """
    out = []
    for step in module["steps"]:
        if "foreach" not in step:
            out.append(step | {"symbols": symbols})
            continue
        loop = step["foreach"]
        for i in range(len(evaluate(loop["over"], cfg, symbols))):
            local = dict(symbols, **{loop["index"]: i})
            for key, expr in loop["let"].items():
                local[key] = evaluate(expr, cfg, local, skip=key)
            for sub in step["steps"]:
                fill = lambda s: s.replace("{i}", str(i)).replace("{n}", str(i + 1))  # noqa: E731
                trace = {k: fill(v) if isinstance(v, str) else v for k, v in sub.get("trace", {}).items()}
                out.append(sub | {"name": fill(sub["name"]), "trace": trace, "symbols": local})
    return out


def shape_numbers(shape: list, cfg: dict, symbols: dict, batch: int) -> list[int]:
    """Numeric shape of a spec shape, with B = batch."""
    return [batch if d == "B" else evaluate(d, cfg, symbols) for d in shape]
