"""Calculator tool: arithmetic only, via a restricted AST evaluator.

No `eval()`. Only a fixed set of AST node types are permitted, so this
cannot be used to execute arbitrary Python.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from arl.tools.base import Tool, ToolError

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    raise ToolError(f"Disallowed expression: {ast.dump(node)}", error_type="invalid_arguments")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a numeric arithmetic expression, e.g. '(3 + 4) * 2'."
    input_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }

    def execute(self, arguments: dict[str, Any]) -> str:
        expression = arguments.get("expression")
        if not expression:
            raise ToolError("'expression' is required", error_type="invalid_arguments")
        if not isinstance(expression, str):
            raise ToolError("'expression' must be a string", error_type="invalid_arguments")
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            raise ToolError(f"Could not parse expression: {exc}", "invalid_arguments") from exc
        try:
            result = _eval_node(tree)
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero", error_type="execution_error") from exc
        except OverflowError as exc:
            raise ToolError("Result is too large", error_type="execution_error") from exc
        except (TypeError, ValueError) as exc:
            # e.g. (-8) ** 0.5 yields a complex number, which float() rejects.
            raise ToolError("Result is not a real number", error_type="execution_error") from exc
        except RecursionError as exc:
            raise ToolError("Expression is nested too deeply", "invalid_arguments") from exc
        if not math.isfinite(result):
            raise ToolError("Result is not finite", error_type="execution_error")
        return repr(result)
