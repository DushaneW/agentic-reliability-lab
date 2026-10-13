from __future__ import annotations

import pytest

from arl.tools.base import ToolError
from arl.tools.calculator import CalculatorTool


def test_calculator_basic_arithmetic() -> None:
    tool = CalculatorTool()
    result = tool.run({"expression": "(3 + 4) * 2"})
    assert result.success
    assert float(result.output) == 14.0


def test_calculator_rejects_non_arithmetic() -> None:
    tool = CalculatorTool()
    result = tool.run({"expression": "__import__('os').system('echo hi')"})
    assert not result.success
    assert "invalid_arguments" in (result.error or "")


def test_calculator_rejects_name_lookup() -> None:
    tool = CalculatorTool()
    result = tool.run({"expression": "os.getcwd()"})
    assert not result.success


def test_calculator_handles_division_by_zero() -> None:
    tool = CalculatorTool()
    result = tool.run({"expression": "1/0"})
    assert not result.success
    assert "execution_error" in (result.error or "")


def test_calculator_missing_expression_raises() -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolError):
        tool.execute({})


# --- inputs that used to raise out of the tool ------------------------------


def test_overflow_is_a_tool_error_not_a_crash() -> None:
    tool = CalculatorTool()
    for expression in ("9**9**9", "10**400", "1e308 * 1e308"):
        result = tool.run({"expression": expression})
        assert not result.success, expression
        assert "unexpected_error" not in (result.error or ""), expression


def test_complex_result_is_a_tool_error() -> None:
    result = CalculatorTool().run({"expression": "(-8) ** 0.5"})
    assert not result.success
    assert "not a real number" in (result.error or "")


def test_deeply_nested_expression_is_a_tool_error() -> None:
    result = CalculatorTool().run({"expression": "(" * 3000 + "1" + ")" * 3000})
    assert not result.success
    result = CalculatorTool().run({"expression": "-" * 5000 + "1"})
    assert not result.success


def test_non_string_expression_is_rejected() -> None:
    result = CalculatorTool().run({"expression": 42})
    assert not result.success
    assert "invalid_arguments" in (result.error or "")


def test_null_byte_in_expression_is_rejected() -> None:
    result = CalculatorTool().run({"expression": "1 + 1\x00"})
    assert not result.success
