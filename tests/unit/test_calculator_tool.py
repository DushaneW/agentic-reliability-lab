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
