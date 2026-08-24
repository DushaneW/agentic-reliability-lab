from __future__ import annotations

from arl.agents.base import ActionKind, AgentDecision
from arl.agents.reference import FaultProfile, ReferenceAgent, agent_from_task_metadata
from arl.domain.task import GraderSpec, Task, TaskCategory, TaskDifficulty


def test_scripted_step_overrides_plan_at_exact_index() -> None:
    agent = ReferenceAgent(
        scripted_steps={
            0: AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name="filesystem",
                arguments={"op": "bogus"},
                rationale="forced",
            )
        }
    )
    decision = agent.decide("desc", [], [])
    assert decision.tool_name == "filesystem"
    assert decision.arguments == {"op": "bogus"}


def test_unscripted_calls_fall_through_to_normal_plan() -> None:
    # No scripted_steps at all -> must behave exactly like a plain agent.
    plain = ReferenceAgent(seed=0)
    scripted = ReferenceAgent(seed=0, scripted_steps={})
    d1 = plain.decide("desc", [], [])
    d2 = scripted.decide("desc", [], [])
    assert d1 == d2


def test_scripted_finish_decision() -> None:
    agent = ReferenceAgent(scripted_steps={0: AgentDecision(kind=ActionKind.FINISH)})
    decision = agent.decide("desc", [], [])
    assert decision.kind == ActionKind.FINISH


def test_only_the_scripted_index_is_overridden() -> None:
    # index 1 is scripted; index 0 and index 2 must use the normal plan.
    agent = ReferenceAgent(
        scripted_steps={
            1: AgentDecision(
                kind=ActionKind.TOOL_CALL, tool_name="calculator", arguments={"expression": "1/0"}
            )
        }
    )
    first = agent.decide("desc", [], [])
    assert first.tool_name == "filesystem"  # normal plan step 0: list
    second = agent.decide("desc", [], [])
    assert second.tool_name == "calculator"  # scripted override
    assert second.arguments == {"expression": "1/0"}


def _task(metadata: dict) -> Task:  # noqa: ANN001
    return Task(
        id="t1",
        description="d",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.EASY,
        grader=GraderSpec(kind="numeric_file_equals", config={"path": "output.txt", "expected": 0}),
        metadata=metadata,
    )


def test_agent_from_task_metadata_with_no_special_keys_is_a_clean_agent() -> None:
    task = _task({"initial_files": {"input.txt": "1 2"}})
    agent = agent_from_task_metadata(task, seed=0)
    assert agent.fault_profile.is_clean
    assert agent.scripted_steps == {}


def test_agent_from_task_metadata_builds_fault_profile() -> None:
    task = _task({"fault_profile": {"repeat_action_rate": 1.0}})
    agent = agent_from_task_metadata(task, seed=0)
    assert agent.fault_profile == FaultProfile(repeat_action_rate=1.0)


def test_agent_from_task_metadata_builds_scripted_steps() -> None:
    task = _task(
        {
            "scripted_steps": {
                0: {"tool_name": "filesystem", "arguments": {"op": "bogus"}},
                1: {"kind": "finish"},
            }
        }
    )
    agent = agent_from_task_metadata(task, seed=0)
    assert agent.scripted_steps[0].tool_name == "filesystem"
    assert agent.scripted_steps[0].arguments == {"op": "bogus"}
    assert agent.scripted_steps[1].kind == ActionKind.FINISH


def test_agent_from_task_metadata_string_keys_from_yaml_are_coerced_to_int() -> None:
    # YAML mapping keys can round-trip as strings depending on the loader;
    # the factory must not silently drop scripted steps because of that.
    task = _task(
        {"scripted_steps": {"0": {"tool_name": "filesystem", "arguments": {"op": "list"}}}}
    )
    agent = agent_from_task_metadata(task, seed=0)
    assert 0 in agent.scripted_steps
