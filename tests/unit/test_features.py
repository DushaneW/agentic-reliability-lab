from __future__ import annotations

from arl.domain.events import EventType, TrajectoryEvent
from arl.reliability.features import extract_features


def _event(step: int, event_type: EventType, payload: dict) -> TrajectoryEvent:
    return TrajectoryEvent(run_id="r1", step_index=step, event_type=event_type, payload=payload)


def test_features_on_empty_trajectory_do_not_divide_by_zero() -> None:
    features = extract_features([])
    assert features.values["tool_call_count"] == 0
    assert features.values["tool_error_rate"] == 0.0
    assert features.values["repeated_action_ratio"] == 0.0


def test_repeated_action_ratio_detects_repeats() -> None:
    args = {"tool_name": "calculator", "arguments": {"expression": "1+1"}}
    events = [
        _event(0, EventType.TOOL_CALL, args),
        _event(1, EventType.TOOL_CALL, args),
    ]
    features = extract_features(events)
    assert features.values["repeated_action_ratio"] == 0.5  # 1 repeated pair / 2 calls


def test_tool_error_rate_computed_over_tool_calls() -> None:
    events = [
        _event(0, EventType.TOOL_CALL, {"tool_name": "calculator", "arguments": {}}),
        _event(1, EventType.TOOL_ERROR, {"tool_name": "calculator"}),
        _event(2, EventType.TOOL_CALL, {"tool_name": "calculator", "arguments": {}}),
    ]
    features = extract_features(events)
    assert features.values["tool_call_count"] == 2
    assert features.values["tool_error_count"] == 1
    assert features.values["tool_error_rate"] == 0.5


def test_feature_vector_as_list_matches_names_order() -> None:
    from arl.reliability.features import FEATURE_NAMES

    features = extract_features([])
    assert len(features.as_list()) == len(FEATURE_NAMES)


def test_consecutive_error_pairs_counts_back_to_back_failed_calls() -> None:
    call = {"tool_name": "calculator", "arguments": {}}
    events = [
        _event(0, EventType.TOOL_CALL, call),
        _event(1, EventType.TOOL_ERROR, {"tool_name": "calculator"}),
        _event(2, EventType.TOOL_CALL, call),
        _event(3, EventType.TOOL_ERROR, {"tool_name": "calculator"}),
        _event(4, EventType.TOOL_CALL, call),
        _event(5, EventType.TOOL_RESULT, {"tool_name": "calculator", "success": True}),
        _event(6, EventType.TOOL_CALL, call),
        _event(7, EventType.TOOL_ERROR, {"tool_name": "calculator"}),
    ]
    # error, error, success, error -> exactly one adjacent error pair.
    assert extract_features(events).values["consecutive_error_pairs"] == 1
