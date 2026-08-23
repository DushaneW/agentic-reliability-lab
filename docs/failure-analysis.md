# Failure Analysis

## How diagnosis works

`arl/analysis/engine.py::diagnose(task, trajectory, evaluation)` runs a
fixed set of rule-based detectors (`arl/analysis/detectors.py`) over the
trajectory's event log and picks the highest-weight match:

| Category | Detector | What it looks for |
|---|---|---|
| `timeout_failure` | `detect_timeout` | Run ended via the timeout path |
| `looping_failure` | `detect_looping` | 2+ consecutive identical tool calls |
| `error_recovery_failure` | `detect_error_recovery_failure` | A tool error immediately followed by retrying the exact same failing call |
| `tool_execution_failure` | `detect_repeated_tool_errors` | The same tool failed 2+ times |
| `premature_termination` | `detect_premature_termination` | Agent finished, unsuccessfully, using less than half its step budget |
| `tool_selection_failure` | `detect_tool_selection_failure` | At least one tool error with no clear retry/loop pattern (weakest signal) |
| `unknown_failure` | (fallback) | Failed, but no detector matched |

Every diagnosis carries `evidence`: a description plus the specific
`event_ids` that triggered it, so "why was this classified as looping" is
always answerable by pointing at concrete events, not a paragraph of
LLM-generated prose. Confidence is a simple function of detector weight
(`0.5 + weight * 0.5`, capped at 0.95) — never presented as a calibrated
probability, unlike the reliability model's `predict_proba`.

## What this deliberately does not do

There is no LLM call anywhere in the failure analysis engine. This was a
choice, not a limitation to apologize for: a rule-based system's diagnoses
are exactly as trustworthy as the rules, and every rule here is inspectable
in `arl/analysis/detectors.py`. The tradeoff is coverage — `unknown_failure`
exists because some failure shapes (e.g. an agent that silently computes
the wrong answer with zero retries, zero loops, zero tool errors) genuinely
don't match any of the patterns above. See real (unforced) examples of
this in a full `arl reliability train` run.

## Inspecting a diagnosis

```bash
uv run arl runs analyze <RUN_ID>
```

```text
Run 8f29c1a4-...
Diagnosis: looping_failure
Confidence: 0.83
Wasted tool calls: 2
Recovered: False
  - 4 events are part of an identical repeated tool call (events: a1b2c3d4, ...)
```
