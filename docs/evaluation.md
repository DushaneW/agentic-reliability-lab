# Evaluation

## Why no LLM judges

The evaluation engine (`arl/evaluation/engine.py`) only supports
deterministic graders (`arl/evaluation/graders.py`). This is a hard
design constraint, not an oversight: a benchmark whose ground truth
depends on another model's judgment is not reproducible in the way this
project needs it to be — the reliability dataset (`arl/reliability/
dataset.py`) generates thousands of labeled examples from grading results,
and a nondeterministic or API-dependent grader would make that dataset
non-reproducible and expensive to regenerate.

If you need subjective or open-ended grading (e.g. "is this code
well-written"), that's legitimately out of scope for the core system as
built. The `Grader` protocol in `arl/evaluation/graders.py` is the
extension point if you want to add one for your own use, understanding
that it will make anything downstream of it (the reliability dataset in
particular) nondeterministic.

## How grading works

```python
from arl.evaluation.engine import evaluate

evaluation_result = evaluate(task, task_result)
# EvaluationResult(run_id=..., task_id=..., success=bool, score=float, ...)
```

`evaluate` looks up the grader named in `task.grader.kind`
(`arl/evaluation/graders.py::GRADERS`) and calls it with the task's
`grader.config` and the run's `TaskResult` (which includes the final
virtual filesystem snapshot and termination reason — see
`arl/runtime/runner.py`).

## Writing a new grader

```python
class MyGrader:
    def grade(self, spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
        # spec.config is whatever the task YAML put under `grader.config`
        # return (success, score in [0,1], human-readable detail)
        ...

GRADERS["my_grader"] = MyGrader()
```

See `examples/custom_benchmark.py` for a worked example.
