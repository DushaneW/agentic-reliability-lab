# Limitations

This document exists because a research project that only lists its
strengths isn't trustworthy. Read this before citing any number from
`experiments.md`.

## No real LLM agent

The single biggest limitation: **the reference agent is not a real LLM.**
It's a deterministic scripted policy (`arl/agents/reference.py`) with a
`FaultProfile` mechanism for injecting synthetic incompetence. This was a
deliberate choice given the constraints of the environment this project
was built in (no API key wired in), not a simplification made for
convenience that happens to be free of consequences. Concretely, this
means:

- Every result in `experiments.md` describes whether the detection
  pipeline can find signal in *synthetic* failure trajectories, not
  whether it would work on a real agent's trajectories.
- The failure modes a real LLM agent exhibits (hallucinated tool
  arguments, misunderstanding task instructions, subtle reasoning errors
  that don't show up as tool errors at all, context window degradation
  over long trajectories) are not represented here. The `FaultProfile`
  faults (wrong tool, repeat action, ignore error, premature termination)
  are a plausible-looking but unvalidated proxy for real agent failure
  *shapes*, not a simulation of real agent failure *causes*.
- The 1.000 AUROC at 100% trajectory completion (see `experiments.md`) is
  a symptom of this: synthetic faults are, by construction, tightly
  coupled to the features that detect them. A real agent's trajectory
  would not be this clean.

`arl/agents/llm.py` now implements that adapter (Anthropic Messages API and
any OpenAI-compatible endpoint such as Ollama). It is tested with fake
transports only and **has never been run against a live model in this
repository**. Running it on `benchmarks/diverse`, importing the trajectories,
and re-running the pipeline is still the most important next step for making
any claim about real agents. `arl reliability score` can score external JSONL
trajectories (`arl/importers/jsonl.py`), but the predictor is trained on
synthetic data and its scores may not transfer.

## Narrow task suite

All 23 *generated* benchmark tasks are the same underlying operation: sum the numbers
in `input.txt`, write the result to `output.txt`. Difficulty and category
labels (`coding`, `tool_use`, `error_recovery`, `adversarial`) organize
the benchmark/evaluation infrastructure and are exercised by the runner,
grader, and CLI — but they do not represent genuinely different task
*types*, because the reference agent only implements one generic strategy
and would not solve a structurally different task (e.g. "debug this
function," "find the file matching X pattern"). Building a task suite with
real category diversity requires either a general-purpose agent (see
above) or writing category-specific scripted strategies, which was
explicitly avoided (see `scripts/generate_benchmark_tasks.py`'s docstring)
because per-task scripted logic would make "agent failure" analysis
meaningless — failures would be bugs in the task-specific script, not
agent behavior.

`benchmarks/diverse` adds 10 hand-written tasks of different shapes for real
agents. The reference agent cannot solve them, so none of the reported
reliability numbers use them.

## Failure detectors are unvalidated

The rule-based detectors were written by the same person who wrote the fault
injector and were only checked on that injector's output and on 5 hand-built
scenarios. Four enum categories have no detector at all, and
`tool_selection_failure` is a catch-all for any isolated tool error. See
`docs/failure-analysis.md`. Precision/recall of the detectors on real
trajectories is unknown.

## Reliability model caveats

- **Collinearity, not causal features.** As noted in `experiments.md`,
  `tool_error_count`'s negative coefficient doesn't have a clean causal
  reading — it's confounded with other features in this dataset.
  Logistic regression coefficients here should be read as "the model
  found linearly separable structure using these features," not as "this
  feature causes failure."
- **Grouped CV exists, but intervals are optimistic.** `arl reliability cv`
  reports grouped cross-validation with cluster-bootstrap intervals. The
  bootstrap resamples runs, not seeds, and seeds of one (task, profile) are
  near-duplicates of a deterministic policy, so the true uncertainty is larger
  than the printed intervals.
- **Calibration is only partly measured.** Brier score is reported, but there
  is no reliability diagram, and `arl/experiments/intervention.py` treats
  `predict_proba` as a calibrated probability. It may be over- or
  under-confident.
- **Early warning at 20% is close to chance** (logistic AUROC 0.596), and the
  usable signal appears from 40% onward.
- **Class balance via `class_weight="balanced"`**, not resampling —
  this changes the loss function's weighting but doesn't change the
  underlying data distribution, and its effect on the reported metrics
  (particularly precision/recall at the 0.5 threshold) hasn't been
  compared against an unweighted baseline.

## Intervention mechanism is minimal

`arl/experiments/intervention.py` implements exactly one strategy: force
`finish` when predicted failure probability crosses a threshold. This
trades off wasted steps against any chance of recovery — it cannot
improve success rate, because it has no way to change the agent's
behavior other than stopping it. Real interventions like "re-plan" or
"critic" (mentioned in the original scope for this kind of system) would
require a second model call to actually generate an alternative action,
which this system does not have without a real LLM. Section 18's original
"success/cost/latency/recovery rate" comparison table is not fully
implemented — recovery rate is meaningless for a stop-only intervention,
since it can only ever be zero.

## Shell tool sandboxing

Documented in detail in `SECURITY.md`. Summary: process-level restriction
(allow-list, empty env, temp dir, timeout), not container isolation. No
Docker daemon was available in the build environment, so this was not
upgraded to real sandboxing, and the repository's `Dockerfile` containers
the *application*, not individual tool calls.

## What's NOT measured

- Cost (in the "how much would this cost against a real API" sense) — not
  applicable without a real LLM, so no cost metrics are reported anywhere,
  despite cost being mentioned in the original project scope.
- Latency — the reference agent has no meaningful latency to measure;
  `wall_time_seconds` in results reflects Python/subprocess overhead, not
  anything representative of real agent response times.
- Generalization across different SQLite-scale data volumes, concurrent
  runs, or anything resembling production load. This is a local-first
  single-user tool and has only been exercised at the scale of hundreds of
  runs on one machine during development.
