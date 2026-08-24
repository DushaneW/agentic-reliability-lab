<!--
This file is the real, unedited output of:

    arl benchmark run benchmarks/failures
    arl report generate fef6f0f5

against benchmarks/failures/error_recovery.yaml on the commit that added
this example. It is committed as-is (not hand-written) to show what
`arl report generate` actually produces, including the pre-existing
"Status: succeeded" / "Success: False" inconsistency described in
docs/benchmarks.md (Run.status reflects the agent's own termination
reason, independent of the grader's verdict) -- that inconsistency is
left visible here deliberately rather than edited out.

Regenerate with a fresh run id any time; the run id in the filename and
headings below will differ, everything else should not.
-->

# Run report: fef6f0f5-0b41-4ed8-a2ca-791251992d6a

- Task: `failure-error-recovery-001`
- Agent: `reference`
- Status: `succeeded`
- Steps taken: 5
- Wall time: 0.000s
- Terminated: agent_finished

## Evaluation

- Success: **False**
- Score: 0.0

## Failure diagnosis

- Category: **error_recovery_failure**
- Confidence: 0.95
- Wasted tool calls: 3
  - a tool error was followed by retrying the exact same failing call

## Trajectory

0. `run_started` — {'kind': 'run_lifecycle', 'reason': 'failure-error-recovery-001'}
1. `tool_call` — {'kind': 'tool_call', 'tool_name': 'filesystem', 'arguments': {'op': 'read', 'path': 'does_not_exist.txt'}}
2. `tool_error` — {'kind': 'tool_error', 'tool_name': 'filesystem', 'error_type': 'tool_execution_error', 'message': 'not_found: File not found: does_not_exist.txt'}
3. `tool_call` — {'kind': 'tool_call', 'tool_name': 'filesystem', 'arguments': {'op': 'read', 'path': 'does_not_exist.txt'}}
4. `tool_error` — {'kind': 'tool_error', 'tool_name': 'filesystem', 'error_type': 'tool_execution_error', 'message': 'not_found: File not found: does_not_exist.txt'}
5. `tool_call` — {'kind': 'tool_call', 'tool_name': 'filesystem', 'arguments': {'op': 'read', 'path': 'does_not_exist.txt'}}
6. `tool_error` — {'kind': 'tool_error', 'tool_name': 'filesystem', 'error_type': 'tool_execution_error', 'message': 'not_found: File not found: does_not_exist.txt'}
7. `tool_call` — {'kind': 'tool_call', 'tool_name': 'filesystem', 'arguments': {'op': 'write', 'path': 'output.txt', 'content': '0'}}
8. `tool_result` — {'kind': 'tool_result', 'tool_name': 'filesystem', 'success': True, 'result_summary': 'wrote output.txt'}
9. `run_finished` — {'kind': 'run_lifecycle', 'reason': 'agent_finished'}
