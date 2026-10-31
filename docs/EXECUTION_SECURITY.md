# Code Execution Security

**Current policy: code execution is disabled until an OS-enforced sandbox is implemented and verified.**

OxDroid currently allows users to edit and save source files. Pressing **Run** must not execute submitted source code through the app process or a local daemon.

## Enforced behavior

- `EngineCore.executeCode()` saves the requested source file, then returns an explicit unavailable message. It does not invoke the daemon or the legacy language evaluators.
- `ox_industrial_daemon.py` is a fail-closed HTTP endpoint. POST requests receive HTTP 501 and a structured `unsupported` response. The request body is not passed to a process or interpreter.
- A temporary directory, subprocess timeout, working-directory change, or local TCP binding alone is not a security sandbox.
- The editor must not display a successful execution status when execution is unavailable.

## Requirements before enabling Run

A future execution provider must enforce isolation at the operating-system boundary, not with source-text filtering. It must demonstrate:

1. Separate process identity / Android isolated process or an equivalently strong Linux isolation boundary.
2. No access to app credentials, signing keys, private workspace files, or unrelated app data.
3. Default-deny network access, with explicit opt-in where a language workflow requires it.
4. CPU, memory, storage, process-count, and wall-clock limits enforced outside the submitted program.
5. Safe cancellation, bounded output capture, and cleanup after crashes or timeouts.
6. Automated escape/containment tests plus Android device tests across supported API levels and ABIs.

Until those conditions pass review, Run remains unavailable. Editing, saving, project navigation, and hosted CI builds remain separate workflows.

AI intent routing remains disabled.
