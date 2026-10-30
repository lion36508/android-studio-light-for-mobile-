# [P1] Report real save, exit, and execution status instead of unconditional success

## Summary

The execution pipeline can report “Process finished with exit code 0” even when saving failed or the interpreted/compiled program encountered an error. This makes the terminal output unreliable.

## Evidence in current source

File: `src/main/java/com/bikash/mobileide/EngineCore.java`

- Lines 186–188 call `saveFile(context, fileName, code)` and ignore the returned boolean.
- Lines 223–225 append an error message when an exception occurs.
- Lines 227–230 then unconditionally append `Process finished with exit code 0`, regardless of whether an exception was caught.
- In the daemon path, the code accepts only HTTP 200 and extracts an `output` field using string slicing; non-200 responses and malformed JSON are reduced to a null fallback without a structured diagnostic.

File: `src/main/java/com/bikash/mobileide/EditorActivity.java`

- `runCurrentCode()` saves first, which is good, but the engine's public execution method can also be called from other paths and should enforce its own preconditions.
- The UI displays the returned text as terminal output, so inaccurate engine status is shown directly to users.

## User impact

- A failing program can look successful.
- Users cannot reliably distinguish program output from engine diagnostics.
- A save failure can be followed by execution of unsaved input in other call paths.
- Automated tests and troubleshooting become unreliable because the exit status is fabricated.

## Reproduction plan

1. Run code that deliberately throws an exception or causes the interpreter to return an error.
2. Confirm the output includes an error and currently may still end with exit code 0.
3. Force a save failure and call the execution engine directly; verify it refuses to run.
4. Make the local daemon return a non-200 response or malformed JSON and confirm the UI shows a useful diagnostic rather than a generic fallback.

## Recommended fix

1. Replace the plain output string as the internal return type with a structured execution result:
   - status (success, program error, engine error, save error, timeout, cancelled);
   - stdout;
   - stderr/diagnostics;
   - nullable exit code;
   - duration;
   - execution backend identifier.
2. Check the result of `saveFile()`. If it fails, return `SAVE_ERROR` and do not execute.
3. Make interpreter/compiler methods return explicit success/failure and exit status. Do not infer success merely because no Java exception escaped.
4. Append exit code 0 only when the backend actually completed successfully with code 0.
5. For an internal interpreter that cannot provide operating-system exit codes, report a truthful status such as “Interpreter completed” or “Program error”; do not invent an exit code.
6. Parse daemon JSON with a JSON parser (Android JSON APIs are available) rather than substring offsets. Validate response schema, status, output, and error fields.
7. Distinguish connection refused, timeout, HTTP error, malformed response, and backend-reported program failure.
8. Keep UI formatting separate from execution logic so results can be unit-tested.

## Acceptance criteria

- [ ] A successful program is marked successful only when the backend reports success.
- [ ] A runtime/program error is not followed by a success or exit-code-0 message.
- [ ] A save failure prevents execution and is displayed clearly.
- [ ] Non-zero process exit codes are preserved.
- [ ] Timeout and cancellation have distinct statuses.
- [ ] Daemon HTTP/network/JSON failures are surfaced with actionable diagnostics.
- [ ] Tests cover success, thrown error, interpreter error, save failure, non-zero exit, timeout, malformed daemon response, and daemon unavailable.

## Priority

P1 — false success messages undermine trust in every language/runtime feature.
