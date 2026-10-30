# [P1] Enforce process timeouts while continuously draining output

## Summary

Commands launched by `OxLinuxEngine.exec()` can run longer than the intended 60-second limit, and can hang indefinitely when the child process writes enough output to fill its pipe.

## Evidence in current source

File: `src/main/java/com/bikash/mobileide/OxLinuxEngine.java`

- Lines 67–70 start the process and synchronously read its output until EOF.
- Only after the read loop finishes, lines 73–76 call `waitFor(60, TimeUnit.SECONDS)`.
- Therefore the timeout is not active while waiting for output EOF. A command that never closes stdout/stderr can prevent execution from reaching the timeout check.
- If output is large, reading on the calling thread can also block other work depending on the caller.

Related path: `EngineCore.runShellCommand()` reads output and calls unbounded `waitFor()` (around lines 81–96), so shell execution has no enforced deadline there either.

## User impact

- The terminal may remain stuck even though a 60-second timeout is advertised.
- Long-running or silent commands can make the app appear frozen.
- A noisy process can consume memory because all output is accumulated in one `StringBuilder`.
- A process may outlive the screen that started it if cancellation/lifecycle handling is not added.

## Reproduction plan

1. Call `OxLinuxEngine.exec("sleep 90")`. It should return near the configured deadline, but the current code can wait for the command to finish before checking the deadline.
2. Call a command that continuously writes output and then sleeps. Verify that the UI remains responsive and that the timeout still fires.
3. Run a command that spawns a child process and verify that cancellation does not leave the child running.
4. Repeat on the minimum supported Android API and a current Android release.

## Recommended fix

1. Start the process and drain stdout/stderr concurrently with process waiting. Use a dedicated executor or reader thread; do not wait for EOF before starting the timeout clock.
2. Use a monotonic deadline (for example, `System.nanoTime()`) and calculate remaining time for every wait.
3. On timeout:
   - call `destroy()`;
   - wait a short grace period;
   - call `destroyForcibly()` if still alive;
   - on supported Android versions, terminate descendants where the API permits;
   - close process streams and join the reader task.
4. Keep output bounded. Retain the last configurable number of bytes/characters and append an explicit truncation marker rather than allowing unbounded memory growth.
5. Return a structured result containing stdout, stderr (or merged output), exit code, timed-out flag, duration, and launch error. Keep a string formatter only at the UI boundary.
6. Apply the same process runner to `EngineCore.runShellCommand()` so there is one tested timeout/output implementation.
7. Provide cancellation tied to the editor/activity lifecycle; avoid updating a destroyed Activity.

## Suggested implementation shape

Create a reusable process runner with:
- command and working directory;
- environment overrides;
- timeout and output-size limit;
- output consumer or bounded collector;
- cancellation handle;
- structured result.

Do not implement this by simply moving `waitFor(60, ...)` above the read loop: that can deadlock if the child blocks while writing to a full pipe. Output must be drained concurrently.

## Acceptance criteria

- [ ] A command exceeding the configured deadline is terminated and returns a timeout result within a small grace margin.
- [ ] A silent command cannot bypass the timeout.
- [ ] A command producing output faster than the consumer can display it does not deadlock.
- [ ] Output memory use is bounded and truncation is reported.
- [ ] The process exit code is captured accurately.
- [ ] Cancellation terminates the process and does not leave a background reader thread.
- [ ] UI remains responsive while a command is running.
- [ ] Automated tests cover success, non-zero exit, timeout, large output, silent output, and cancellation.

## Priority

P1 — process execution is a core IDE feature, and the current timeout check does not protect the blocking read operation.
