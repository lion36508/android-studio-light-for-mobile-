# OxDroid Studio — Engineering Issue Documents

This folder contains implementation-ready bug reports based on a read-only review of the current `main` branch.

## Scope and evidence

Reviewed source files:
- `src/main/java/com/bikash/mobileide/OxLinuxEngine.java`
- `src/main/java/com/bikash/mobileide/EngineCore.java`
- `src/main/java/com/bikash/mobileide/MainActivity.java`
- `src/main/java/com/bikash/mobileide/EditorActivity.java`

These documents describe source-level defects and proposed fixes. They are not claims that a runtime/device reproduction has already been completed. Each issue includes reproduction guidance and acceptance tests.

## Issue documents

| Priority | Document | Problem |
|---|---|---|
| P1 | [001-process-timeout-and-output-deadlock.md](001-process-timeout-and-output-deadlock.md) | Process output is read before timeout is enforced; commands can hang indefinitely |
| P1 | [002-safe-complete-file-io.md](002-safe-complete-file-io.md) | File reads can be partial; file writes are not atomic; save failures are hidden by execution |
| P1 | [003-rootfs-extraction-atomic-recovery.md](003-rootfs-extraction-atomic-recovery.md) | Rootfs extraction failures are swallowed and partial extraction can be mistaken for a valid install |
| P1 | [004-project-metadata-persistence.md](004-project-metadata-persistence.md) | User-created project entries exist only in memory and disappear from the dashboard after process restart |
| P1 | [005-accurate-execution-results.md](005-accurate-execution-results.md) | Execution reports success/exit code 0 even when the command or program fails |

## Relationship to existing GitHub issues

The repository already has open issues covering a real build backend (#2), editor intelligence (#3), visual layout design (#4), low-memory compilation (#5), and Logcat (#6). The documents here focus on additional source-level reliability defects and should be cross-linked to those issues where relevant rather than duplicating their feature requests.

## Recommended implementation order

1. Fix process lifecycle and output draining (001).
2. Fix file I/O and propagate save failures (002).
3. Make rootfs extraction verifiable and recoverable (003).
4. Persist project metadata (004).
5. Return truthful execution results (005).

No application source code is changed by adding these documentation files.
