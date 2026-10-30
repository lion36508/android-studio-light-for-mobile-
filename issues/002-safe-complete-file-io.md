# [P1] Make workspace file reads complete and writes atomic

## Summary

Workspace file handling has several reliability gaps: `loadFile()` assumes one read fills the entire buffer, save operations write directly over the existing file, and `executeCode()` ignores the save result.

## Evidence in current source

File: `src/main/java/com/bikash/mobileide/EngineCore.java`

- Lines 71–75 allocate a byte array using `(int) file.length()`, call `fis.read(data)` once, ignore the returned byte count, and then decode the entire array.
- A stream read is allowed to return fewer bytes than requested. The resulting text can be truncated or contain zero-filled trailing bytes.
- Lines 49–55 write directly to the destination file. If the process is interrupted during a save, the previous valid file may already have been truncated.
- Lines 64–78 return the default content on any exception, making a real read failure indistinguishable from a missing file.
- Lines 186–188 call `saveFile()` but ignore its boolean result before executing the supplied code.

## User impact

- Large or interrupted reads may open incomplete source text.
- A failed save can destroy the last good version of a file.
- The editor may show starter/default text when the real file could not be read.
- RUN can execute code that was not successfully saved, making the visible workspace and executed content disagree.

## Reproduction plan

1. Create a large source file and repeatedly open it; compare its byte length and content with the original.
2. Simulate an I/O failure or process interruption during a save and verify that the previous file remains intact.
3. Make the workspace unwritable (or use a controlled failing test stream), then invoke RUN and verify execution is refused with a visible save error.
4. Try opening a missing file and a deliberately unreadable file; ensure the UI can distinguish those states.

## Recommended fix

### Reading

- Use a buffered stream and a loop until EOF, or use a well-tested `Files.readAllBytes()` implementation if supported by the app's Android/API constraints.
- Avoid blindly casting an arbitrary file length to `int`; impose a documented maximum source-file size and return a clear error when exceeded.
- Use try-with-resources so streams always close.
- Distinguish `NOT_FOUND`, `READ_ERROR`, `TOO_LARGE`, and `SUCCESS`. Do not silently substitute default content for a read error.
- When creating a missing file, report whether the default file was successfully created.

### Writing

- Write to a temporary sibling file first.
- Flush and sync the temporary file.
- Close it successfully, then atomically rename/replace the destination where the filesystem supports it.
- Preserve the previous file if the temporary write fails.
- Clean up abandoned temporary files safely.
- Keep the canonical-path traversal checks already present in `workspaceFile()`.

### Callers

- Change execution flow to stop immediately if saving fails.
- Show a clear error in the editor and do not run stale or unsaved content unless the user explicitly chooses a documented run-without-saving action.
- Propagate a typed result rather than collapsing all failures into a boolean/string default.

## Acceptance criteria

- [ ] Reads loop until EOF and return the exact original bytes for small, large, and multibyte UTF-8 files.
- [ ] A partial read cannot silently produce truncated or padded source text.
- [ ] Files above the configured maximum size fail with a clear message and no excessive allocation.
- [ ] A failed write preserves the previous valid file.
- [ ] A successful save replaces the destination completely and durably as far as Android filesystem guarantees allow.
- [ ] Missing file and read error are distinct outcomes.
- [ ] RUN/BUILD never proceeds after a failed required save.
- [ ] Unit/instrumentation tests cover UTF-8, empty files, large files, interrupted writes, permission failures, and path traversal.

## Priority

P1 — this risks source corruption and running code different from what the user believes is saved.
