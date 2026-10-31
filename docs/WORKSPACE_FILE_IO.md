# Workspace file I/O guarantees

OxDroid uses the application's private `files/workspace` directory for editor source files.

## Writes

- Source files are written through Android's `AtomicFile`.
- A write first goes to a sibling staging/backup path. The old file is retained until the new contents are flushed, synced, and committed.
- If a write fails, `failWrite()` asks `AtomicFile` to restore its backup.
- Workspace path resolution rejects absolute paths, traversal outside the workspace, directory targets, and failed parent-directory creation.

## Reads

- Reads use an 8 KiB buffer and continue until EOF; they do not assume one `read()` call fills the file.
- A genuinely missing file is initialized from the caller's starter content.
- Permission, I/O, and other read errors return an explicit failed `WorkspaceReadResult`; they are not treated as missing files and are not replaced with starter content.
- The editor disables editing, saving, Run, and Build for a file that could not be loaded safely. It remains possible to leave the editor or switch to another readable file.
- Legacy `loadFile()` callers receive an exception on a read error rather than silently getting default content.

## Scope

AtomicFile protects an individual file from partial writes and recovers its backup on the next read. It is not a multi-file project transaction or a cloud backup. Project catalog persistence and multi-file scaffolding are tracked separately.
