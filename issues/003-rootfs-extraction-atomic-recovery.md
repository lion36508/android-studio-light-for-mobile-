# [P1] Make OxLinux root filesystem extraction fail visibly and recover safely

## Summary

`OxLinuxEngine.extractIfNeeded()` suppresses every extraction exception and can leave a partially extracted root filesystem. It also writes the `.ready` marker without validating that required files were extracted successfully.

## Evidence in current source

File: `src/main/java/com/bikash/mobileide/OxLinuxEngine.java`

- Lines 23–25 treat the presence of `.ready` as sufficient proof that extraction is complete.
- Lines 28–53 extract the asset and create the marker after the loop.
- Line 55 catches every exception and ignores it.
- There is no validation that required executables/directories exist or that expected files have usable permissions before the environment is considered ready.
- If extraction fails after writing some files, the partial tree remains. A later retry may overwrite some files but does not explicitly clean/reconcile stale partial contents.

The existing canonical-path check is a useful security control and must be retained.

## User impact

- Linux commands may fail later with confusing “not found” or permission errors.
- The user receives no explanation that installation/extraction failed.
- A partial rootfs can persist across launches and cause repeatable failures.
- Disk-full, corrupt asset, and permission problems are difficult to diagnose.

## Reproduction plan

1. Use a test asset that is truncated/corrupt and construct the engine; verify a clear extraction failure is surfaced.
2. Simulate an I/O failure partway through extraction; confirm no ready marker is created.
3. Restart after a partial failure; verify recovery starts from a clean staging directory.
4. Include a ZIP entry containing traversal segments and verify extraction rejects it without writing outside the root.
5. Verify executable permissions for required binaries after a successful extraction.

## Recommended fix

1. Extract into a staging directory such as `oxlinux.tmp-<unique-id>`, not directly into the live root directory.
2. Validate every ZIP entry:
   - canonical destination must remain under the staging root;
   - reject absolute paths and traversal;
   - enforce maximum total uncompressed size, per-entry size, and entry count to reduce ZIP-bomb/disk-exhaustion risk;
   - handle directory creation and file-write failures as fatal.
3. Use try-with-resources for asset, ZIP, and output streams.
4. Check every write, close, and permission-setting operation; do not ignore failures.
5. Verify a defined list of required directories and binaries, and verify expected executable bits.
6. Write a manifest containing asset/version/hash and extraction completion metadata only after validation.
7. Promote staging to the live root directory using a safe rename strategy. If replacement is not atomic on the target filesystem, use a backup/rollback sequence.
8. Create the ready marker last. On startup, verify marker version and manifest; do not trust a marker from an older asset version.
9. Return a typed status or throw a handled exception. Display an actionable error to the user and provide a retry action.
10. On failure, remove only the staging directory created by this attempt; never recursively delete arbitrary paths.

## Acceptance criteria

- [ ] A valid rootfs extracts and passes required-file/permission checks.
- [ ] Corrupt or truncated assets produce a visible actionable error.
- [ ] A failed extraction never creates `.ready`.
- [ ] A retry recovers from partial prior extraction without mixing old and new files.
- [ ] Traversal entries are rejected and cannot escape the root directory.
- [ ] Entry count and uncompressed-size limits are enforced.
- [ ] Asset version/hash changes trigger a controlled refresh.
- [ ] No broad empty catch hides extraction failures.
- [ ] Tests cover valid ZIP, corrupt ZIP, traversal entry, disk/write failure, missing required binary, and retry.

## Priority

P1 — this is the initialization path for the Linux environment and failures currently become delayed, hard-to-diagnose runtime errors.
