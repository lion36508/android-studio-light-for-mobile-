# [P1] Persist user-created project metadata across app restarts

## Summary

Projects created from the dashboard are added to the in-memory `projectList`, but the list is reset to four hard-coded demo projects during every `MainActivity.onCreate()`. The project source file may remain in the workspace, but its dashboard project entry and metadata are not restored.

## Evidence in current source

File: `src/main/java/com/bikash/mobileide/MainActivity.java`

- Lines 34–35 declare an in-memory `ArrayList` and SharedPreferences field.
- Lines 59–65 call `initDefaultProjects()` on every Activity creation.
- Lines 67–109 clear the list and add four fixed sample entries.
- Lines 605–612 add a newly created project only to `projectList`, refresh the UI, and open the editor.
- The reviewed code does not persist the new project's name, language, filename, template/type, or ordering to durable metadata.

The source file itself is saved through `EngineCore`, so this is specifically a project-catalog/metadata persistence defect, not proof that every source file is deleted.

## User impact

- A user-created project can disappear from the dashboard after the app process is killed or the app is restarted.
- Users may have to find the file manually through the workspace explorer.
- Project names, type/language labels, and dashboard ordering are lost.
- The dashboard can look as if the project was never created, even though its source file remains.

## Reproduction plan

1. Create a new Java or Python project from the dashboard.
2. Confirm it appears in the project list and its file exists in the workspace.
3. Force-stop the app and relaunch it.
4. Observe whether the new project is restored in the dashboard.
5. Repeat after creating multiple projects and after renaming/deleting a project if those operations are supported.

## Recommended fix

1. Introduce a versioned project-catalog model with stable project IDs. Store at least:
   - project ID;
   - display name;
   - language;
   - primary relative workspace path;
   - project/template type;
   - creation and last-opened timestamps;
   - schema version.
2. Persist the catalog in app-private storage. Use a transactional database (Room/SQLite) if the catalog will grow or support rename/delete/search; a versioned JSON file can be an acceptable small first step if writes are atomic.
3. Keep paths relative to the workspace root. Validate them through the existing canonical-path protection before opening.
4. Load the catalog before rendering the dashboard. Migrate the current four demo entries into sample/demo status rather than re-adding them as user projects on every launch.
5. Handle missing source files explicitly: mark the project unavailable and offer locate/remove actions rather than silently creating a new default file.
6. Persist create/rename/delete/reorder operations before reporting success to the user.
7. Avoid storing full source contents in the project catalog; keep source files in the workspace and metadata in the catalog.
8. Add schema migration/version handling so future releases can extend metadata without losing existing projects.

## Acceptance criteria

- [ ] A newly created project remains listed after force-stop and relaunch.
- [ ] Project name, language, type, and primary file path survive restart.
- [ ] Multiple projects retain stable identities and ordering.
- [ ] Missing project files are reported without silently overwriting or recreating user data.
- [ ] Invalid/out-of-workspace paths are rejected.
- [ ] A failed catalog write is reported and does not show a false success state.
- [ ] Existing default/demo projects are migrated or clearly distinguished from user projects.
- [ ] Tests cover fresh install, upgrade/migration, restart, missing file, malformed catalog, and interrupted persistence.

## Priority

P1 — project discovery and continuity are essential for a mobile development workspace.
