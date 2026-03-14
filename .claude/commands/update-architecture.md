Re-audit the codebase and update the architecture map.

Steps:

1. **Read current state**: Read `architecture.jsonld` and all source files in `whisperSSTis/`, `app.py`, `launcher.py`
2. **Diff against map**: Identify:
   - New modules, functions, or dependencies not in the graph
   - Removed or renamed components
   - Changed data flows or relationships
   - New environment variables or configuration
3. **Update JSON-LD**:
   - Preserve all existing `@id` values
   - Add new nodes with new IDs
   - Set `arch:deprecated: true` on removed nodes (do not delete them)
   - Update `arch:lastAuditedAt` on every touched node
   - Update `arch:lastAuditedBy` to `claude-code/architecture-update`
4. **Update CLAUDE.md** if structural changes affect the documented architecture
5. **Report**: List all changes made to the architecture map

Do NOT modify any application source code.