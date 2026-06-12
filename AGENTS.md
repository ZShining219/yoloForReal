# Agent Guide

## Identity And Reply Rules
- Every final result reply to the user must start with `Dear Z`.
- Keep replies concise, warm, and action-oriented. Explain what changed, how it was verified, and what remains.
- If a task is ambiguous, make the safest useful assumption and state it briefly before acting.

## Project Context
- Goal: convert real-world video data into SUMO-ready traffic simulation inputs.
- Keep source videos, generated intermediate data, and SUMO outputs clearly separated.
- Prefer reproducible scripts and documented parameters over one-off notebook-only workflows.
- Optimize for reversible progress: small steps, clear checkpoints, and clean working state.

## Working Conventions
- Read existing files before editing and keep changes narrowly scoped.
- Preserve user data and generated artifacts unless explicitly asked to clean them.
- Use `rg`/`rg --files` for searches when available.
- Avoid committing large media, model weights, cache folders, or generated simulation outputs.
- Prefer simple, local, inspectable solutions before introducing new frameworks or services.
- Keep code and workflow decisions documented only where they help future execution.

## File Hygiene
- Do not create loose documents, scratch files, or ad hoc notes in the project root.
- Durable documentation belongs in `docs/` and should have a clear purpose, owner, and filename.
- Temporary exploration belongs in `.scratch/` or `tmp/`; clean it up before finishing unless it is useful evidence.
- Generated outputs belong in `outputs/`, `data/interim/`, or `data/processed/` and should stay ignored by Git by default.
- Before adding a new top-level directory, check whether an existing directory already fits the purpose.
- If a file is no longer useful after a task, remove it or explicitly mention why it is being kept.

## File Lifecycle And Source Of Truth
- Prefer updating the current authoritative file over creating a parallel replacement.
- Do not create files named like `new`, `final`, `v2`, `backup`, `copy`, or similar version-by-filename variants.
- Before adding a file with overlapping purpose, identify whether it is a new responsibility, a replacement, or a temporary experiment.
- If a new file replaces an old file, migrate the old file to a unified deprecation area instead of leaving it beside active files.
- Use `archive/deprecated/` for deprecated source, config, or documentation that must be retained for reference.
- Deprecated files must include a short note at the top or nearby README entry explaining what replaced them and when.
- Do not allow multiple long-lived implementations of the same pipeline step unless their boundaries are documented.

## Documentation Creation Gate
- Do not create Markdown documents during conversation by default.
- First explain documentation recommendations in the chat and ask for user approval before writing a new Markdown document.
- Only create documentation after the user agrees to turn the recommendation into a file.
- Place approved documentation according to its purpose: `docs/workflows/` for runbooks, `docs/design/` for design notes, `docs/runs/` for experiment records, and `archive/deprecated/` for retained obsolete docs.
- Do not export or drop Markdown files into the project root unless the root file is a recognized project entrypoint such as `README.md`, `AGENTS.md`, or `CHANGELOG.md`.
- When updating existing documentation, edit the authoritative file in place and avoid creating duplicate summaries.

## Codex Task Tracking Mode
- Maintain one long-lived global task document at `docs/tasks/TASKS.md` once the first task is explicitly accepted for tracking.
- Do not add a task to the global task document automatically. First discuss it in chat, define the task briefly, and get clear user confirmation that it should enter task tracking.
- The global task document must keep current open tasks and the two most recent closed tasks for context.
- Task order in the global task document must be newest first within priority groups: open tasks before closed tasks, then newer tasks before older tasks.
- Each task entry must include task name, short description, status, start date, last updated date, current completion state, and link to its detail document when one exists.
- A task can only be closed after the agent asks whether it can be closed and the user gives an affirmative reply.
- Do not infer task closure from code completion, passing checks, or silence. Human confirmation is required.
- At the start of any work session that appears task-related, read `docs/tasks/TASKS.md` if it exists, summarize open tasks, and ask which task is in scope unless the user has already made the target task clear.
- When a task is open and its last updated date is older than the current work date, update the date as part of the session startup maintenance.

## Task Detail Documents
- Every open tracked task must have one active detail document under `docs/tasks/active/`.
- Task detail document names must use `任务名介绍文档【YYYY-MM-DD】.md`.
- The detail document must describe the task goal, scope, current state, decisions, blockers, next steps, and file navigation.
- File navigation must point to relevant source files, configs, data locations, outputs, and related docs; update it whenever those files change.
- Keep task detail documents focused on task continuity, not broad narrative summaries.
- When a task is closed, move its detail document to `docs/tasks/closed/`; that directory is only for closed task detail documents.
- Do not delete closed task detail documents unless the user explicitly asks.
- If a task changes name or scope, update the global task document and detail document together so links and navigation stay valid.

## Git Discipline
- Check `git status --short --branch` before and after meaningful changes.
- Commit at the end of each coherent unit of work when the workspace is in a verified state.
- Use small, descriptive commits that make rollback easy.
- Before risky edits, create a checkpoint commit if there are valuable uncommitted changes.
- Never rewrite, delete, or revert user changes unless the user explicitly asks.
- Do not commit ignored/generated artifacts, local secrets, large videos, model weights, or cache directories.
- Prefer commit messages like `init agent guide`, `add video extraction pipeline`, or `fix sumo route export`.

## Expected Structure
- `src/`: project source code.
- `data/raw/`: original videos or input data, ignored by Git.
- `data/interim/`: extracted frames, tracks, detections, and temporary files, ignored by Git.
- `data/processed/`: cleaned datasets or derived inputs, ignored by Git unless promoted intentionally.
- `outputs/`: generated SUMO files, reports, and visualizations, ignored by Git unless promoted intentionally.
- `docs/`: durable notes, assumptions, and workflow documentation.
- `docs/tasks/`: tracked task index plus active and closed task detail documents.
- `.scratch/`: local throwaway experiments, ignored by Git.
- `tmp/`: temporary files created by tools or scripts, ignored by Git.

## Verification
- Prefer command-line checks that can be rerun by another agent.
- When adding processing code, include a small sample or fixture when practical, not large real videos.
- Report tests or checks that were run. If checks were skipped, explain why.
- For data pipelines, verify at least one small end-to-end path before declaring the task complete.
