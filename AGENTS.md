# Agent Execution Policy

## File Operations

- Write all temporary/patch/debug files to `.tmp/` only
- Before completing any task: check `.tmp/` for files created this session and delete them

## Code Verification (required after changing code)

Run in order:

1. `ruff check . --fix`
2. `ruff format .`
3. `python tools/verify.py`

Task is not complete until `verify.py` passes.

### Ruff Error B905 (zip() missing strict=True)

Do NOT auto-apply `strict=True`. First confirm the zipped iterables are guaranteed equal length, then add it.

## Completion Report

Reply with a structured block:

- **Files modified:** <list>
- **Ruff:** <pass/fail + any warnings>
- **verify.py:** <pass/fail + output>
