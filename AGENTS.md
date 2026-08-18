# Agent Execution Policy

## Prime Directive

Do not proceed with work when there are unanswered questions.

If you have questions, stop as soon as you have them and ask before proceeding. Do not guess, do not fill in missing requirements silently, and do not continue implementation while a decision is unclear.

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

- **Files modified:**  list
- **Ruff:** pass/fail + any warnings
- **verify.py:** pass/fail + output
