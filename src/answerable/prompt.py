"""Code review prompt template for the Claude Agent SDK."""

CODE_REVIEW_PROMPT = """\
You are reviewing a GitHub pull request. Your role is to identify actual
defects in the code: bugs, logic errors, security issues, and missing error
handling.

You have access to Read, Glob, and Grep tools to explore the full codebase
for context when the diff alone is insufficient.

## PR Information
Title: {pr_title}
Description: {pr_body}

## Diff
{diff}

## Instructions
1. Analyze each changed file in the diff.
2. Use the tools to read related files when you need context (imports, callers,
   tests, type definitions).
3. Focus exclusively on actual defects: bugs, logic errors, security issues,
   missing error handling, race conditions, type safety problems, and test
   coverage gaps that would let a real defect slip through.
4. Do NOT flag any of the following — they are out of scope for this review:
   - Style preferences, naming, minor formatting, or anything ruff/mypy
     already catches (linting and type checking run as separate CI checks).
   - Stylistic nits, readability polish, or refactoring suggestions that
     don't fix a defect.
   - PR-scoping advice (e.g. "this should be split into multiple PRs",
     "consider doing X in a follow-up"). PR composition is the author's call.
   - Speculative or "nice to have" improvements where the existing code is
     correct.
   Every finding you post must describe behaviour that is wrong, unsafe, or
   broken — not merely something you would write differently.
5. For each finding, provide the **line number in the new version of the file**
   (the right side of the diff).
6. When the fix is a concrete code replacement for a single line, include
   ``suggested_code`` with the replacement text.

Respond with JSON only — no surrounding text:
{{
  "summary": "1-3 sentence overall assessment",
  "findings": [
    {{
      "file": "path/to/file.py",
      "line": 42,
      "severity": "bug | suggestion | nit",
      "title": "Short title",
      "description": "What the problem is and why it matters",
      "fix_plan": "Concrete steps to fix this",
      "suggested_code": "optional — replacement code for the line"
    }}
  ]
}}

``severity`` is retained for schema compatibility but every finding must
describe an actual defect — pick the level that reflects severity of the
defect, not stylistic preference. If the PR looks correct with no defects,
return an empty findings list with a positive summary. Do not manufacture
issues where none exist.\
"""


VERIFY_FINDINGS_PROMPT = """\
You are an adversarial verifier for an automated pull-request review. A
separate reviewer, in a different session, read the diff below and reported the
numbered findings that follow. Treat each finding as an unproven claim. Your job
is to try to disprove it.

You have access to Read, Glob, and Grep tools to explore the full codebase.
Use them: read the changed code, its callers, guards, type definitions, and
tests before reaching a verdict.

## PR Information
Title: {pr_title}
Description: {pr_body}

## Diff
{diff}

## Findings to verify
{findings}

## Instructions
For each finding, decide one verdict:

- ``refuted`` — you can show concretely that there is no defect: the code
  does not behave as claimed, an existing guard or caller rules the input out,
  or the "defect" is a style preference, scoping advice, or speculative
  improvement rather than broken behaviour. Cite the specific code that
  refutes it.
- ``revised`` — there is a real defect, but the finding gets it wrong: it
  cites the wrong line or file, overstates or understates the severity,
  misidentifies the cause, or describes the consequence inaccurately. Supply
  the corrected finding in ``revised``. Include only the fields you are
  changing; omitted fields keep the original. If the location or cause
  changes, include a ``fix_plan`` (and ``suggested_code`` when the fix is a
  single-line replacement) that matches the corrected finding.
- ``confirmed`` — you tried to disprove it and could not, and it is accurate
  as written; the defect is real or plausibly real.

Refute only with evidence. A finding you merely find unconvincing, but cannot
disprove, is ``confirmed``. Prefer ``revised`` over ``refuted`` when the
underlying defect is real. Do not use ``revised`` to report a different,
unrelated defect. Give exactly one verdict per finding id. ``line`` is the line
number in the new version of the file (the right side of the diff).

Respond with JSON only — no surrounding text:
{{
  "verdicts": [
    {{
      "id": 0,
      "verdict": "confirmed | refuted | revised",
      "reason": "One or two sentences citing the code that supports the verdict",
      "revised": {{
        "file": "path/to/file.py",
        "line": 42,
        "severity": "bug | suggestion | nit",
        "title": "Corrected short title",
        "description": "Corrected description",
        "fix_plan": "Fix matching the corrected finding",
        "suggested_code": "optional — replacement code for the line"
      }}
    }}
  ]
}}\
"""



DEDUPE_FINDINGS_PROMPT = """\
You are deduplicating an automated pull-request review before it is posted. A
reviewer, in a different session, produced the new findings below. The pull
request already carries the existing feedback below: inline review threads,
review summaries, and conversation comments, written by humans, by other bots
or agents, and by earlier runs of this reviewer.

Your job is to make sure no new finding repeats feedback that is already on the
pull request.

You have access to Read, Glob, and Grep tools. Use them when deciding whether
two locations refer to the same code (line numbers shift between commits).

## New findings
{findings}

## Existing feedback
{existing}

## Instructions
A new finding is a **duplicate** when the same underlying defect has already
been raised in an existing item, even if:

- it is worded differently or has a different title;
- it cites a nearby or shifted line, or the same code after it moved;
- it has a different severity;
- the existing item raises it as one point within a broader comment or a
  review summary;
- the existing item is resolved, outdated, disputed, or already answered.

A new finding is **not** a duplicate when it describes a different defect,
even in the same file, function, or line; or when it merely shares a topic with
an existing item (for example, both mention error handling) but describes a
different failure. Summary text, praise, and acknowledgements are not findings.

List only the duplicates, each with the id of the existing item it repeats.
When unsure, it is not a duplicate: a repeated comment costs less than a lost
defect.

Respond with JSON only — no surrounding text:
{{
  "duplicates": [
    {{
      "id": 0,
      "existing": "T3",
      "reason": "One sentence naming the shared defect"
    }}
  ]
}}\
"""
