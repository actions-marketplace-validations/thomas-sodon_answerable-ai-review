#!/usr/bin/env python3
"""CLI entry point for AI code review of pull requests.

Usage:
    answerable --repo owner/repo --pr 42 [--dry-run]
    answerable --repo owner/repo --pr 42 --require-comment-replies
    answerable --repo owner/repo --pr 42 --skip-verification
    answerable --repo owner/repo --pr 42 --skip-dedupe

Findings are adversarially verified before posting: a second agent session,
fresh and sharing none of the reviewer's context, tries to disprove each one,
and findings it refutes are dropped. ``--skip-verification`` posts the
reviewer's findings unverified.

A further fresh session then compares the surviving findings with feedback
already on the PR — review threads, review summaries, and conversation comments
from humans, other agents, or earlier runs — and drops any that repeat it.
``--skip-dedupe`` turns that pass off.

Exits 1 whenever the reviewer produces a comment — i.e. a finding of any
severity (bug, suggestion, or nit) is newly posted, or a bug is flagged as
persisting on a resolved thread — so the PR is blocked until those comments
are addressed. Findings already tracked by open issues, escalated to issues,
or matching an existing review thread are not freshly produced and do not
affect the exit code. Infrastructure failures (API errors, SDK crashes) also
exit 1 so CI surfaces broken review runs.

With ``--require-comment-replies`` the diff review and posting are skipped
entirely: the run becomes a read-only audit that makes no model call and exits
1 if any of the reviewer's own findings (inline thread or overflow) still lack
a reply. A gh/GraphQL outage during the audit fails the gate closed (exit 1)
rather than being mistaken for "all comments replied" — the audit probes
connectivity first.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import tomllib
from pathlib import Path

from answerable.review import PRReviewer, _extract_thread_title

# ---------------------------------------------------------------------------
# Config defaults — optionally loaded from a TOML [review] section and/or
# ANSWERABLE_* environment variables. CLI arguments always win.
# ---------------------------------------------------------------------------

_DEFAULT_MAX_DIFF_CHARS = 80_000
_DEFAULT_MAX_FILES = 50
_DEFAULT_GH_TIMEOUT = 30
# Reply-coverage gate default (0 = off). Stored as 0/1 alongside the int
# options; coerced to bool at the argparse default below.
_DEFAULT_REQUIRE_COMMENT_REPLIES = 0
# Adversarial verification default (1 = on). Stored as 0/1 like the gate.
_DEFAULT_VERIFY_FINDINGS = 1
# Dedupe-against-existing-feedback default (1 = on).
_DEFAULT_DEDUPE_FINDINGS = 1

_CONFIG_KEYS: dict[str, int] = {
    "max_diff_chars": _DEFAULT_MAX_DIFF_CHARS,
    "max_files": _DEFAULT_MAX_FILES,
    "gh_timeout": _DEFAULT_GH_TIMEOUT,
    "require_comment_replies": _DEFAULT_REQUIRE_COMMENT_REPLIES,
    "verify_findings": _DEFAULT_VERIFY_FINDINGS,
    "dedupe_findings": _DEFAULT_DEDUPE_FINDINGS,
}

# Environment variable naming an explicit config file (searched first).
CONFIG_ENV_VAR = "ANSWERABLE_CONFIG"
# Per-key environment overrides: ANSWERABLE_MAX_DIFF_CHARS, etc.
_ENV_PREFIX = "ANSWERABLE_"


def _config_candidates() -> list[Path]:
    """Config files searched in order; the first with a non-empty [review] wins."""
    candidates: list[Path] = []
    explicit = os.environ.get(CONFIG_ENV_VAR)
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend((
        Path.cwd() / "answerable.toml",
    ))
    return candidates


def _load_toml_defaults() -> dict[str, int]:
    """Load the [review] section from the first config file that has one."""
    for candidate in _config_candidates():
        if candidate.is_file():
            try:
                data = tomllib.loads(candidate.read_text(encoding="utf-8"))
                section = data.get("review", {})
                if isinstance(section, dict) and section:
                    result: dict[str, int] = {}
                    for key, default in _CONFIG_KEYS.items():
                        with contextlib.suppress(TypeError, ValueError):
                            result[key] = int(section.get(key, default))
                    return result
            except Exception:
                pass
    return {}


def _load_env_overrides() -> dict[str, int]:
    """Read ANSWERABLE_<KEY> integer overrides from the environment."""
    result: dict[str, int] = {}
    for key in _CONFIG_KEYS:
        raw = os.environ.get(_ENV_PREFIX + key.upper())
        if raw is None or not raw.strip():
            continue
        with contextlib.suppress(ValueError):
            result[key] = int(raw.strip())
    return result


def _load_review_defaults() -> dict[str, int]:
    """Merge config-file defaults with environment overrides (env wins)."""
    result = _load_toml_defaults()
    result.update(_load_env_overrides())
    return result


def main() -> None:
    toml_defaults = _load_review_defaults()

    parser = argparse.ArgumentParser(description="AI code review for PRs")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")
    parser.add_argument("--pr", required=True, type=int, help="PR number")
    parser.add_argument(
        "--max-diff-chars", type=int,
        default=toml_defaults.get("max_diff_chars", _DEFAULT_MAX_DIFF_CHARS),
    )
    parser.add_argument(
        "--max-files", type=int,
        default=toml_defaults.get("max_files", _DEFAULT_MAX_FILES),
    )
    parser.add_argument(
        "--gh-timeout", type=int,
        default=toml_defaults.get("gh_timeout", _DEFAULT_GH_TIMEOUT),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print review to stdout instead of posting",
    )
    parser.add_argument(
        "--require-comment-replies",
        action="store_true",
        default=bool(toml_defaults.get(
            "require_comment_replies", _DEFAULT_REQUIRE_COMMENT_REPLIES,
        )),
        help=(
            "Check-only: skip the diff review and exit 1 if any of the "
            "reviewer's own findings (inline or overflow) lack a reply"
        ),
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        default=not toml_defaults.get(
            "verify_findings", _DEFAULT_VERIFY_FINDINGS,
        ),
        help=(
            "Post the reviewer's findings without the adversarial "
            "verification pass that drops findings a fresh session refutes"
        ),
    )
    parser.add_argument(
        "--skip-dedupe",
        action="store_true",
        default=not toml_defaults.get(
            "dedupe_findings", _DEFAULT_DEDUPE_FINDINGS,
        ),
        help=(
            "Skip the session that drops findings repeating feedback already "
            "on the PR (from humans, other agents, or earlier reviews)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("answerable.cli")

    reviewer = PRReviewer(
        github_repo=args.repo,
        max_diff_chars=args.max_diff_chars,
        max_files=args.max_files,
        gh_timeout=args.gh_timeout,
        verify_findings=not args.skip_verification,
        dedupe_findings=not args.skip_dedupe,
    )

    if args.require_comment_replies:
        # Check-only mode: skip the diff review and posting entirely. This is a
        # read-only audit of reply state — it makes no model call and posts
        # nothing, so reviewer.review / asyncio.run never run.
        try:
            report = reviewer.find_unreplied_comments(args.pr)
        except Exception:
            log.exception("Reply-coverage check failed")
            sys.exit(1)  # infrastructure failure — surface in CI

        for thread in report.inline:
            title = _extract_thread_title(thread.body or "") or "(untitled)"
            log.error("Unreplied inline comment: %s — %s", thread.path, title)
        for finding in report.overflow:
            log.error(
                "Unreplied overflow comment: %s %s:%d",
                finding.fingerprint, finding.file, finding.line,
            )

        unreplied = len(report.inline) + len(report.overflow)
        if unreplied:
            log.error(
                "Failing review: %d unreplied comment(s) on PR #%d",
                unreplied, args.pr,
            )
            sys.exit(1)
        log.info("No unreplied comments on PR #%d", args.pr)
        sys.exit(0)

    try:
        result = asyncio.run(reviewer.review(args.pr))
    except Exception:
        log.exception("Review failed")
        sys.exit(1)

    if result.refuted or result.revised:
        log.info(
            "Adversarial verification dropped %d and revised %d of %d finding(s)",
            len(result.refuted), len(result.revised),
            len(result.refuted) + len(result.findings) + len(result.duplicates),
        )
    if result.duplicates:
        log.info(
            "Dedupe dropped %d finding(s) already raised on the PR",
            len(result.duplicates),
        )

    if args.dry_run:
        # Apply the same filtering as the real run so the exit code matches.
        # _filter_tracked_findings removes findings already tracked by open
        # GitHub issues; _categorize_by_thread_state further skips findings
        # with existing unresolved review threads.  Any finding that survives
        # both filters (i.e. would actually be posted or cause a reopen)
        # triggers exit 1, regardless of severity.
        try:
            filtered_findings = reviewer._filter_tracked_findings(result.findings)
            if filtered_findings:
                new_findings, _skip, reopen = reviewer._categorize_by_thread_state(
                    args.pr, filtered_findings,
                )
                actioned = new_findings + [f for f, _ in reopen]
            else:
                actioned = []
        except Exception:
            log.exception("Dry-run filtering failed; falling back to raw findings")
            actioned = list(result.findings)
        has_findings = bool(actioned)

        print(f"Summary: {result.summary}")
        print(f"Findings: {len(result.findings)}")
        for f in result.findings:
            print(f"  [{f.severity}] {f.file}:{f.line} — {f.title}")
        if result.refuted:
            print(f"Refuted by verification: {len(result.refuted)}")
            for r in result.refuted:
                f = r.finding
                print(f"  [{f.severity}] {f.file}:{f.line} — {f.title}: {r.reason}")
        if result.revised:
            print(f"Revised by verification: {len(result.revised)}")
            for rv in result.revised:
                o, n = rv.original, rv.revised
                print(
                    f"  [{o.severity}] {o.file}:{o.line} — {o.title} → "
                    f"[{n.severity}] {n.file}:{n.line} — {n.title}: {rv.reason}"
                )
        if result.duplicates:
            print(f"Already raised on the PR: {len(result.duplicates)}")
            for d in result.duplicates:
                f = d.finding
                print(
                    f"  [{f.severity}] {f.file}:{f.line} — {f.title} "
                    f"(repeats {d.existing}): {d.reason}"
                )
        sys.exit(1 if has_findings else 0)

    try:
        post_result = reviewer.post_review(
            args.pr, result.findings, result.summary,
            diff=result.diff or None,
        )
    except Exception:
        log.exception("Failed to post review comment")
        # Print to CI log as fallback
        print(f"Summary: {result.summary}")
        for f in result.findings:
            print(f"  [{f.severity}] {f.file}:{f.line} — {f.title}: {f.description}")
        sys.exit(1)  # infrastructure failure — surface in CI

    actioned = post_result.posted + post_result.reopened

    if post_result.disputed:
        log.info(
            "Escalated %d disputed finding(s) to GitHub issues on PR #%d",
            len(post_result.disputed), args.pr,
        )

    if post_result.filtered:
        log.info(
            "Skipped %d finding(s) already tracked by open issues",
            len(post_result.filtered),
        )

    if post_result.skipped:
        log.info(
            "Skipped %d finding(s) with existing review threads",
            len(post_result.skipped),
        )

    # Fail whenever the reviewer produced a comment: any finding of any
    # severity that was newly posted, or a bug flagged as persisting on a
    # resolved thread, must be addressed before the PR proceeds. Findings
    # already tracked by open issues (filtered), escalated to issues
    # (disputed), or matching an existing review thread (skipped) are not
    # freshly produced, so they do not block.
    if actioned:
        log.error(
            "Failing review: %d comment(s) posted on PR #%d",
            len(actioned), args.pr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
