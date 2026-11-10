"""Tests for the dedupe-against-existing-feedback session — SDK calls mocked."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from answerable.review import (
    _DEDUPE_ITEM_CHARS,
    DuplicateFinding,
    PRReviewer,
)
from answerable.sdk import QueryResult, QueryUsage

DIFF = """\
diff --git a/app/a.py b/app/a.py
--- a/app/a.py
+++ b/app/a.py
@@ -1,2 +1,4 @@
 x = 1
+y = x / 0
+lock = None
 z = 2
"""

FINDINGS = [
    {"file": "app/a.py", "line": 2, "severity": "bug",
     "title": "Division by zero", "description": "y divides by zero",
     "fix_plan": "guard"},
    {"file": "app/a.py", "line": 3, "severity": "bug",
     "title": "Lock never acquired", "description": "lock is None",
     "fix_plan": "init lock"},
]


def _thread(
    body: str, *, path: str = "app/a.py", line: int | None = 2,
    author: str | None = "alice", resolved: bool = False,
    outdated: bool = False, replies: tuple[str, ...] = (),
) -> dict[str, Any]:
    root: dict[str, Any] = {"databaseId": 1, "path": path, "body": body, "line": line}
    if author is not None:
        root["author"] = {"login": author}
    return {
        "id": "T", "isResolved": resolved, "isOutdated": outdated,
        "comments": {"nodes": [root, *({"body": r} for r in replies)]},
    }


def _host(
    threads: list[dict[str, Any]] | None = None,
    reviews: list[str] | None = None,
    comments: list[str] | None = None,
) -> MagicMock:
    host = MagicMock()
    host.get_pr_diff.return_value = DIFF
    host.get_pr_info.return_value = {"title": "Add y", "body": ""}
    host.get_pr_review_threads.return_value = threads or []
    host.get_pr_review_bodies.return_value = reviews or []
    host.get_pr_comment_bodies.return_value = comments or []
    return host


def _result(data: object, tokens: int = 1000, cost: float | None = 0.05) -> QueryResult:
    return QueryResult(
        response=json.dumps(data),
        usage=QueryUsage(total_tokens=tokens, total_cost_usd=cost),
    )


def _review() -> QueryResult:
    return _result({"summary": "s", "findings": FINDINGS})


async def _run(
    host: MagicMock, *responses: object, dedupe: bool = True, verify: bool = False,
) -> tuple[Any, AsyncMock]:
    reviewer = PRReviewer(
        "owner/repo", code_host=host, verify_findings=verify, dedupe_findings=dedupe,
    )
    with patch("answerable.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = list(responses)
        result = await reviewer.review(1)
    return result, mock_query


HUMAN_THREAD = _thread("Careful: x / 0 will raise here.", author="alice")


@pytest.mark.asyncio
async def test_duplicate_of_human_thread_is_dropped() -> None:
    dedupe = _result({"duplicates": [
        {"id": 0, "existing": "T0", "reason": "alice already flagged the zero division"},
    ]})
    result, mock_query = await _run(_host(threads=[HUMAN_THREAD]), _review(), dedupe)

    assert mock_query.await_count == 2
    assert [f.title for f in result.findings] == ["Lock never acquired"]
    assert len(result.duplicates) == 1
    dup = result.duplicates[0]
    assert isinstance(dup, DuplicateFinding)
    assert dup.finding.title == "Division by zero"
    assert dup.existing == "T0 (review thread on app/a.py:2 by @alice (open))"
    assert dup.reason == "alice already flagged the zero division"


@pytest.mark.asyncio
async def test_dedupe_is_a_separate_session_seeing_all_feedback_kinds() -> None:
    host = _host(
        threads=[
            HUMAN_THREAD,
            _thread("old note", author=None, line=None, resolved=True, outdated=True,
                    replies=("fixed in abc123", "")),
        ],
        reviews=["", "Other-bot summary: the lock is never initialised."],
        comments=["LGTM once the lock is fixed"],
    )
    _, mock_query = await _run(host, _review(), _result({"duplicates": []}))

    review_call, dedupe_call = mock_query.await_args_list
    prompt = dedupe_call.kwargs["prompt"]
    assert prompt != review_call.kwargs["prompt"]
    assert "deduplicating an automated pull-request review" in prompt
    # Both candidate findings, as claims.
    assert "### Finding 0" in prompt and "### Finding 1" in prompt
    assert "Lock never acquired" in prompt
    # Every kind of existing feedback, with ids, authors, locations, state.
    assert "### T0 — review thread on app/a.py:2 by @alice (open)" in prompt
    assert "### T1 — review thread on app/a.py (resolved, outdated)" in prompt
    assert "Reply: fixed in abc123" in prompt
    assert "### R1 — review summary\nOther-bot summary" in prompt
    assert "### R0" not in prompt  # empty review body skipped
    assert "### C0 — conversation comment\nLGTM once the lock is fixed" in prompt

    options = dedupe_call.kwargs["options"]
    assert set(options.allowed_tools) == {"Read", "Glob", "Grep"}
    for attr in ("resume", "continue_conversation", "session_id", "fork_session"):
        assert not hasattr(options, attr)


@pytest.mark.asyncio
async def test_thread_matched_findings_bypass_dedupe() -> None:
    """A finding the title matcher ties to a thread stays on the deterministic
    skip / defect-persists / dispute path in post_review, so it is neither sent
    to the dedupe session nor dropped by it."""
    own_thread = _thread("\U0001f41b **Division by zero**\n\ny divides by zero")
    dedupe = _result({"duplicates": [
        {"id": 0, "existing": "T0", "reason": "same thread"},
    ]})
    result, mock_query = await _run(_host(threads=[own_thread]), _review(), dedupe)

    prompt = mock_query.await_args_list[1].kwargs["prompt"]
    assert "Division by zero" not in prompt.split("## Existing feedback")[0]
    assert "### Finding 0" in prompt and "### Finding 1" not in prompt
    # id 0 in the dedupe call is the *second* finding (the only candidate).
    assert [f.title for f in result.findings] == ["Division by zero"]
    assert [d.finding.title for d in result.duplicates] == ["Lock never acquired"]


@pytest.mark.asyncio
async def test_all_findings_thread_matched_skips_session() -> None:
    threads = [
        _thread("\U0001f41b **Division by zero**"),
        _thread("\U0001f41b **Lock never acquired**", line=3),
    ]
    result, mock_query = await _run(_host(threads=threads), _review())

    assert mock_query.await_count == 1
    assert len(result.findings) == 2


@pytest.mark.asyncio
async def test_no_existing_feedback_skips_session() -> None:
    result, mock_query = await _run(_host(reviews=["  "], comments=[""]), _review())

    assert mock_query.await_count == 1
    assert len(result.findings) == 2
    assert result.duplicates == []


@pytest.mark.asyncio
async def test_unknown_or_malformed_entries_keep_findings() -> None:
    dedupe = _result({"duplicates": [
        {"id": 0, "existing": "T9", "reason": "no such item"},
        {"id": 1, "existing": "", "reason": "missing"},
        {"id": "x", "existing": "T0"},
        {"id": True, "existing": "T0"},
        {"id": 7, "existing": "T0"},  # out of range
        "junk",
    ]})
    result, _ = await _run(_host(threads=[HUMAN_THREAD]), _review(), dedupe)

    assert len(result.findings) == 2
    assert result.duplicates == []


@pytest.mark.asyncio
async def test_session_failure_keeps_findings(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="answerable"):
        result, mock_query = await _run(
            _host(threads=[HUMAN_THREAD]), _review(), RuntimeError("SDK down"),
        )

    assert mock_query.await_count == 2
    assert len(result.findings) == 2
    assert result.tokens_used == 1000
    assert "Dedupe session failed" in caplog.text


@pytest.mark.asyncio
async def test_unparseable_response_keeps_findings() -> None:
    garbage = QueryResult(response="no idea", usage=QueryUsage(total_tokens=3))
    result, _ = await _run(_host(threads=[HUMAN_THREAD]), _review(), garbage)

    assert len(result.findings) == 2
    assert result.tokens_used == 1003


@pytest.mark.asyncio
async def test_feedback_fetch_failure_keeps_findings() -> None:
    host = _host()
    host.get_pr_review_threads.side_effect = RuntimeError("gh down")
    result, mock_query = await _run(host, _review())

    assert mock_query.await_count == 1
    assert len(result.findings) == 2


@pytest.mark.asyncio
async def test_dedupe_disabled_fetches_nothing() -> None:
    host = _host(threads=[HUMAN_THREAD])
    result, mock_query = await _run(host, _review(), dedupe=False)

    assert mock_query.await_count == 1
    host.get_pr_review_threads.assert_not_called()
    assert len(result.findings) == 2


@pytest.mark.asyncio
async def test_usage_summed() -> None:
    result, _ = await _run(
        _host(threads=[HUMAN_THREAD]), _review(),
        _result({"duplicates": []}, tokens=200, cost=0.02),
    )
    assert result.tokens_used == 1200
    assert result.cost_usd == pytest.approx(0.07)


@pytest.mark.asyncio
async def test_runs_after_verification_on_its_output() -> None:
    """Verify first, then dedupe what survived (with revisions applied)."""
    verify = _result({"verdicts": [
        {"id": 0, "verdict": "refuted", "reason": "x is never 0"},
        # Downgraded to a suggestion, which a review-summary match may drop
        # (a bug matching one would still be posted).
        {"id": 1, "verdict": "revised",
         "revised": {"title": "Lock is None", "severity": "suggestion"}},
    ]})
    dedupe = _result({"duplicates": [
        {"id": 0, "existing": "R0", "reason": "other bot said so"},
    ]})
    host = _host(reviews=["lock is never initialised"])
    result, mock_query = await _run(host, _review(), verify, dedupe, verify=True)

    assert mock_query.await_count == 3
    prompt = mock_query.await_args_list[2].kwargs["prompt"]
    assert "Lock is None" in prompt and "Division by zero" not in prompt
    assert result.findings == []
    assert [r.finding.title for r in result.refuted] == ["Division by zero"]
    assert [d.finding.title for d in result.duplicates] == ["Lock is None"]
    assert result.duplicates[0].existing == "R0 (review summary)"


def test_long_items_are_clipped_and_budget_enforced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    host = _host(comments=["x" * (_DEDUPE_ITEM_CHARS * 3)] * 50)
    reviewer = PRReviewer("owner/repo", code_host=host)
    items = reviewer._collect_existing_feedback(1, [])

    assert len(items) == 50
    assert all(len(text) < _DEDUPE_ITEM_CHARS + 100 for _, _, text in items)
    assert "clipped" in items[0][2]

    with caplog.at_level(logging.WARNING, logger="answerable"):
        rendered = PRReviewer._format_existing_feedback(items)
    assert "more existing item(s) omitted due to size limits" in rendered
    assert "### C49" not in rendered
    assert "Dedupe context budget reached" in caplog.text


def test_parse_review_threads_reads_line_and_author() -> None:
    nodes = [
        _thread("a", line=12, author="bob"),
        {"id": "T", "comments": {"nodes": [
            {"databaseId": 2, "path": "b.py", "body": "b", "line": None,
             "originalLine": 7, "author": None},
        ]}},
    ]
    first, second = PRReviewer._parse_review_threads(nodes)
    assert (first.line, first.author) == (12, "bob")
    assert (second.line, second.author) == (7, "")
