"""Resolved threads whose defect persists get a reply and are unresolved."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from answerable import cli
from answerable.github import GitHubClient, GitHubError
from answerable.markers import _DEFECT_PERSISTS_MARKER
from answerable.review import (
    PersistingFinding,
    PRReviewer,
    ReviewFinding,
    ReviewResult,
    ReviewThread,
)
from answerable.sdk import QueryResult, QueryUsage

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _node(
    thread_id: str, body: str, *, resolved: bool = True, outdated: bool = False,
    path: str = "app/a.py", comment_id: int = 100, author: str = "alice",
    replies: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = {"databaseId": comment_id, "path": path, "body": body, "line": 2,
            "author": {"login": author}}
    return {
        "id": thread_id, "isResolved": resolved, "isOutdated": outdated,
        "comments": {"nodes": [root, *({"body": r} for r in replies)]},
    }


def _bug(title: str = "Division by zero", severity: str = "bug") -> ReviewFinding:
    return ReviewFinding(
        file="app/a.py", line=2, severity=severity, title=title,
        description="y divides by zero", fix_plan="guard x",
    )


def _host(nodes: list[dict[str, Any]], issues: list[dict[str, Any]] | None = None) -> MagicMock:
    host = MagicMock()
    host.get_pr_review_threads.return_value = nodes
    host.list_issues.return_value = issues or []
    host.get_pr_review_bodies.return_value = []
    host.get_pr_comment_bodies.return_value = []
    host.unresolve_review_thread.return_value = True
    return host


def _persisting(node: dict[str, Any], finding: ReviewFinding | None = None) -> PersistingFinding:
    (thread,) = PRReviewer._parse_review_threads([node])
    return PersistingFinding(
        finding=finding or _bug(), thread=thread,
        existing="T0 (review thread)", reason="same zero division",
    )


HUMAN = "Careful: x / 0 raises here."


# ------------------------------------------------------------------
# GitHubClient.unresolve_review_thread
# ------------------------------------------------------------------


class TestUnresolveReviewThread:
    def _client_run(self, stdout: str) -> tuple[GitHubClient, MagicMock]:
        client = GitHubClient(repo="owner/repo")
        result = MagicMock()
        result.stdout = stdout
        return client, MagicMock(return_value=result)

    def test_sends_unresolve_mutation(self) -> None:
        client, run = self._client_run(json.dumps(
            {"data": {"unresolveReviewThread": {"thread": {"id": "PRRT_1", "isResolved": False}}}},
        ))
        with patch.object(client, "_run", run):
            assert client.unresolve_review_thread("PRRT_1") is True
        cmd = run.call_args[0][0]
        assert cmd[:3] == ["gh", "api", "graphql"]
        assert any("unresolveReviewThread" in a for a in cmd)
        assert "threadId=PRRT_1" in cmd

    def test_graphql_errors_return_false(self, caplog: pytest.LogCaptureFixture) -> None:
        client, run = self._client_run(json.dumps({"data": None, "errors": [{"message": "nope"}]}))
        with patch.object(client, "_run", run), caplog.at_level(logging.WARNING):
            assert client.unresolve_review_thread("PRRT_1") is False
        assert "refused to unresolve" in caplog.text

    def test_gh_failure_returns_false(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run", side_effect=GitHubError("boom")):
            assert client.unresolve_review_thread("PRRT_1") is False

    def test_empty_thread_id_is_a_no_op(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as run:
            assert client.unresolve_review_thread("") is False
        run.assert_not_called()


# ------------------------------------------------------------------
# Dedupe routing
# ------------------------------------------------------------------


async def _dedupe(
    nodes: list[dict[str, Any]], findings: list[ReviewFinding], duplicates: list[dict[str, Any]],
) -> ReviewResult:
    reviewer = PRReviewer(
        "owner/repo", code_host=_host(nodes), verify_findings=False,
    )
    result = ReviewResult(findings=list(findings))
    response = QueryResult(
        response=json.dumps({"duplicates": duplicates}), usage=QueryUsage(),
    )
    with patch("answerable.review.query", new_callable=AsyncMock, return_value=response):
        await reviewer._dedupe(1, result)
    return result


@pytest.mark.asyncio
async def test_bug_repeating_resolved_human_thread_is_persisting() -> None:
    result = await _dedupe(
        [_node("PRRT_h", HUMAN)], [_bug()],
        [{"id": 0, "existing": "T0", "reason": "same zero division"}],
    )
    assert result.findings == []
    assert result.duplicates == []
    (pf,) = result.persisting
    assert pf.thread.thread_id == "PRRT_h"
    assert pf.existing == "T0 (review thread on app/a.py:2 by @alice (resolved))"
    assert pf.reason == "same zero division"


@pytest.mark.asyncio
@pytest.mark.parametrize("node", [
    _node("PRRT_open", HUMAN, resolved=False),  # still open: just a repeat
    _node("PRRT_issue", HUMAN, replies=("Tracked in #12",)),  # issue tracks it
])
async def test_open_or_issue_linked_threads_are_plain_duplicates(node: dict[str, Any]) -> None:
    result = await _dedupe(
        [node], [_bug()], [{"id": 0, "existing": "T0", "reason": "r"}],
    )
    assert result.persisting == []
    assert len(result.duplicates) == 1


@pytest.mark.asyncio
async def test_non_bug_repeating_resolved_thread_is_a_duplicate() -> None:
    result = await _dedupe(
        [_node("PRRT_h", HUMAN)], [_bug(severity="suggestion")],
        [{"id": 0, "existing": "T0", "reason": "r"}],
    )
    assert result.persisting == []
    assert len(result.duplicates) == 1


@pytest.mark.asyncio
async def test_one_persisting_finding_per_thread() -> None:
    result = await _dedupe(
        [_node("PRRT_h", HUMAN)], [_bug("Zero division"), _bug("Divides by zero")],
        [{"id": 0, "existing": "T0"}, {"id": 1, "existing": "T0"}],
    )
    assert [p.finding.title for p in result.persisting] == ["Zero division"]
    assert [d.finding.title for d in result.duplicates] == ["Divides by zero"]


async def _dedupe_against_bodies(
    finding: ReviewFinding, existing_id: str,
) -> ReviewResult:
    host = _host([])
    host.get_pr_review_bodies.return_value = ["Additional findings: zero division"]
    host.get_pr_comment_bodies.return_value = ["Division by zero — fixed in abc123"]
    reviewer = PRReviewer("owner/repo", code_host=host, verify_findings=False)
    result = ReviewResult(findings=[finding])
    response = QueryResult(
        response=json.dumps({"duplicates": [{"id": 0, "existing": existing_id}]}),
        usage=QueryUsage(),
    )
    with patch("answerable.review.query", new_callable=AsyncMock, return_value=response):
        await reviewer._dedupe(1, result)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_id", ["R0", "C0"])
async def test_bug_matching_summary_or_comment_is_still_posted(existing_id: str) -> None:
    """A review summary or conversation comment can't be reopened and isn't
    gated, so dropping a verified bug that repeats one would let a still-present
    defect pass the run.  It is kept and posted instead."""
    bug = _bug()
    result = await _dedupe_against_bodies(bug, existing_id)

    assert result.findings == [bug]
    assert result.duplicates == []
    assert result.persisting == []


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_id", ["R0", "C0"])
async def test_non_bug_matching_summary_or_comment_is_dropped(existing_id: str) -> None:
    result = await _dedupe_against_bodies(_bug(severity="nit"), existing_id)

    assert result.findings == []
    assert len(result.duplicates) == 1
    assert result.persisting == []


# ------------------------------------------------------------------
# post_review
# ------------------------------------------------------------------


def test_persisting_thread_gets_reply_and_is_unresolved() -> None:
    node = _node("PRRT_h", HUMAN, comment_id=555)
    host = _host([node])
    reviewer = PRReviewer("owner/repo", code_host=host)
    pf = _persisting(node)

    result = reviewer.post_review(1, [], "Summary", diff="", persisting=[pf])

    host.reply_to_review_comment.assert_called_once()
    pr, comment_id, body = host.reply_to_review_comment.call_args[0]
    assert (pr, comment_id) == (1, 555)
    assert body.startswith(_DEFECT_PERSISTS_MARKER)
    assert "**Division by zero**" in body
    host.unresolve_review_thread.assert_called_once_with("PRRT_h")
    host.post_pr_review.assert_not_called()
    assert result.reopened == [pf.finding]
    assert result.posted == []


def test_persisting_alongside_new_findings() -> None:
    node = _node("PRRT_h", HUMAN)
    host = _host([node])
    reviewer = PRReviewer("owner/repo", code_host=host)
    new = ReviewFinding(file="app/b.py", line=1, severity="bug", title="Other bug",
                        description="d", fix_plan="f")

    result = reviewer.post_review(1, [new], "S", diff=None, persisting=[_persisting(node)])

    host.unresolve_review_thread.assert_called_once_with("PRRT_h")
    host.post_pr_review.assert_called_once()
    assert result.posted == [new]
    assert len(result.reopened) == 1


@pytest.mark.parametrize(("fresh_nodes", "reason"), [
    ([], "thread gone"),
    ([_node("PRRT_h", HUMAN, resolved=False)], "already reopened"),
    ([_node("PRRT_h", HUMAN, replies=("See #40",))], "issue-linked now"),
])
def test_persisting_rechecked_against_fresh_threads(
    fresh_nodes: list[dict[str, Any]], reason: str,
) -> None:
    pf = _persisting(_node("PRRT_h", HUMAN))
    host = _host(fresh_nodes)
    reviewer = PRReviewer("owner/repo", code_host=host)

    result = reviewer.post_review(1, [], "S", diff="", persisting=[pf])

    host.reply_to_review_comment.assert_not_called()
    host.unresolve_review_thread.assert_not_called()
    assert result.reopened == []
    assert result.skipped == [pf.finding], reason


def test_persisting_tracked_by_open_issue_is_filtered() -> None:
    node = _node("PRRT_h", HUMAN)
    issue = {"number": 7, "title": "Division by zero in app/a.py",
             "body": "app/a.py divides by zero"}
    host = _host([node], issues=[issue])
    reviewer = PRReviewer("owner/repo", code_host=host)
    pf = _persisting(node)

    result = reviewer.post_review(1, [], "S", diff="", persisting=[pf])

    host.unresolve_review_thread.assert_not_called()
    assert result.filtered == [pf.finding]
    assert result.reopened == []


def test_one_reply_per_thread_across_both_paths() -> None:
    """The title matcher and the dedupe session can both pick the same
    resolved thread; it gets one reply and one unresolve."""
    own = PRReviewer._format_finding_body(_bug())
    node = _node("PRRT_own", own)
    host = _host([node])
    reviewer = PRReviewer("owner/repo", code_host=host)
    other = _bug("Zero divisor unchecked")

    result = reviewer.post_review(
        1, [_bug()], "S", diff=None, persisting=[_persisting(node, other)],
    )

    host.reply_to_review_comment.assert_called_once()
    host.unresolve_review_thread.assert_called_once_with("PRRT_own")
    assert [f.title for f in result.reopened] == ["Division by zero"]
    assert other in result.skipped


def test_unresolve_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    node = _node("PRRT_h", HUMAN)
    host = _host([node])
    host.unresolve_review_thread.return_value = False
    reviewer = PRReviewer("owner/repo", code_host=host)

    with caplog.at_level(logging.WARNING, logger="answerable"):
        result = reviewer.post_review(1, [], "S", diff="", persisting=[_persisting(node)])

    host.reply_to_review_comment.assert_called_once()
    assert len(result.reopened) == 1
    assert "could not unresolve" in caplog.text


# ------------------------------------------------------------------
# Reply-coverage gate
# ------------------------------------------------------------------


def _thread(**kwargs: Any) -> ReviewThread:
    defaults: dict[str, Any] = {
        "thread_id": "T", "comment_id": 1, "path": "a.py", "body": HUMAN,
        "all_bodies": "", "reply_bodies": [],
    }
    defaults.update(kwargs)
    return ReviewThread(**defaults)


PERSIST_REPLY = f"{_DEFECT_PERSISTS_MARKER} in the current PR:\n\n\U0001f41b **X**"


@pytest.mark.parametrize(("thread", "expected"), [
    # Reopened by the reviewer, nobody answered since — on anyone's thread.
    (_thread(reply_bodies=["looks fixed", PERSIST_REPLY]), True),
    (_thread(body="\U0001f41b **X**", reply_bodies=["fixed", PERSIST_REPLY]), True),
    # Someone answered after the reopen.
    (_thread(reply_bodies=[PERSIST_REPLY, "Fixed automatically"]), False),
    # Outdated or re-resolved: exempt.
    (_thread(reply_bodies=[PERSIST_REPLY], is_outdated=True), False),
    (_thread(reply_bodies=[PERSIST_REPLY], is_resolved=True), False),
    # Ordinary human thread with a reply: not the reviewer's to audit.
    (_thread(reply_bodies=["ok"]), False),
])
def test_gate_treats_reviewer_persists_reply_as_unanswered(
    thread: ReviewThread, expected: bool,
) -> None:
    assert PRReviewer._is_unreplied_ai_thread(thread) is expected


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


class _Result:
    def __init__(self, persisting: list[PersistingFinding]) -> None:
        self.summary = "s"
        self.findings: list[ReviewFinding] = []
        self.diff = "d"
        self.refuted: list[object] = []
        self.revised: list[object] = []
        self.duplicates: list[object] = []
        self.persisting = persisting
        self.fixed: list[object] = []


ARGV = ["answerable", "--repo", "owner/repo", "--pr", "1"]


def test_cli_passes_persisting_to_post_review() -> None:
    pf = _persisting(_node("PRRT_h", HUMAN))
    with (
        patch.object(sys, "argv", ARGV),
        patch("answerable.cli.PRReviewer") as mock_cls,
        patch("answerable.cli.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = _Result([pf])
        mock_cls.return_value.post_review.return_value.posted = []
        mock_cls.return_value.post_review.return_value.reopened = [pf.finding]
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 1
    assert mock_cls.return_value.post_review.call_args.kwargs["persisting"] == [pf]


@pytest.mark.parametrize(("tracked", "code"), [(False, 1), (True, 0)])
def test_cli_dry_run_counts_persisting(
    tracked: bool, code: int, capsys: pytest.CaptureFixture[str],
) -> None:
    pf = _persisting(_node("PRRT_h", HUMAN))

    def _filter(findings: list[ReviewFinding]) -> list[ReviewFinding]:
        return [] if tracked else findings

    with (
        patch.object(sys, "argv", [*ARGV, "--dry-run"]),
        patch("answerable.cli.PRReviewer") as mock_cls,
        patch("answerable.cli.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = _Result([pf])
        mock_cls.return_value._filter_tracked_findings.side_effect = _filter
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == code
    out = capsys.readouterr().out
    assert "Still broken on resolved threads (would reopen): 1" in out
    assert "app/a.py:2 — Division by zero (reopens T0 (review thread))" in out
