"""Open review threads whose defect is fixed get resolved, with a reply."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from answerable import cli
from answerable.github import GitHubClient, GitHubError
from answerable.markers import _AUTO_RESOLVED_MARKER, _DEFECT_PERSISTS_MARKER
from answerable.review import (
    DuplicateFinding,
    FixedThread,
    PostReviewResult,
    PRReviewer,
    ReviewFinding,
    ReviewResult,
    ReviewThread,
)
from answerable.sdk import QueryResult, QueryUsage

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

DIFF = """\
diff --git a/app/a.py b/app/a.py
--- a/app/a.py
+++ b/app/a.py
@@ -1,2 +1,3 @@
 x = 1
+y = x / 1
 z = 2
"""

AI_ROOT = "\U0001f41b **Division by zero**\n\ny divides by zero"
HUMAN_ROOT = "Careful: x / 0 raises here."
PERSISTS = f"{_DEFECT_PERSISTS_MARKER} in the current PR:\n\n..."
AUTO = f"{_AUTO_RESOLVED_MARKER} in the current PR, so this thread was resolved.\n\nguarded"


def _node(
    thread_id: str = "PRRT_1", body: str = AI_ROOT, *, resolved: bool = False,
    outdated: bool = False, path: str = "app/a.py", comment_id: int = 100,
    replies: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = {"databaseId": comment_id, "path": path, "body": body, "line": 2,
            "author": {"login": "github-actions"}}
    return {
        "id": thread_id, "isResolved": resolved, "isOutdated": outdated,
        "comments": {"nodes": [root, *({"body": r} for r in replies)]},
    }


def _thread(node: dict[str, Any]) -> ReviewThread:
    (thread,) = PRReviewer._parse_review_threads([node])
    return thread


def _host(nodes: list[dict[str, Any]]) -> MagicMock:
    host = MagicMock()
    host.get_pr_diff.return_value = DIFF
    host.get_pr_info.return_value = {"title": "Guard y", "body": ""}
    host.get_pr_review_threads.return_value = nodes
    host.get_pr_review_bodies.return_value = []
    host.get_pr_comment_bodies.return_value = []
    host.list_issues.return_value = []
    host.resolve_review_thread.return_value = True
    return host


def _result(data: object, tokens: int = 1000, cost: float | None = 0.05) -> QueryResult:
    return QueryResult(
        response=json.dumps(data),
        usage=QueryUsage(total_tokens=tokens, total_cost_usd=cost),
    )


def _review(findings: list[dict[str, Any]] | None = None) -> QueryResult:
    return _result({"summary": "s", "findings": findings or []})


def _fixed(idx: int = 0, reason: str = "x is checked before the division") -> dict[str, Any]:
    return {"id": idx, "verdict": "fixed", "reason": reason}


async def _run(
    host: MagicMock, *responses: object, dedupe: bool = False,
) -> tuple[ReviewResult, AsyncMock]:
    reviewer = PRReviewer(
        "owner/repo", code_host=host, verify_findings=False, dedupe_findings=dedupe,
    )
    with patch("answerable.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = list(responses)
        result = await reviewer.review(1)
    return result, mock_query


BUG = {"file": "app/a.py", "line": 2, "severity": "bug",
       "title": "Division by zero", "description": "y divides by zero",
       "fix_plan": "guard"}


# ------------------------------------------------------------------
# GitHubClient.resolve_review_thread
# ------------------------------------------------------------------


class TestResolveReviewThread:
    def _client_run(self, stdout: str) -> tuple[GitHubClient, MagicMock]:
        client = GitHubClient(repo="owner/repo")
        result = MagicMock()
        result.stdout = stdout
        return client, MagicMock(return_value=result)

    def test_sends_resolve_mutation(self) -> None:
        client, run = self._client_run(json.dumps(
            {"data": {"resolveReviewThread": {"thread": {"id": "PRRT_1", "isResolved": True}}}},
        ))
        with patch.object(client, "_run", run):
            assert client.resolve_review_thread("PRRT_1") is True
        cmd = run.call_args[0][0]
        assert cmd[:3] == ["gh", "api", "graphql"]
        assert any("resolveReviewThread" in a and "unresolve" not in a for a in cmd)
        assert "threadId=PRRT_1" in cmd

    @pytest.mark.parametrize("payload", [
        {"data": None, "errors": [{"message": "nope"}]},
        {"data": {"resolveReviewThread": {"thread": {"id": "PRRT_1", "isResolved": False}}}},
        {"data": {"resolveReviewThread": None}},
        [],
    ])
    def test_not_resolved_returns_false(
        self, payload: object, caplog: pytest.LogCaptureFixture,
    ) -> None:
        client, run = self._client_run(json.dumps(payload))
        with patch.object(client, "_run", run), caplog.at_level(logging.WARNING):
            assert client.resolve_review_thread("PRRT_1") is False
        assert "resolve review thread" in caplog.text

    def test_gh_failure_returns_false(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run", side_effect=GitHubError("boom")):
            assert client.resolve_review_thread("PRRT_1") is False

    def test_empty_thread_id_is_a_no_op(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as run:
            assert client.resolve_review_thread("") is False
        run.assert_not_called()


class TestReplyToReviewComment:
    def test_returns_true_when_posted(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as run:
            assert client.reply_to_review_comment(1, 5, "hi") is True
        run.assert_called_once()

    def test_returns_false_on_gh_failure(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run", side_effect=GitHubError("boom")):
            assert client.reply_to_review_comment(1, 5, "hi") is False

    def test_returns_false_without_repo(self) -> None:
        assert GitHubClient().reply_to_review_comment(1, 5, "hi") is False


# ------------------------------------------------------------------
# Which threads may be resolved
# ------------------------------------------------------------------


@pytest.mark.parametrize(("node", "expected"), [
    (_node(), True),  # reviewer's own open finding
    (_node(outdated=True), True),  # code changed under it
    (_node(replies=("Will fix",)), True),  # answered but still open
    (_node(resolved=True), False),  # already resolved
    (_node(replies=("Tracked in #12",)), False),  # an issue tracks it
    (_node(body=HUMAN_ROOT), False),  # someone else's thread
    (_node(body=HUMAN_ROOT, replies=(PERSISTS,)), True),  # reviewer reopened it
    (_node(body=HUMAN_ROOT, replies=(PERSISTS, "on it")), True),
    (_node(replies=(AUTO,)), False),  # resolved once, someone reopened it
    (_node(replies=(AUTO, "still broken")), False),
    (_node(replies=(AUTO, PERSISTS)), True),  # reopened by the reviewer since
    (_node(thread_id=""), False),  # cannot be resolved without an id
])
def test_is_fix_candidate(node: dict[str, Any], expected: bool) -> None:
    assert PRReviewer._is_fix_candidate(_thread(node)) is expected


# ------------------------------------------------------------------
# Fix-check session
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fixed_thread_is_recorded_even_without_new_findings() -> None:
    host = _host([_node("PRRT_1", outdated=True)])
    result, mock_query = await _run(host, _review(), _result({"verdicts": [_fixed()]}))

    assert mock_query.await_count == 2
    (ft,) = result.fixed
    assert ft.thread.thread_id == "PRRT_1"
    assert ft.reason == "x is checked before the division"

    prompt = mock_query.await_args_list[1].kwargs["prompt"]
    assert "### Thread 0 — app/a.py:2 (outdated)" in prompt
    assert "**Division by zero**" in prompt
    assert "y = x / 1" in prompt  # the diff is included
    assert mock_query.await_args_list[1].kwargs["options"].allowed_tools == ["Read", "Glob", "Grep"]


@pytest.mark.asyncio
@pytest.mark.parametrize("verdicts", [
    [{"id": 0, "verdict": "present", "reason": "still divides"}],
    [{"id": 0, "verdict": "unsure", "reason": "can't tell"}],
    [{"id": 0, "verdict": "fixed", "reason": ""}],  # no evidence
    [{"id": 0, "verdict": "FIXED-ish", "reason": "r"}],  # unknown verdict
    [{"id": 1, "verdict": "fixed", "reason": "r"}],  # id that was not shown
    [{"id": True, "verdict": "fixed", "reason": "r"}],
    [],
])
async def test_anything_but_a_reasoned_fixed_verdict_leaves_thread_open(
    verdicts: list[dict[str, Any]],
) -> None:
    result, _ = await _run(_host([_node()]), _review(), _result({"verdicts": verdicts}))
    assert result.fixed == []


@pytest.mark.asyncio
async def test_first_verdict_per_thread_wins() -> None:
    result, _ = await _run(_host([_node()]), _review(), _result({"verdicts": [
        {"id": 0, "verdict": "present", "reason": "still there"}, _fixed(),
    ]}))
    assert result.fixed == []


@pytest.mark.asyncio
async def test_query_failure_resolves_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result, _ = await _run(_host([_node()]), _review(), RuntimeError("SDK down"))
    assert result.fixed == []
    assert "Fix-check session failed" in caplog.text


@pytest.mark.asyncio
async def test_unparseable_response_resolves_nothing() -> None:
    bad = QueryResult(response="not json", usage=QueryUsage())
    result, _ = await _run(_host([_node()]), _review(), bad)
    assert result.fixed == []


@pytest.mark.asyncio
async def test_thread_fetch_failure_resolves_nothing() -> None:
    host = _host([])
    host.get_pr_review_threads.side_effect = RuntimeError("gh down")
    result, mock_query = await _run(host, _review())
    assert result.fixed == []
    assert mock_query.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("node", [
    _node(resolved=True), _node(body=HUMAN_ROOT), _node(replies=("Tracked in #3",)),
])
async def test_no_candidates_means_no_session(node: dict[str, Any]) -> None:
    result, mock_query = await _run(_host([node]), _review())
    assert mock_query.await_count == 1
    assert result.fixed == []


@pytest.mark.asyncio
async def test_thread_a_current_finding_matches_is_not_checked() -> None:
    """The reviewer just found that defect again, so it is not fixed."""
    other = _node("PRRT_2", "\U0001f41b **Lock never acquired**\n\nlock is None", comment_id=200)
    host = _host([_node("PRRT_1"), other])
    result, mock_query = await _run(
        host, _review([BUG]), _result({"verdicts": [_fixed()]}),
    )

    prompt = mock_query.await_args_list[1].kwargs["prompt"]
    assert "Lock never acquired" in prompt
    assert "Division by zero" not in prompt.split("## Open review threads")[1]
    assert [ft.thread.thread_id for ft in result.fixed] == ["PRRT_2"]


@pytest.mark.asyncio
async def test_only_candidate_matched_by_a_finding_means_no_session() -> None:
    result, mock_query = await _run(_host([_node()]), _review([BUG]))
    assert mock_query.await_count == 1
    assert result.fixed == []


@pytest.mark.asyncio
async def test_thread_the_dedupe_matched_is_not_checked() -> None:
    """A finding the dedupe session tied to a thread by meaning, not title,
    still says that thread's defect is present."""
    reworded = {**BUG, "title": "Zero denominator in ratio"}
    host = _host([_node()])
    dedupe = _result({"duplicates": [{"id": 0, "existing": "T0", "reason": "same"}]})
    result, mock_query = await _run(host, _review([reworded]), dedupe, dedupe=True)

    assert mock_query.await_count == 2  # review + dedupe, no fix check
    assert result.duplicates[0].thread is not None
    assert result.fixed == []


@pytest.mark.asyncio
async def test_disabled_runs_no_session() -> None:
    reviewer = PRReviewer(
        "owner/repo", code_host=_host([_node()]), verify_findings=False,
        dedupe_findings=False, resolve_fixed=False,
    )
    with patch("answerable.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = [_review()]
        result = await reviewer.review(1)
    assert mock_query.await_count == 1
    assert result.fixed == []


@pytest.mark.asyncio
async def test_usage_summed() -> None:
    result, _ = await _run(
        _host([_node()]), _review(),
        _result({"verdicts": [_fixed()]}, tokens=300, cost=0.01),
    )
    assert result.tokens_used == 1300
    assert result.cost_usd == pytest.approx(0.06)


def test_prompt_budget_leaves_overflowing_threads_out() -> None:
    reviewer = PRReviewer("owner/repo", code_host=_host([]))
    big = AI_ROOT + "\n" + "x" * 5_000
    threads = [_thread(_node(f"PRRT_{i}", big)) for i in range(40)]
    text, shown = reviewer._format_fix_candidates(threads)
    assert 0 < len(shown) < 40
    assert f"### Thread {len(shown) - 1} " in text
    assert f"### Thread {len(shown)} " not in text


# ------------------------------------------------------------------
# post_review
# ------------------------------------------------------------------


def _fixed_entry(node: dict[str, Any], reason: str = "guarded at line 2") -> FixedThread:
    return FixedThread(thread=_thread(node), reason=reason)


def test_fixed_thread_is_resolved_then_replied_to() -> None:
    node = _node("PRRT_1", comment_id=555)
    host = _host([node])
    reviewer = PRReviewer("owner/repo", code_host=host)

    out = reviewer.post_review(1, [], "s", diff=DIFF, fixed=[_fixed_entry(node)])

    assert [t.thread_id for t in out.resolved] == ["PRRT_1"]
    assert out.posted == [] and out.reopened == []
    host.post_pr_review.assert_not_called()
    resolve_then_reply = [
        c for c in host.mock_calls
        if c[0] in ("resolve_review_thread", "reply_to_review_comment")
    ]
    assert resolve_then_reply[0] == call.resolve_review_thread("PRRT_1")
    _, (pr, comment_id, body), _ = resolve_then_reply[1]
    assert (pr, comment_id) == (1, 555)
    assert body.startswith(_AUTO_RESOLVED_MARKER)
    assert "guarded at line 2" in body


def test_resolve_failure_posts_no_reply(caplog: pytest.LogCaptureFixture) -> None:
    node = _node()
    host = _host([node])
    host.resolve_review_thread.return_value = False
    reviewer = PRReviewer("owner/repo", code_host=host)

    with caplog.at_level(logging.WARNING):
        out = reviewer.post_review(1, [], "s", diff=DIFF, fixed=[_fixed_entry(node)])

    assert out.resolved == []
    host.reply_to_review_comment.assert_not_called()
    assert "Could not resolve fixed thread" in caplog.text


@pytest.mark.parametrize("failure", ["returns_false", "raises"])
def test_reply_failure_unresolves_the_thread_again(
    failure: str, caplog: pytest.LogCaptureFixture,
) -> None:
    """Without the marker reply nothing records that the reviewer resolved the
    thread, so a later run could resolve it again over a person's unresolve.
    The resolve is undone instead, and the next run retries both steps."""
    node = _node()
    host = _host([node])
    host.unresolve_review_thread.return_value = True
    if failure == "raises":
        host.reply_to_review_comment.side_effect = RuntimeError("blip")
    else:
        host.reply_to_review_comment.return_value = False
    reviewer = PRReviewer("owner/repo", code_host=host)

    with caplog.at_level(logging.WARNING):
        out = reviewer.post_review(1, [], "s", diff=DIFF, fixed=[_fixed_entry(node)])

    assert out.resolved == []
    host.resolve_review_thread.assert_called_once_with("PRRT_1")
    host.unresolve_review_thread.assert_called_once_with("PRRT_1")
    assert "unresolved it again" in caplog.text
    # The thread is back where it was: still a candidate for the next run.
    (fresh,) = PRReviewer._parse_review_threads(host.get_pr_review_threads(1))
    assert PRReviewer._is_fix_candidate(fresh)


def test_reply_and_unresolve_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    node = _node()
    host = _host([node])
    host.reply_to_review_comment.return_value = False
    host.unresolve_review_thread.return_value = False
    reviewer = PRReviewer("owner/repo", code_host=host)

    with caplog.at_level(logging.WARNING):
        out = reviewer.post_review(1, [], "s", diff=DIFF, fixed=[_fixed_entry(node)])

    assert out.resolved == []
    assert "stays resolved without a reply" in caplog.text


@pytest.mark.parametrize("fresh", [
    _node(resolved=True),  # someone resolved it meanwhile
    _node(replies=("No, still broken",)),  # a reply arrived after the check
    _node(replies=("Tracked in #7",)),  # now tracked by an issue
    _node("PRRT_other"),  # the checked thread is gone
])
def test_thread_changed_since_the_check_is_left_alone(fresh: dict[str, Any]) -> None:
    host = _host([fresh])
    reviewer = PRReviewer("owner/repo", code_host=host)

    out = reviewer.post_review(1, [], "s", diff=DIFF, fixed=[_fixed_entry(_node())])

    assert out.resolved == []
    host.resolve_review_thread.assert_not_called()
    host.reply_to_review_comment.assert_not_called()


def test_thread_a_posted_finding_matches_is_left_alone() -> None:
    node = _node()
    host = _host([node])
    reviewer = PRReviewer("owner/repo", code_host=host)
    finding = ReviewFinding(**{k: BUG[k] for k in ("file", "line", "severity", "title", "description", "fix_plan")})

    out = reviewer.post_review(1, [finding], "s", diff=DIFF, fixed=[_fixed_entry(node)])

    assert out.resolved == []
    host.resolve_review_thread.assert_not_called()


def test_same_thread_listed_twice_is_resolved_once() -> None:
    node = _node()
    host = _host([node])
    reviewer = PRReviewer("owner/repo", code_host=host)

    out = reviewer.post_review(
        1, [], "s", diff=DIFF, fixed=[_fixed_entry(node), _fixed_entry(node)],
    )

    assert len(out.resolved) == 1
    host.resolve_review_thread.assert_called_once_with("PRRT_1")


def test_resolves_alongside_new_findings() -> None:
    fixed_node = _node("PRRT_1")
    host = _host([fixed_node])
    reviewer = PRReviewer("owner/repo", code_host=host)
    new = ReviewFinding(
        file="app/a.py", line=2, severity="bug", title="Lock never acquired",
        description="lock is None", fix_plan="init",
    )

    out = reviewer.post_review(1, [new], "s", diff=DIFF, fixed=[_fixed_entry(fixed_node)])

    assert out.posted == [new]
    assert [t.thread_id for t in out.resolved] == ["PRRT_1"]
    host.post_pr_review.assert_called_once()


# ------------------------------------------------------------------
# Reply-coverage gate
# ------------------------------------------------------------------


@pytest.mark.parametrize(("node", "unreplied"), [
    (_node(resolved=True, replies=(AUTO,)), False),  # resolved: answered
    (_node(replies=(AUTO,)), True),  # someone reopened it without a word
    (_node(replies=(AUTO, "It is still broken because ...")), False),
])
def test_gate_does_not_count_the_auto_resolve_reply(
    node: dict[str, Any], unreplied: bool,
) -> None:
    assert PRReviewer._is_unreplied_ai_thread(_thread(node)) is unreplied


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

ARGV = ["answerable", "--repo", "owner/repo", "--pr", "1"]


@dataclass
class _Result:
    fixed: list[FixedThread]
    summary: str = "s"
    findings: list[ReviewFinding] = field(default_factory=list)
    diff: str = "d"
    refuted: list[object] = field(default_factory=list)
    revised: list[object] = field(default_factory=list)
    duplicates: list[DuplicateFinding] = field(default_factory=list)
    persisting: list[object] = field(default_factory=list)


def test_cli_passes_fixed_and_resolving_does_not_fail_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ft = _fixed_entry(_node())
    with (
        patch.object(sys, "argv", ARGV),
        patch("answerable.cli.PRReviewer") as mock_cls,
        patch("answerable.cli.asyncio") as mock_asyncio,
        caplog.at_level(logging.INFO),
    ):
        mock_asyncio.run.return_value = _Result([ft])
        mock_cls.return_value.post_review.return_value = PostReviewResult(
            posted=[], reopened=[], filtered=[], skipped=[], resolved=[ft.thread],
        )
        cli.main()  # no SystemExit: exit 0

    assert mock_cls.return_value.post_review.call_args.kwargs["fixed"] == [ft]
    assert mock_cls.call_args.kwargs["resolve_fixed"] is True
    assert "Resolved 1 review thread(s)" in caplog.text


def test_cli_dry_run_lists_threads_it_would_resolve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ft = _fixed_entry(_node())
    with (
        patch.object(sys, "argv", [*ARGV, "--dry-run"]),
        patch("answerable.cli.PRReviewer") as mock_cls,
        patch("answerable.cli.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = _Result([ft])
        mock_cls.return_value._filter_tracked_findings.side_effect = lambda f: f
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Fixed in the current code (would resolve): 1" in out
    assert "app/a.py:2 — Division by zero: guarded at line 2" in out
    mock_cls.return_value.post_review.assert_not_called()


@pytest.mark.parametrize(("argv", "env", "expected"), [
    ([*ARGV, "--skip-resolve-fixed"], {}, False),
    (ARGV, {"ANSWERABLE_RESOLVE_FIXED_THREADS": "0"}, False),
    (ARGV, {}, True),
])
def test_cli_skip_flag_and_env(
    argv: list[str], env: dict[str, str], expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with (
        patch.object(sys, "argv", argv),
        patch("answerable.cli.PRReviewer") as mock_cls,
        patch("answerable.cli.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = _Result([])
        mock_cls.return_value.post_review.return_value = PostReviewResult(
            posted=[], reopened=[], filtered=[], skipped=[],
        )
        cli.main()
    assert mock_cls.call_args.kwargs["resolve_fixed"] is expected
