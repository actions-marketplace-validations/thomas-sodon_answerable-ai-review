"""Tests for the adversarial verification pass — SDK calls are mocked."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from answerable.review import PRReviewer, RefutedFinding, ReviewFinding, RevisedFinding
from answerable.sdk import QueryResult, QueryUsage

DIFF = """\
diff --git a/app/a.py b/app/a.py
--- a/app/a.py
+++ b/app/a.py
@@ -1,2 +1,3 @@
 x = 1
+y = x / 0
 z = 2
"""

FINDINGS = [
    {"file": "app/a.py", "line": 2, "severity": "bug",
     "title": "Division by zero", "description": "y divides by zero",
     "fix_plan": "SECRET-FIX-PLAN", "suggested_code": "y = x"},
    {"file": "app/a.py", "line": 3, "severity": "suggestion",
     "title": "Unused z", "description": "z is never read",
     "fix_plan": "remove z"},
]


def _host() -> MagicMock:
    host = MagicMock()
    host.get_pr_diff.return_value = DIFF
    host.get_pr_info.return_value = {"title": "Add y", "body": ""}
    return host


def _result(data: object, tokens: int = 1000, cost: float | None = 0.05) -> QueryResult:
    return QueryResult(
        response=json.dumps(data),
        usage=QueryUsage(total_tokens=tokens, total_cost_usd=cost),
    )


def _review_result() -> QueryResult:
    return _result({"summary": "REVIEWER-SUMMARY", "findings": FINDINGS})


async def _run(*responses: object, verify: bool = True) -> tuple[object, AsyncMock]:
    reviewer = PRReviewer(
        "owner/repo", code_host=_host(), verify_findings=verify, dedupe_findings=False,
    )
    with patch("answerable.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = list(responses)
        result = await reviewer.review(1)
    return result, mock_query


@pytest.mark.asyncio
async def test_refuted_findings_are_dropped_and_recorded() -> None:
    verdicts = _result({"verdicts": [
        {"id": 0, "verdict": "confirmed", "reason": "really divides by zero"},
        {"id": 1, "verdict": "refuted", "reason": "z is read in b.py"},
    ]})
    result, mock_query = await _run(_review_result(), verdicts)

    assert mock_query.await_count == 2
    assert [f.title for f in result.findings] == ["Division by zero"]
    assert len(result.refuted) == 1
    assert isinstance(result.refuted[0], RefutedFinding)
    assert result.refuted[0].finding.title == "Unused z"
    assert result.refuted[0].reason == "z is read in b.py"


@pytest.mark.asyncio
async def test_verifier_runs_as_a_separate_fresh_query() -> None:
    """The verifier is its own query() call: a fresh session that sees the
    findings as claims but not the reviewer's summary or fix plan."""
    verdicts = _result({"verdicts": []})
    _, mock_query = await _run(_review_result(), verdicts)

    review_call, verify_call = mock_query.await_args_list
    verify_prompt = verify_call.kwargs["prompt"]
    assert verify_prompt != review_call.kwargs["prompt"]
    assert "adversarial verifier" in verify_prompt
    assert "``revised``" in verify_prompt
    assert "### Finding 0" in verify_prompt and "### Finding 1" in verify_prompt
    assert "Division by zero" in verify_prompt
    assert DIFF in verify_prompt
    # Not anchored on the reviewer's own argument.
    assert "REVIEWER-SUMMARY" not in verify_prompt
    assert "SECRET-FIX-PLAN" not in verify_prompt

    options = verify_call.kwargs["options"]
    assert set(options.allowed_tools) == {"Read", "Glob", "Grep"}
    assert options.model == "claude-opus-5-5"
    assert options.effort == "high"
    # QueryOptions carries no session-continuation knob, so the call cannot
    # resume the reviewer's session.
    for attr in ("resume", "continue_conversation", "session_id", "fork_session"):
        assert not hasattr(options, attr)


@pytest.mark.asyncio
async def test_missing_or_unknown_verdicts_keep_findings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    verdicts = _result({"verdicts": [
        {"id": 0, "verdict": "maybe"},
        {"id": "x", "verdict": "refuted"},
        {"id": True, "verdict": "refuted"},
        "junk",
    ]})
    with caplog.at_level(logging.WARNING, logger="answerable"):
        result, _ = await _run(_review_result(), verdicts)

    assert len(result.findings) == 2
    assert result.refuted == []
    assert "no verdict" in caplog.text


@pytest.mark.asyncio
async def test_first_verdict_for_an_id_wins() -> None:
    verdicts = _result({"verdicts": [
        {"id": 0, "verdict": "confirmed"},
        {"id": 0, "verdict": "refuted"},
        {"id": "1", "verdict": "REFUTED", "reason": "string id ok"},
    ]})
    result, _ = await _run(_review_result(), verdicts)

    assert [f.title for f in result.findings] == ["Division by zero"]
    assert [r.finding.title for r in result.refuted] == ["Unused z"]


@pytest.mark.asyncio
async def test_unparseable_verification_keeps_all_findings() -> None:
    garbage = QueryResult(response="I could not decide", usage=QueryUsage(total_tokens=5))
    result, _ = await _run(_review_result(), garbage)

    assert len(result.findings) == 2
    assert result.refuted == []


@pytest.mark.asyncio
async def test_verification_query_failure_keeps_all_findings() -> None:
    result, mock_query = await _run(_review_result(), RuntimeError("SDK down"))

    assert mock_query.await_count == 2
    assert len(result.findings) == 2
    assert result.refuted == []
    assert result.tokens_used == 1000


@pytest.mark.asyncio
async def test_usage_is_summed_across_both_sessions() -> None:
    verdicts = _result({"verdicts": []}, tokens=300, cost=0.01)
    result, _ = await _run(_review_result(), verdicts)

    assert result.tokens_used == 1300
    assert result.cost_usd == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_verification_cost_counts_when_review_cost_unknown() -> None:
    review = _result({"summary": "s", "findings": FINDINGS}, cost=None)
    verdicts = _result({"verdicts": []}, cost=0.01)
    result, _ = await _run(review, verdicts)

    assert result.cost_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_no_findings_skips_verification() -> None:
    result, mock_query = await _run(_result({"summary": "clean", "findings": []}))

    assert mock_query.await_count == 1
    assert result.findings == []


@pytest.mark.asyncio
async def test_verification_disabled_makes_one_call() -> None:
    result, mock_query = await _run(_review_result(), verify=False)

    assert mock_query.await_count == 1
    assert len(result.findings) == 2
    assert result.refuted == []


# ------------------------------------------------------------------
# Revised findings
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revised_finding_replaces_original_in_place() -> None:
    verdicts = _result({"verdicts": [
        {"id": 0, "verdict": "revised", "reason": "only on the zero path",
         "revised": {"line": 3, "severity": "suggestion",
                     "title": "Division by zero when x is 0",
                     "fix_plan": "guard x", "suggested_code": "y = x / 1"}},
        {"id": 1, "verdict": "confirmed"},
    ]})
    result, _ = await _run(_review_result(), verdicts)

    assert [f.title for f in result.findings] == [
        "Division by zero when x is 0", "Unused z",
    ]
    new = result.findings[0]
    assert (new.file, new.line, new.severity) == ("app/a.py", 3, "suggestion")
    assert new.description == "y divides by zero"  # omitted → original kept
    assert new.fix_plan == "guard x"
    assert new.suggested_code == "y = x / 1"

    assert len(result.revised) == 1
    rv = result.revised[0]
    assert isinstance(rv, RevisedFinding)
    assert rv.original.title == "Division by zero"
    assert rv.original.line == 2
    assert rv.revised is new
    assert rv.reason == "only on the zero path"
    assert result.refuted == []


@pytest.mark.asyncio
async def test_revision_never_carries_over_original_suggested_code() -> None:
    """The verifier never saw the original suggested_code, so it may no longer
    match the corrected finding — drop it unless the revision supplies one."""
    verdicts = _result({"verdicts": [
        {"id": 0, "verdict": "revised", "revised": {"description": "sharper"}},
    ]})
    result, _ = await _run(_review_result(), verdicts)

    new = result.findings[0]
    assert new.description == "sharper"
    assert new.suggested_code is None
    assert new.fix_plan == "SECRET-FIX-PLAN"  # falls back to the original


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [
    None,
    "not a dict",
    {},
    {"title": "Division by zero", "line": 2},  # no actual change
    {"title": "   ", "file": "", "severity": None},  # blanks only
])
async def test_unusable_revision_keeps_original(
    revision: object, caplog: pytest.LogCaptureFixture,
) -> None:
    entry: dict[str, object] = {"id": 0, "verdict": "revised"}
    if revision is not None:
        entry["revised"] = revision
    with caplog.at_level(logging.WARNING, logger="answerable"):
        result, _ = await _run(_review_result(), _result({"verdicts": [entry]}))

    original = result.findings[0]
    assert original.title == "Division by zero"
    assert original.suggested_code == "y = x"  # untouched original
    assert result.revised == []
    assert "without a usable revision" in caplog.text


def _finding() -> ReviewFinding:
    return ReviewFinding(
        file="a.py", line=5, severity="bug", title="T",
        description="D", fix_plan="F", suggested_code="code",
    )


@pytest.mark.parametrize(("revision", "field", "expected"), [
    ({"line": -3, "title": "T2"}, "line", 5),
    ({"line": "N/A", "title": "T2"}, "line", 5),
    ({"line": True, "title": "T2"}, "line", 5),
    ({"line": "7"}, "line", 7),
    ({"line": 0}, "line", 0),
    ({"severity": "BOGUS", "title": "T2"}, "severity", "bug"),
    ({"severity": " Nit "}, "severity", "nit"),
    ({"file": "  b.py  "}, "file", "b.py"),
    ({"file": 42, "title": "T2"}, "file", "a.py"),
])
def test_apply_revision_normalises_fields(
    revision: dict[str, object], field: str, expected: object,
) -> None:
    revised = PRReviewer._apply_revision(_finding(), revision)
    assert revised is not None
    assert getattr(revised, field) == expected


@pytest.mark.asyncio
async def test_mixed_verdicts_preserve_order() -> None:
    findings = [*FINDINGS, {
        "file": "app/a.py", "line": 1, "severity": "nit", "title": "Third",
        "description": "d", "fix_plan": "f",
    }]
    review = _result({"summary": "s", "findings": findings})
    verdicts = _result({"verdicts": [
        {"id": 0, "verdict": "refuted", "reason": "no"},
        {"id": 1, "verdict": "revised", "revised": {"title": "Unused z (revised)"}},
        {"id": 2, "verdict": "confirmed"},
    ]})
    result, _ = await _run(review, verdicts)

    assert [f.title for f in result.findings] == ["Unused z (revised)", "Third"]
    assert [r.finding.title for r in result.refuted] == ["Division by zero"]
    assert [r.original.title for r in result.revised] == ["Unused z"]
