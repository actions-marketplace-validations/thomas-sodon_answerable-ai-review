"""Tests for the code review prompt template."""

from __future__ import annotations

from answerable.prompt import CODE_REVIEW_PROMPT


def test_prompt_formats_with_review_placeholders() -> None:
    """The template formats with exactly the placeholders ``review`` passes."""
    rendered = CODE_REVIEW_PROMPT.format(pr_title="T", pr_body="B", diff="D")
    assert "Title: T" in rendered
