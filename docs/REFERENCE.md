# Answerable reference

Detailed behaviour of the CLI and the GitHub Action. For setup, see the
[README](../README.md).

- [Credentials](#credentials)
- [CLI usage](#cli-usage)
- [Exit codes](#exit-codes)
- [Configuration](#configuration)
- [GitHub Action inputs](#github-action-inputs)
- [De-duplication rules](#de-duplication-rules)

## Credentials

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` **or** `CLAUDE_CODE_OAUTH_TOKEN` | the review itself (not needed for `--require-comment-replies`) |
| `GH_TOKEN` / `GITHUB_TOKEN` (or a `gh auth login` session) | reading the PR, posting reviews, opening escalation issues |

The GitHub token needs `pull-requests: write` to post reviews and
`issues: write` for the disputed-finding escalation.

## CLI usage

```bash
answerable --repo owner/repo --pr 42                     # review and post
answerable --repo owner/repo --pr 42 --dry-run           # print findings, post nothing
answerable --repo owner/repo --pr 42 --max-diff-chars 100000 --max-files 75
answerable --repo owner/repo --pr 42 --require-comment-replies   # reply-coverage audit only
answerable --repo owner/repo --pr 42 --skip-verification         # post findings unverified
```

| Option | Default | Meaning |
|---|---|---|
| `--repo` | (required) | GitHub repository, `owner/repo` |
| `--pr` | (required) | Pull request number |
| `--max-diff-chars` | `80000` | Diff characters sent to the model before truncation |
| `--max-files` | `50` | Files included from the diff |
| `--gh-timeout` | `30` | Timeout in seconds for each `gh` call |
| `--dry-run` | off | Print the summary and findings to stdout instead of posting |
| `--require-comment-replies` | off | Check-only mode: skip the review, audit reply coverage |
| `--skip-verification` | off | Skip the adversarial verification pass and post the reviewer's findings as-is |

`--require-comment-replies` takes precedence: when it is set, `--dry-run` has
no effect and no diff review is run.

## Adversarial verification

After the review returns findings, a second `query()` call — a fresh agent
session that shares none of the reviewer's context — receives the PR title and
description, the same truncated diff, and each finding's site, severity, title
and description (not the reviewer's summary, fix plan or suggested code). It
tries to disprove each finding with the same read-only tools, model and effort,
and answers `confirmed` or `refuted` per finding.

- Only an explicit `refuted` verdict drops a finding. A missing verdict, an
  unparseable response, or a failed verification query keeps every finding;
  verification never fails the run.
- Refuted findings are not posted. They are logged with the verifier's reason,
  and `--dry-run` prints them under *Refuted by verification*.
- The pass is skipped when the review produced no findings, and roughly doubles
  model cost otherwise. Disable it with `--skip-verification`,
  `verify_findings = 0`, or the `verify-findings: false` action input.

## Exit codes

| Mode | Exit `0` | Exit `1` |
|---|---|---|
| Review (default) | Nothing new was posted: no findings, an empty diff, or every finding was already tracked by an issue, escalated to an issue, or matched an existing thread that is still open (or was resolved with an issue link) | At least one finding (any severity) was newly posted, or a bug was flagged as persisting with a reply on its resolved thread; **or** the review failed (API/SDK error); **or** posting the review failed (findings are then printed to the log) |
| `--dry-run` | No finding would be posted or flagged as persisting after the same issue/thread filtering a real run applies | At least one finding would be posted or flagged as persisting (if the filtering itself errors, any raw finding counts); **or** the review failed |
| `--require-comment-replies` | Every reviewer finding has been answered | At least one unanswered finding; **or** the audit could not read the PR (a `gh`/GraphQL outage fails closed) |

Invalid or missing arguments exit `2` (standard `argparse` behaviour).

A finding counts as **answered** when:

- *inline*: its thread is resolved, outdated, or has any reply;
- *overflow*: a PR comment containing the `<!-- answerable:overflow-ack -->` marker
  lists its `of-<hash>` handle on a line together with one of these verdict
  phrases: `Could not prove this defect`, `Could not prove this bug`,
  `Fixed automatically`, `Fixed locally with implementation-plan context.`,
  `Tracked in #`, `Declined with implementation-plan context.`,
  `Answered with implementation-plan context.`

## Configuration

Command-line options always win. Defaults for `max_diff_chars`, `max_files`,
`gh_timeout`, `require_comment_replies` and `verify_findings` can also come from a config file
and/or environment variables.

**Config file.** The first of these files that exists and has a non-empty
`[review]` section is used:

1. the path in `ANSWERABLE_CONFIG`;
2. `./answerable.toml`.

```toml
[review]
max_diff_chars = 80000
max_files = 50
gh_timeout = 30
require_comment_replies = 0   # 1 = run in reply-coverage audit mode by default
verify_findings = 1           # 0 = skip adversarial verification of findings
```

A file that fails to parse is ignored; a value that is not an integer falls
back to its default.

**Environment variables** override the config file:
`ANSWERABLE_MAX_DIFF_CHARS`, `ANSWERABLE_MAX_FILES`,
`ANSWERABLE_GH_TIMEOUT`, `ANSWERABLE_REQUIRE_COMMENT_REPLIES`,
`ANSWERABLE_VERIFY_FINDINGS`
(integers; blank or non-integer values are ignored).

The model and effort are fixed in `src/answerable/sdk.py` (`REVIEW_MODEL`,
`REVIEW_EFFORT`).

## GitHub Action inputs

| Input | Default | Meaning |
|---|---|---|
| `anthropic-api-key` | — | Anthropic API key |
| `claude-code-oauth-token` | — | Claude Code OAuth token (alternative to the key) |
| `github-token` | `github.token` | Token used by `gh` |
| `repo` | current repository | `owner/repo` |
| `pr-number` | triggering PR | Pull request number |
| `mode` | `review` | `review` or `replies` (`--require-comment-replies`; needs no Claude credential) |
| `dry-run` | `false` | `true` prints findings instead of posting |
| `verify-findings` | `true` | `false` skips the [adversarial verification](#adversarial-verification) pass |
| `max-diff-chars`, `max-files`, `gh-timeout` | tool defaults | Passed through to the CLI |
| `python-version` | `3.14` | Python used to run the reviewer |

The step's exit status is the CLI's exit code, so the job fails exactly when
the table under [Exit codes](#exit-codes) says it does. Pull requests from
forks do not receive repository secrets, so the review step cannot run there.

The action runs `actions/setup-python`, which puts the chosen `python-version`
on the job's `PATH`, so later steps in the same job use that Python too. Run
the action last, or set up your own Python again after it, if that matters.

The examples use the moving `@v0` tag for brevity. Since the action
receives an Anthropic credential, pin it to a full commit SHA in your own
workflows (`uses: thomas-sodon/answerable-ai-review@<commit-sha>`) so a later change
to a moving tag cannot run with your secret unreviewed.

## De-duplication rules

Before posting, the reviewer filters findings against the PR's current state:

- findings already tracked by an open GitHub issue are dropped;
- findings matching an existing review thread are skipped — except a
  bug-severity finding whose thread was resolved without an issue link while
  the problem is still there: the thread stays resolved, but the reviewer
  replies on it that the defect persists;
- findings on a thread where someone replied *"Could not prove this defect"*
  are escalated to a GitHub issue (labels `ai-review` and the severity)
  instead of being re-posted.
