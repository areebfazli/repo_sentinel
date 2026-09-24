# Vendored static-analysis rules (Semgrep format)

Used by `backend/app/core/semgrep_scanner.py` (ROADMAP 1f step 5). Vendored so a
scan needs no network; the scanner runs the whole directory (every language's
rules; the engine applies only the rules matching each target's language).

## Source

- Repository: https://gitlab.com/gitlab-org/security-products/sast-rules
- Commit: `53bf5cf6df3c51b6c02110f5a638b5e6213666cd` (2026-09-21)
- Only the `rule-*.yml` files were copied (no test fixtures, no mappings). Rule
  YAML is unmodified; every file keeps its upstream licence header.

| here | upstream path | rules | licence (per-file header) | derived from |
|---|---|---|---|---|
| `python/` | `python/` | 68 | MIT (c) GitLab Inc. | Bandit (Apache-2.0) |
| `javascript/` | `javascript/` | 11 | MIT (c) GitLab Inc. | eslint-plugin-security (Apache-2.0) |
| `javascript-lgpl3/` | `rules/lgpl/javascript/` | 83 | LGPL-3.0 (`javascript-lgpl3/LICENSE`) | njsscan (LGPL-3.0) |
| `go/` | `go/` | 27 | 26 Apache-2.0 (c) gosec, 1 MIT (c) GitLab | gosec (Apache-2.0) |
| `java/` | `java/` | 55 | MIT (c) GitLab Inc. | find-sec-bugs |

`LICENSE-MIT-GitLab` is the repository's top-level licence (MIT for content
outside `doc/`, `ee/`, `jh/`).

## Deliberately NOT vendored

- `rules/lgpl-cc/` of the same repo and the Semgrep-maintained rules
  (`semgrep/semgrep-rules`, registry packs `p/python`, `p/javascript`, and the
  archived `opengrep/opengrep-rules` fork): LGPL-2.1 **plus Commons Clause**
  (fork) or the **Semgrep Rules License v1.0** (current upstream, "internal
  business purposes" only, no distribution, no offering "as a service" —
  https://semgrep.dev/legal/rules-license). A PR-review product must not bundle
  or serve them.
- `rules/gitlab/`: GitLab Enterprise Edition licence (needs a GitLab
  subscription).
- `trailofbits/semgrep-rules`: AGPL-3.0; narrow coverage for our languages.

## Updating

Re-copy the same directories from a newer commit, check each new file's licence
header, and update the commit above. Rule ids are the upstream ids (e.g.
`python_exec_rule-subprocess-popen-shell-true`); they appear in reports, so
renames upstream change what users see.
