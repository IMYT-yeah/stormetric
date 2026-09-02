# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Instead, report them privately so we can fix the issue before it is
publicly disclosed:

1. Open a **private security advisory** on GitHub:
   https://github.com/IMYT-yeah/stormetric/security/advisories/new
2. Include as much of the following as possible:
   - Affected version(s) and commit SHA(s)
   - A minimal proof-of-concept / reproduction steps
   - Impact assessment (what an attacker could do)
   - Suggested fix (optional)

Alternatively, email the maintainers if GitHub advisories are
unavailable for your case — the address is listed in the advisory
form.

## What is in scope

- Code execution, remote or local, via package inputs (metric
  parameters, CLI arguments, data files)
- Dependency supply-chain issues (e.g. a compromised build dependency)
- Secrets or credentials accidentally committed to the repository
- Anything that would undermine the scientific integrity of the
  package's results

## Response timeline

| Time | Action |
|------|--------|
| 48 h  | Acknowledgment of the report |
| 1 week | Triage and confirmation of the issue |
| 90 days | A fix is published as a new release (or a clear explanation of why no fix is needed) |

We will credit reporters in the advisory unless anonymity is requested.

## Out of scope

- General bug reports (use the [bug report template](https://github.com/IMYT-yeah/stormetric/issues/new?template=bug_report.md))
- Theoretical/physically motivated disagreements with the metric model
  (those belong in Issues or Discussions)
