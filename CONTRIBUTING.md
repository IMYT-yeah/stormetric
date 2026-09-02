# Contributing to Stormetric

Thank you for considering contributing to Stormetric! This project is a
**scientific reproducibility effort** — every change to `main` is a
statement about the code underlying a testable alternative to GR.
Please read this guide before opening a pull request.

## Code of conduct

Be respectful, constructive, and scientifically rigorous. This project
welcomes contributions regardless of background or experience level.

## Development workflow

### 1. Fork & clone

1. Fork the repository on GitHub.
2. Clone your fork and add the upstream remote:

```bash
git clone https://github.com/<your-username>/stormetric.git
cd stormetric
git remote add upstream https://github.com/IMYT-yeah/stormetric.git
```

### 2. Branch strategy

- `main` is the **protected, release-candidate branch**. Never commit to
  it directly.
- Always work on a **feature branch** with a descriptive name:

```bash
git checkout -b feature/<short-description>
# or
git checkout -b fix/<short-description>
```

Branch naming conventions:

| Prefix   | Purpose                                |
|----------|----------------------------------------|
| `feature/` | New functionality or metrics          |
| `fix/`    | Bug fixes                             |
| `docs/`   | Documentation-only changes            |
| `refactor/` | Non-behavioural code restructuring  |

### 3. Set up a development environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Make changes with tests

- Every behavioural change must include or update tests in `tests/`.
- Run the full suite before pushing:

```bash
pytest tests/ -v
```

- Keep the code formatted:

```bash
black src tests examples
ruff check src tests
```

### 5. Open a pull request

- Use the provided [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- Target the `main` branch.
- CI must pass (3 OS × 5 Python versions, `pytest` + demo smoke test).
- If the PR changes scientific outputs (e.g. the shadow radius or the
  EMRI coupling), state the before/after numbers in the PR description.

## Review process

- `main` requires **at least 1 approval** and a green CI before merging.
- Maintainers may request changes; address them on the same branch.
- Squash-merging is preferred to keep `main` history clean.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`:

- [Bug report](https://github.com/IMYT-yeah/stormetric/issues/new?template=bug_report.md)
- [Feature request](https://github.com/IMYT-yeah/stormetric/issues/new?template=feature_request.md)

For **security vulnerabilities**, do **not** open a public issue — follow
the process in [SECURITY.md](SECURITY.md).

## Release process

Releases are tagged on `main` (e.g. `v0.1.0`) and, once publishing is
set up, uploaded to PyPI by CI. Each release is tied to a specific
commit SHA so papers and code versions stay traceable.
