---
name: Pull Request
about: Submit changes to Stormetric
title: ''
labels: ''
assignees: ''
---

## Summary

_What does this PR do, and why? Link any related issue._

Closes #(issue)

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor (no behavioural change)
- [ ] Scientific-output change (shadow radius, PPN, EMRI numbers…)

## Checklist

- [ ] `pytest tests/ -v` passes locally
- [ ] `black src tests examples` applied
- [ ] `ruff check src tests` clean
- [ ] Tests added/updated for new behaviour
- [ ] CHANGELOG.md updated (user-visible changes)
- [ ] README/docs updated if public API changed

## Scientific impact (if applicable)

_Before/after values for any observable the PR touches, e.g.:_

| Quantity | Before | After |
|----------|--------|-------|
| `b_shadow` (GM/c²) | | |
| `δ` relative deviation | | |

## Verification

_Describe how you verified the change (commands run, tests, manual
checks)._
