# Behavioural proof

A test makes a behaviour visible at a public seam.

- Input and expected output, including the significant failure.
- A stable, reproducible command named by the repository
  (`python -m pytest tests` in this tree).
- A proportionate level: unit or `contract` for a seam invariant;
  `integration` when several modules sit behind the same seam; `e2e` only
  for a significant multi-component journey.

Relevant negative cases are part of the contract. Data are deterministic.
A mock has a declared limit: it does not simulate coverage that is absent.

When a bug is the subject: reproduce the failure (red) with the test, then
fix, then obtain green. A green obtained by softening the test, ignoring
it, or accepting an unread snapshot is not proof.

A keyword, heading or prose test does not establish a behaviour. The
oracle changes when the contract changes, and that change is a separate
decision.

Before writing an observable behaviour, declare the public seam, the
significant failure modes, the level (`contract` | `integration` | `e2e`)
and the reproduction command. E2E is required only for a significant
multi-component journey; a declared E2E is medium or hard and produces a
replayable JSON report.
