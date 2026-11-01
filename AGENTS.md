# Sopa de Letras agent instructions

These instructions apply throughout this repository.

## Shared PWA baseline

Before making changes to the PWA, read
[`../pwa/AGENTS.md`](../pwa/AGENTS.md) as the shared trekm / BaseCentral PWA
baseline. Adopted revision: **2026-10-27, decision set 6**. Read the current shared
file on subsequent work; this adoption date does not freeze the baseline.

These are requirements for new work and audits, not a claim that the existing
app already complies. Preserve app-specific domain and data-integrity rules.
If the sibling baseline is unavailable, report that it could not be checked
and use a reviewed copy with its source/revision recorded; do not silently
substitute weaker rules or block unrelated safe maintenance.

Adoption alone does not authorize data migrations, runtime changes, pushes or
production deployment. Implement those only within the user's requested scope.

## Project context

Read [README.md](README.md) and relevant source before changing behavior.
The PWA runtime is in `pwa/`; keep the Python command-line tools working too.
The root-level `../pwa/AGENTS.md` reference above is the sibling shared-baseline
repository, not this project's `pwa/` runtime folder. Preserve puzzle semantics,
saved progress, languages and print/export behavior. Read root `deploy.sh` and
README before deployment-related work. Documentation-only adoption does not
require a runtime version bump.
