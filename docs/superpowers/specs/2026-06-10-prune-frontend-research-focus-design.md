# Prune Frontend Research Focus Design

## Verdict

Active frontend code should be removed from `ten-love` and from the local `system` source tree. The repository should present itself as a research engineering workspace for experiments, data governance, model code, result ledgers, and manuscript claim boundaries.

## Scope

This change removes only frontend surfaces and their active references:
- `frontend/` in the GitHub checkout.
- `frontend/` and `apps/frontend/` in the local source tree.
- README, audit, tracking, and manuscript draft statements that describe Vue/Vite as an active system component.

This change does not remove raw data, processed datasets, model weights, training outputs, or historical result files. Those assets require separate data-safety and hash-manifest gates.

## Research Rationale

The stated project goal is to build reliable experiments and evidence strong enough for a top-journal submission. A frontend demo does not currently support the core claims. Keeping it in the active repository increases maintenance surface and makes the project look like a product demo rather than a reproducible research package.

## Design Decisions

1. Use Git history as provenance for removed frontend code rather than creating a local archive directory.
2. Add an OpenSpec change named `prune-frontend-research-focus` to record intent, affected capability, tasks, and validation gates.
3. Keep backend service code and visualization-format helpers only when they support experiment execution or reproducible outputs.
4. Reword manuscript-facing claims away from enterprise/full-stack language and toward evidence-bounded research infrastructure.

## Verification

Verification must include:
- `openspec validate prune-frontend-research-focus --type change --strict --no-interactive`
- `rg -n "frontend|前端|Vue|Vite|pnpm" ...` to confirm remaining matches are historical or explicitly legacy-scoped
- `git status --short`
- `git diff --check`

Frontend build checks are intentionally out of scope because the frontend is removed.
