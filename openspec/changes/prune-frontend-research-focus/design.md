## Context

`system` began as a mixed research and application snapshot. Prior cleanup established data inventories and result boundaries, but the active repository still contained a frontend demo and language that emphasized full-stack application delivery. The current target is research quality: reproducible experiments, data provenance, evaluation protocols, and bounded claims.

## Goals / Non-Goals

**Goals:**
- Remove active frontend code from the research repository.
- Keep a clear provenance record through Git history and OpenSpec.
- Update documentation so reviewers see a research engineering package, not a product demo.
- Preserve all data and result artifacts for later hash-governed review.

**Non-Goals:**
- No deletion of data, model weights, checkpoints, or experiment outputs.
- No rewrite of research algorithms.
- No new frontend, dashboard, or UI replacement.
- No claim upgrade about model performance.

## Decisions

1. Delete active frontend directories instead of archiving them under the repository.
2. Keep backend and workflow visualization helpers only as research service or output-format utilities.
3. Update manuscript-facing language to describe model and experiment infrastructure, with frontend terms removed from active evidence claims.
4. Use OpenSpec validation and stale-reference search as the primary gates.

## Risks / Trade-offs

- Historical documents may still mention older frontend plans. Those mentions should be treated as legacy context if retained.
- Removing UI code reduces demo capability, but improves repository focus and reduces maintenance load.
- Some package metadata in the local source tree references frontend workspaces. Those references must be removed so local structure matches the research-first boundary.
