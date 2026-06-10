# Prune Frontend Research Focus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove active frontend code and update repository documentation so `ten-love` is focused on reproducible research assets.

**Architecture:** Treat the repository as a research package with model code, data governance, result evidence, and manuscript claim boundaries. Record the scope change through OpenSpec and Superpowers docs, then remove frontend paths and stale references.

**Tech Stack:** Python research modules, Markdown governance docs, OpenSpec 1.3.1.

---

### Task 1: Create Change Records

**Files:**
- Create: `openspec/changes/prune-frontend-research-focus/proposal.md`
- Create: `openspec/changes/prune-frontend-research-focus/design.md`
- Create: `openspec/changes/prune-frontend-research-focus/tasks.md`
- Create: `openspec/changes/prune-frontend-research-focus/specs/research-repo-governance/spec.md`
- Create: `docs/superpowers/specs/2026-06-10-prune-frontend-research-focus-design.md`
- Create: `docs/superpowers/plans/2026-06-10-prune-frontend-research-focus.md`

- [x] 1.1 Add OpenSpec proposal, design, tasks, and requirement delta.
- [x] 1.2 Add Superpowers design and implementation plan.

### Task 2: Remove Active Frontend Code

**Files:**
- Delete: `frontend/`
- Delete locally only: `/home/data/xzy/system/apps/frontend/`

- [x] 2.1 Remove `frontend/` from the GitHub checkout.
- [x] 2.2 Remove `frontend/` and `apps/frontend/` from the local source tree.

### Task 3: Update Research Scope Documentation

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_AUDIT.md`
- Modify: `GITHUB_TRACKING_MANIFEST.md`
- Modify: `backend/app/services/论文.md`
- Modify locally only: `/home/data/xzy/system/package.json`
- Modify locally only: `/home/data/xzy/system/pnpm-workspace.yaml`

- [x] 3.1 Remove Vue/Vite from active technical stack descriptions.
- [x] 3.2 Mark frontend as removed from active scope with Git history as provenance.
- [x] 3.3 Replace enterprise/full-stack manuscript claims with research-infrastructure claims.
- [x] 3.4 Remove frontend workspace references from local package metadata.

### Task 4: Verify And Publish

**Files:**
- No additional source files.

- [x] 4.1 Run OpenSpec validation.
- [x] 4.2 Run stale frontend-reference search.
- [x] 4.3 Run `git diff --check`.
- [x] 4.4 Stage the intended `ten-love` changes explicitly.
- [x] 4.5 Commit and push to `origin/main`.
