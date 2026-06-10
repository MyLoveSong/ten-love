# System Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a first-round, evidence-bounded cleanup of `/home/data/xzy/system` without moving large data.

**Architecture:** Keep the current three research tracks in place: nutrition, recommendation, and glucose. Add lightweight governance documents and fix only syntax/import/secret issues that block static verification.

**Tech Stack:** Python 3.12, Vue/Vite frontend, FastAPI-style backend modules, local Markdown documentation.

---

### Task 1: Repair Static Syntax Blockers

**Files:**
- Modify: `projects/glucose/src/real_data_collector.py`
- Modify: `backend/app/modules/__init__.py`
- Modify: `backend/app/modules/glucose_prediction/__init__.py`
- Modify: `backend/app/modules/image_recognition/__init__.py`

- [x] Fix the mis-indented `except` block in `real_data_collector.py`.
- [x] Replace illegal `from app..` imports with package-relative imports.
- [x] Run `python3 -m compileall -q` on the repaired files.

### Task 2: Remove Hardcoded API Key Fallback

**Files:**
- Modify: `projects/glucose/src/real_data_collector.py`
- Modify: `.env.example`

- [x] Replace hardcoded external API key fallback with `os.getenv("USDA_API_KEY", "")`.
- [x] Keep `openfoodfacts` as a no-key provider marker.
- [x] Add `USDA_API_KEY` placeholder to `.env.example`.

### Task 3: Add Audit Documents

**Files:**
- Create: `PROJECT_AUDIT.md`
- Create: `DATA_INVENTORY.md`
- Create: `RESULTS_LEDGER.md`

- [x] Summarize what the project contains and which claims are supported.
- [x] Record data directories, approximate sizes, role, and safety action.
- [x] Record key result files and claim boundaries.

### Task 4: Align README And Ignore Rules

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [x] Correct tech stack mismatch: frontend is Vue/Vite, not React/Ant Design.
- [x] Add first-round status, known issues, and verification commands.
- [x] Ignore large data, model checkpoints, env folders, outputs, cache, and local secrets.

### Task 5: Verify And Clean Generated Cache

**Files:**
- No intended source changes.

- [x] Run Python compile checks.
- [x] Run environment checks.
- [x] Remove generated `__pycache__` directories from `system/`.
- [x] Record commands and residual risks in final response.
