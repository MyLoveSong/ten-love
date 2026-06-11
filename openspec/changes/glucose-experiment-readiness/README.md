# Glucose Experiment Readiness

This OpenSpec change defines the evidence gate for making `projects/glucose/`
the primary experiment line.

Status: gate defined, not yet passed.

The gate does not run new training, move data, delete data, or upgrade any
claim. It records the conditions that must be satisfied before prior local
Glucose results can support manuscript figures or main claims. Current
metric definitions are local-only. The old public-preprocessed candidate is
closed as unresolved for manuscript canonical use, and the new BigIdeas-only
draft candidate still requires full baseline parity, final leakage audit, and
claim-boundary review.
