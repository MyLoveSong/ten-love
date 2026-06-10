#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artifact bundling utilities for Stage2 reports.

Provides helpers to package Stage1 verification outputs with
Stage2 validation summaries so that downstream reviewers can
consume a single bundle.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def bundle_stage1_stage2_reports(
    stage1_report: Path,
    stage2_report: Path,
    bundle_root: Path
) -> Optional[Path]:
    """
    Bundle Stage1 verification and Stage2 quick-validation reports.

    Args:
        stage1_report: Path to Stage1 verification JSON.
        stage2_report: Path to Stage2 quick validation JSON.
        bundle_root: Directory where bundles should be emitted.

    Returns:
        Path to the generated zip archive, or None if bundling failed.
    """
    stage1_report = Path(stage1_report)
    stage2_report = Path(stage2_report)
    bundle_root = Path(bundle_root)

    if not stage1_report.exists():
        logger.warning(
            "Stage1 verification report not found at %s; skip bundling", stage1_report
        )
        return None
    if not stage2_report.exists():
        logger.warning(
            "Stage2 quick validation report not found at %s; skip bundling", stage2_report
        )
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = bundle_root / "stage1_stage2_bundles" / f"bundle_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(stage1_report, bundle_dir / stage1_report.name)
        shutil.copy2(stage2_report, bundle_dir / stage2_report.name)

        archive_path = shutil.make_archive(
            str(bundle_dir),
            "zip",
            root_dir=bundle_dir
        )
        archive_path = Path(archive_path)
        logger.info(
            "Stage1+Stage2 bundle created: %s", archive_path
        )
        return archive_path
    except Exception as exc:
        logger.error("Failed to create Stage1+Stage2 bundle: %s", exc)
        return None
