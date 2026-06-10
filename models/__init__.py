#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GluFormer模型模块
包括基线模型和提出的文化感知推荐模型
"""

from .cultural_gate import CulturalGatingMechanism
from .cross_modal_fusion import CrossModalAttention
from .clinical_constraint import ClinicalConstraintLayer
from .recommender import CultureAwareRecommender

# Baseline models for comparison
from .sasrec import SASRecRecommender
from .bert4rec import BERT4RecRecommender
from .lightgcn import LightGCNRecommender

# New innovation modules
from .temporal import TemporalEncoder, TemporalUserProfile, MealTimingEncoder
from .multi_expert import CulturalExpert, ClinicalExpert, NutritionalExpert, ExpertRouter, MultiExpertModule

__all__ = [
    # Proposed models
    'CulturalGatingMechanism',
    'CrossModalAttention',
    'ClinicalConstraintLayer',
    'CultureAwareRecommender',

    # Baseline models
    'SASRecRecommender',
    'BERT4RecRecommender',
    'LightGCNRecommender',
]
