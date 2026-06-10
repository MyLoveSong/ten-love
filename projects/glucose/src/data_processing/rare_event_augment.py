"""
Rare event augmentation for glucose prediction.
Focuses on low/high glucose events and critical transitions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import random

logger = logging.getLogger(__name__)

# Clinical thresholds
LOW_GLUCOSE_THRESHOLD = 70.0   # mg/dL
HIGH_GLUCOSE_THRESHOLD = 180.0  # mg/dL
CRITICAL_LOW_THRESHOLD = 54.0   # Severe hypoglycemia
CRITICAL_HIGH_THRESHOLD = 250.0 # Severe hyperglycemia


class RareEventDetector:
    """Detect rare events in glucose time series."""

    def __init__(self, low_threshold: float = LOW_GLUCOSE_THRESHOLD,
                 high_threshold: float = HIGH_GLUCOSE_THRESHOLD):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect_events(self, glucose_sequence: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect various types of rare events in glucose sequence.

        Returns:
            Dictionary with event types and their indices
        """
        events = {
            'hypoglycemia': [],      # Below low threshold
            'hyperglycemia': [],     # Above high threshold
            'rapid_drop': [],        # Rapid glucose decrease
            'rapid_rise': [],        # Rapid glucose increase
            'critical_low': [],      # Severe hypoglycemia
            'critical_high': [],     # Severe hyperglycemia
            'dawn_phenomenon': [],   # Early morning glucose rise
            'postprandial_spike': [] # Post-meal glucose spike
        }

        for i, glucose in enumerate(glucose_sequence):
            # Basic threshold events
            if glucose < self.low_threshold:
                events['hypoglycemia'].append(i)
            if glucose > self.high_threshold:
                events['hyperglycemia'].append(i)
            if glucose < CRITICAL_LOW_THRESHOLD:
                events['critical_low'].append(i)
            if glucose > CRITICAL_HIGH_THRESHOLD:
                events['critical_high'].append(i)

            # Rate of change events (need at least 2 points)
            if i > 0:
                rate = glucose_sequence[i] - glucose_sequence[i-1]

                # Rapid changes (>20 mg/dL per time step)
                if rate < -20:
                    events['rapid_drop'].append(i)
                elif rate > 20:
                    events['rapid_rise'].append(i)

                # Pattern-based events (need more context)
                if i >= 3:
                    # Dawn phenomenon: sustained rise in early hours
                    recent_trend = np.mean(np.diff(glucose_sequence[i-3:i+1]))
                    if recent_trend > 5 and glucose > 120:
                        events['dawn_phenomenon'].append(i)

                    # Postprandial spike: rapid rise followed by plateau
                    if (glucose_sequence[i-2] < glucose_sequence[i-1] < glucose and
                        glucose - glucose_sequence[i-2] > 30):
                        events['postprandial_spike'].append(i)

        return events

    def get_event_severity(self, glucose_value: float) -> str:
        """Get severity level of glucose event."""
        if glucose_value < CRITICAL_LOW_THRESHOLD:
            return 'critical_low'
        elif glucose_value < self.low_threshold:
            return 'mild_low'
        elif glucose_value > CRITICAL_HIGH_THRESHOLD:
            return 'critical_high'
        elif glucose_value > self.high_threshold:
            return 'mild_high'
        else:
            return 'normal'


class SMOTEGlucose:
    """
    SMOTE (Synthetic Minority Oversampling Technique) adapted for glucose sequences.
    Generates synthetic rare event sequences.
    """

    def __init__(self, k_neighbors: int = 5, random_state: Optional[int] = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def generate_synthetic_sequences(self, rare_sequences: np.ndarray,
                                   n_synthetic: int) -> np.ndarray:
        """
        Generate synthetic sequences using SMOTE-like approach.

        Args:
            rare_sequences: Array of rare event sequences (n_samples, seq_len, features)
            n_synthetic: Number of synthetic sequences to generate

        Returns:
            Synthetic sequences array
        """
        if len(rare_sequences) < 2:
            logger.warning("Not enough rare sequences for SMOTE generation")
            return rare_sequences

        # Flatten sequences for nearest neighbor search
        n_samples, seq_len, n_features = rare_sequences.shape
        flattened = rare_sequences.reshape(n_samples, -1)

        # Fit nearest neighbors
        nn_model = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(rare_sequences)))
        nn_model.fit(flattened)

        synthetic_sequences = []

        for _ in range(n_synthetic):
            # Randomly select a rare sequence
            idx = np.random.randint(0, len(rare_sequences))
            base_sequence = flattened[idx]

            # Find nearest neighbors
            distances, indices = nn_model.kneighbors([base_sequence])

            # Select a random neighbor (excluding self)
            neighbor_idx = np.random.choice(indices[0][1:])
            neighbor_sequence = flattened[neighbor_idx]

            # Generate synthetic sequence by interpolation
            alpha = np.random.random()
            synthetic_flat = base_sequence + alpha * (neighbor_sequence - base_sequence)

            # Reshape back to original dimensions
            synthetic_sequence = synthetic_flat.reshape(seq_len, n_features)

            # Apply physiological constraints
            synthetic_sequence = self._apply_constraints(synthetic_sequence)

            synthetic_sequences.append(synthetic_sequence)

        return np.array(synthetic_sequences)

    def _apply_constraints(self, sequence: np.ndarray) -> np.ndarray:
        """Apply physiological constraints to synthetic sequences."""
        # Clip glucose values to reasonable range
        sequence = np.clip(sequence, 40, 400)  # mg/dL

        # Smooth unrealistic fluctuations
        if sequence.shape[1] == 1:  # Single feature (glucose)
            glucose_values = sequence[:, 0]

            # Limit rate of change
            for i in range(1, len(glucose_values)):
                max_change = 50  # mg/dL per time step
                change = glucose_values[i] - glucose_values[i-1]

                if abs(change) > max_change:
                    glucose_values[i] = glucose_values[i-1] + np.sign(change) * max_change

            sequence[:, 0] = glucose_values

        return sequence


class RareEventAugmenter:
    """
    Main class for rare event augmentation in glucose prediction.
    Combines multiple augmentation strategies.
    """

    def __init__(self, augmentation_factor: float = 2.0,
                 methods: List[str] = None,
                 random_state: Optional[int] = None):
        self.augmentation_factor = augmentation_factor
        self.methods = methods or ['smote', 'noise_injection', 'temporal_shift', 'magnitude_scaling']
        self.random_state = random_state

        # Initialize components
        self.event_detector = RareEventDetector()
        self.smote = SMOTEGlucose(random_state=random_state)

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    def augment_rare_events(self, sequences: np.ndarray, targets: np.ndarray,
                           event_types: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment rare events in glucose sequences.

        Args:
            sequences: Input sequences (n_samples, seq_len, features)
            targets: Target values (n_samples, target_len)
            event_types: Types of events to augment

        Returns:
            Augmented sequences and targets
        """
        if event_types is None:
            event_types = ['hypoglycemia', 'hyperglycemia', 'rapid_drop', 'rapid_rise']

        logger.info(f"Augmenting rare events: {event_types}")

        # Identify rare event sequences
        rare_indices = self._identify_rare_sequences(sequences, targets, event_types)

        if len(rare_indices) == 0:
            logger.warning("No rare events found for augmentation")
            return sequences, targets

        logger.info(f"Found {len(rare_indices)} rare event sequences")

        # Extract rare sequences
        rare_sequences = sequences[rare_indices]
        rare_targets = targets[rare_indices]

        # Calculate number of synthetic samples needed
        n_synthetic = int(len(rare_sequences) * (self.augmentation_factor - 1))

        if n_synthetic <= 0:
            return sequences, targets

        # Generate synthetic sequences using selected methods
        synthetic_sequences = []
        synthetic_targets = []

        samples_per_method = max(1, n_synthetic // len(self.methods))

        for method in self.methods:
            method_samples = min(samples_per_method, n_synthetic - len(synthetic_sequences))

            if method_samples <= 0:
                break

            syn_seq, syn_tgt = self._apply_augmentation_method(
                method, rare_sequences, rare_targets, method_samples
            )

            synthetic_sequences.extend(syn_seq)
            synthetic_targets.extend(syn_tgt)

        # Combine original and synthetic data
        if synthetic_sequences:
            synthetic_sequences = np.array(synthetic_sequences)
            synthetic_targets = np.array(synthetic_targets)

            augmented_sequences = np.concatenate([sequences, synthetic_sequences], axis=0)
            augmented_targets = np.concatenate([targets, synthetic_targets], axis=0)

            logger.info(f"Generated {len(synthetic_sequences)} synthetic rare event sequences")

            return augmented_sequences, augmented_targets

        return sequences, targets

    def _identify_rare_sequences(self, sequences: np.ndarray, targets: np.ndarray,
                                event_types: List[str]) -> List[int]:
        """Identify sequences containing rare events."""
        rare_indices = []

        for i, (seq, target) in enumerate(zip(sequences, targets)):
            # Check sequence for rare events
            if seq.shape[-1] == 1:  # Single feature (glucose)
                glucose_seq = seq[:, 0]
                events = self.event_detector.detect_events(glucose_seq)

                # Check if any target events are rare
                for event_type in event_types:
                    if events.get(event_type, []):
                        rare_indices.append(i)
                        break

                # Also check targets for rare events
                if target.ndim == 1:  # Multi-step targets
                    target_events = self.event_detector.detect_events(target)
                    for event_type in event_types:
                        if target_events.get(event_type, []):
                            rare_indices.append(i)
                            break

        return list(set(rare_indices))  # Remove duplicates

    def _apply_augmentation_method(self, method: str, sequences: np.ndarray,
                                 targets: np.ndarray, n_samples: int) -> Tuple[List, List]:
        """Apply specific augmentation method."""
        synthetic_sequences = []
        synthetic_targets = []

        if method == 'smote':
            # SMOTE-based generation
            syn_seq = self.smote.generate_synthetic_sequences(sequences, n_samples)
            syn_tgt = self.smote.generate_synthetic_sequences(
                targets.reshape(len(targets), -1, 1), n_samples
            ).reshape(n_samples, -1)

            synthetic_sequences.extend(syn_seq)
            synthetic_targets.extend(syn_tgt)

        elif method == 'noise_injection':
            # Add controlled noise to rare sequences
            for _ in range(n_samples):
                idx = np.random.randint(0, len(sequences))

                # Add physiologically reasonable noise
                noise_std = 5.0  # mg/dL
                seq_noise = np.random.normal(0, noise_std, sequences[idx].shape)
                tgt_noise = np.random.normal(0, noise_std, targets[idx].shape)

                syn_seq = sequences[idx] + seq_noise
                syn_tgt = targets[idx] + tgt_noise

                # Apply constraints
                syn_seq = np.clip(syn_seq, 40, 400)
                syn_tgt = np.clip(syn_tgt, 40, 400)

                synthetic_sequences.append(syn_seq)
                synthetic_targets.append(syn_tgt)

        elif method == 'temporal_shift':
            # Temporal shifting with interpolation
            for _ in range(n_samples):
                idx = np.random.randint(0, len(sequences))

                # Random time shift
                shift = np.random.uniform(-0.2, 0.2)  # ±20% of sequence length

                syn_seq = self._apply_temporal_shift(sequences[idx], shift)
                syn_tgt = self._apply_temporal_shift(targets[idx].reshape(-1, 1), shift).flatten()

                synthetic_sequences.append(syn_seq)
                synthetic_targets.append(syn_tgt)

        elif method == 'magnitude_scaling':
            # Scale glucose values while preserving patterns
            for _ in range(n_samples):
                idx = np.random.randint(0, len(sequences))

                # Random scaling factor
                scale = np.random.uniform(0.9, 1.1)

                syn_seq = sequences[idx] * scale
                syn_tgt = targets[idx] * scale

                # Apply constraints
                syn_seq = np.clip(syn_seq, 40, 400)
                syn_tgt = np.clip(syn_tgt, 40, 400)

                synthetic_sequences.append(syn_seq)
                synthetic_targets.append(syn_tgt)

        return synthetic_sequences, synthetic_targets

    def _apply_temporal_shift(self, sequence: np.ndarray, shift: float) -> np.ndarray:
        """Apply temporal shift using interpolation."""
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)

        seq_len = len(sequence)

        # Create new time indices
        original_indices = np.arange(seq_len)
        shifted_indices = original_indices + shift * seq_len

        # Clip to valid range
        shifted_indices = np.clip(shifted_indices, 0, seq_len - 1)

        # Interpolate
        shifted_sequence = np.zeros_like(sequence)

        for feature_idx in range(sequence.shape[1]):
            interp_func = interp1d(original_indices, sequence[:, feature_idx],
                                 kind='linear', fill_value='extrapolate')
            shifted_sequence[:, feature_idx] = interp_func(shifted_indices)

        return shifted_sequence

    def get_augmentation_stats(self, original_sequences: np.ndarray,
                              augmented_sequences: np.ndarray) -> Dict:
        """Get statistics about the augmentation process."""
        original_count = len(original_sequences)
        augmented_count = len(augmented_sequences)
        synthetic_count = augmented_count - original_count

        # Count rare events before and after
        original_rare = 0
        augmented_rare = 0

        for seq in original_sequences:
            if seq.shape[-1] == 1:
                events = self.event_detector.detect_events(seq[:, 0])
                if any(events.values()):
                    original_rare += 1

        for seq in augmented_sequences:
            if seq.shape[-1] == 1:
                events = self.event_detector.detect_events(seq[:, 0])
                if any(events.values()):
                    augmented_rare += 1

        return {
            'original_sequences': original_count,
            'synthetic_sequences': synthetic_count,
            'total_sequences': augmented_count,
            'augmentation_factor': augmented_count / original_count,
            'original_rare_events': original_rare,
            'augmented_rare_events': augmented_rare,
            'rare_event_increase': (augmented_rare - original_rare) / max(original_rare, 1)
        }
