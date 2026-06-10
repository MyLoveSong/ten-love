"""
Base model implementations for glucose prediction.
Provides foundational model classes following SOLID principles.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from .interfaces import ModelInterface
from .exceptions import ModelTrainingError, ModelNotFoundError

logger = logging.getLogger(__name__)


class BaseGlucosePredictor(nn.Module, ModelInterface):
    """Base class for glucose prediction models."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 6,
                 dropout: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_model()
        self.to(self.device)

    def _build_model(self) -> None:
        """Build the model architecture. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_model method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def train_model(self, train_data: torch.Tensor, train_targets: torch.Tensor,
                   val_data: torch.Tensor, val_targets: torch.Tensor,
                   epochs: int = 100, learning_rate: float = 0.001,
                   batch_size: int = 32, patience: int = 10,
                   scheduler_config: Optional[Dict[str, Any]] = None,
                   early_stopping_config: Optional[Dict[str, Any]] = None,
                   grad_clip: float = 1.0) -> Dict[str, Any]:
        """Train the model with early stopping."""
        try:
            scheduler_config = scheduler_config or {}
            early_stopping_config = early_stopping_config or {}

            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=scheduler_config.get('weight_decay', 1e-5)
            )
            train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
            val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)

            if len(train_dataset) == 0 or len(val_dataset) == 0:
                raise ModelTrainingError("Empty training or validation dataset.")

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

            steps_per_epoch = max(1, len(train_loader))

            scheduler, scheduler_step_mode = self._build_scheduler(
                optimizer=optimizer,
                scheduler_config=scheduler_config,
                epochs=epochs,
                patience=patience,
                steps_per_epoch=steps_per_epoch
            )
            criterion = nn.MSELoss()

            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_mae': [],
                'val_mae': [],
                'learning_rate': []
            }

            best_val_loss = float('inf')
            patience_counter = 0
            cooldown_counter = 0
            best_state = None
            best_epoch = 0

            min_delta = early_stopping_config.get('min_delta', 0.0)
            burn_in_epochs = early_stopping_config.get('burn_in_epochs', 0)
            cooldown = max(0, early_stopping_config.get('cooldown', 0))
            early_patience = early_stopping_config.get('patience', patience)

            for epoch in range(epochs):
                # Training phase
                self.train()
                train_loss = 0.0
                train_mae = 0.0

                for batch_data, batch_targets in train_loader:
                    batch_data = batch_data.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = self(batch_data)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
                    optimizer.step()

                    if scheduler and scheduler_step_mode == 'batch':
                        scheduler.step()

                    train_loss += loss.item()
                    train_mae += torch.mean(torch.abs(outputs - batch_targets)).item()

                # Validation phase
                self.eval()
                val_loss = 0.0
                val_mae = 0.0

                with torch.no_grad():
                    for batch_data, batch_targets in val_loader:
                        batch_data = batch_data.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        outputs = self(batch_data)
                        loss = criterion(outputs, batch_targets)

                        val_loss += loss.item()
                        val_mae += torch.mean(torch.abs(outputs - batch_targets)).item()

                # Calculate averages
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                train_mae /= len(train_loader)
                val_mae /= len(val_loader)

                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_mae'].append(train_mae)
                history['val_mae'].append(val_mae)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])

                # Learning rate scheduling
                if scheduler:
                    if scheduler_step_mode == 'epoch':
                        scheduler.step()
                    elif scheduler_step_mode == 'plateau':
                        scheduler.step(val_loss)

                improved_enough = val_loss < (best_val_loss - min_delta)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())
                    best_epoch = epoch

                # Early stopping
                if epoch >= burn_in_epochs:
                    if improved_enough:
                        patience_counter = 0
                        cooldown_counter = 0
                    else:
                        if cooldown_counter < cooldown:
                            cooldown_counter += 1
                        else:
                            patience_counter += 1

                if patience_counter >= early_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Train MAE: {train_mae:.4f}, "
                              f"Val MAE: {val_mae:.4f}")

            # Load best model state
            if best_state is not None:
                self.load_state_dict(best_state)

            return {
                'history': history,
                'best_val_loss': best_val_loss,
                'total_epochs': epoch + 1,
                'best_epoch': best_epoch + 1
            }

        except Exception as e:
            raise ModelTrainingError(f"Training failed: {str(e)}")

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_config: Dict[str, Any],
        epochs: int,
        patience: int,
        steps_per_epoch: int
    ) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], str]:
        """Create learning rate scheduler based on configuration."""
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
        scheduler_type = scheduler_type.lower()

        if scheduler_type == 'one_cycle':
            total_steps = epochs * steps_per_epoch
            if total_steps == 0:
                return None, 'none'
            max_lr = scheduler_config.get('max_lr', optimizer.param_groups[0]['lr'] * scheduler_config.get('max_lr_scale', 5.0))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=scheduler_config.get('warmup_ratio', 0.1),
                anneal_strategy=scheduler_config.get('anneal_strategy', 'cos'),
                div_factor=scheduler_config.get('div_factor', 25.0),
                final_div_factor=scheduler_config.get('final_div_factor', 1000.0),
                three_phase=scheduler_config.get('three_phase', False)
            )
            return scheduler, 'batch'

        if scheduler_type == 'cosine_warmup':
            warmup_epochs = min(
                scheduler_config.get('warmup_epochs', max(1, epochs // 10)),
                max(0, epochs - 1)
            )
            eta_min = scheduler_config.get('min_lr', 1e-5)
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=scheduler_config.get('warmup_start_factor', 0.1),
                    total_iters=warmup_epochs
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs - warmup_epochs),
                    eta_min=eta_min
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs),
                    eta_min=eta_min
                )
            return scheduler, 'epoch'

        # Default ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', max(1, patience // 2)),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
        return scheduler, 'plateau'

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions on input data."""
        self.eval()
        with torch.no_grad():
            data = data.to(self.device)
            return self(data)

    def evaluate(self, test_data: torch.Tensor, test_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance."""
        self.eval()
        with torch.no_grad():
            test_data = test_data.to(self.device)
            test_targets = test_targets.to(self.device)

            predictions = self(test_data)

            mse = torch.mean((predictions - test_targets) ** 2).item()
            mae = torch.mean(torch.abs(predictions - test_targets)).item()
            rmse = np.sqrt(mse)

            # R² score
            ss_res = torch.sum((test_targets - predictions) ** 2)
            ss_tot = torch.sum((test_targets - torch.mean(test_targets)) ** 2)
            r2 = (1 - ss_res / ss_tot).item()

            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

    def save_model(self, path: str) -> None:
        """Save model to specified path."""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'output_dim': self.output_dim,
                    'dropout': self.dropout
                }
            }, save_path)

            logger.info(f"Model saved to {save_path}")

        except Exception as e:
            raise ModelNotFoundError(f"Failed to save model: {str(e)}")

    def load_model(self, path: str) -> None:
        """Load model from specified path."""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise ModelNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(load_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])

            logger.info(f"Model loaded from {load_path}")

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model: {str(e)}")


class LSTMGlucosePredictor(BaseGlucosePredictor):
    """LSTM-based glucose prediction model."""

    def _build_model(self) -> None:
        """Build LSTM model architecture."""
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim, num_layers=2,
            batch_first=True, dropout=self.dropout if self.hidden_dim > 1 else 0
        )
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Multi-step prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            ) for _ in range(self.output_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM model."""
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        # Use last time step
        features = attn_out[:, -1, :]
        features = self.dropout_layer(features)

        # Multi-step predictions
        predictions = []
        for head in self.prediction_heads:
            pred = head(features)
            predictions.append(pred)

        return torch.cat(predictions, dim=1)


class GluFormerPredictor(BaseGlucosePredictor):
    """
    GluFormer: LSTM+GRU fusion model with cross-attention mechanism.
    Implements the architecture described in the research proposal.
    """

    def _build_model(self) -> None:
        """Build GluFormer architecture with LSTM+GRU+Cross-Attention."""
        # LSTM branch for long-term dependencies
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim, num_layers=2,
            batch_first=True, dropout=self.dropout if self.hidden_dim > 1 else 0
        )

        # GRU branch for short-term dynamics
        self.gru = nn.GRU(
            self.input_dim, self.hidden_dim, num_layers=2,
            batch_first=True, dropout=self.dropout if self.hidden_dim > 1 else 0
        )

        # Self-attention for each branch
        self.lstm_self_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )
        self.gru_self_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )

        # Cross-attention for LSTM-GRU fusion
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )

        # Layer normalization
        self.lstm_norm = nn.LayerNorm(self.hidden_dim)
        self.gru_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # Dropout layers
        self.dropout_layer = nn.Dropout(self.dropout)

        # Fusion gate for combining LSTM and GRU features
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        # Multi-step prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            ) for _ in range(self.output_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GluFormer architecture."""
        # LSTM branch processing
        lstm_out, _ = self.lstm(x)
        lstm_attn, _ = self.lstm_self_attention(lstm_out, lstm_out, lstm_out)
        lstm_features = self.lstm_norm(lstm_attn + lstm_out)

        # GRU branch processing
        gru_out, _ = self.gru(x)
        gru_attn, _ = self.gru_self_attention(gru_out, gru_out, gru_out)
        gru_features = self.gru_norm(gru_attn + gru_out)

        # Cross-attention fusion: LSTM queries GRU
        cross_attn, _ = self.cross_attention(
            query=lstm_features,
            key=gru_features,
            value=gru_features
        )

        # Fusion with gating mechanism
        lstm_final = lstm_features[:, -1, :]  # Last time step
        gru_final = gru_features[:, -1, :]    # Last time step
        cross_final = cross_attn[:, -1, :]    # Last time step

        # Combine features
        combined = torch.cat([lstm_final, cross_final], dim=1)
        gate = self.fusion_gate(combined)

        # Gated fusion
        fused_features = gate * lstm_final + (1 - gate) * cross_final
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.dropout_layer(fused_features)

        # Multi-step predictions
        predictions = []
        for head in self.prediction_heads:
            pred = head(fused_features)
            predictions.append(pred)

        return torch.cat(predictions, dim=1)


class WaveletGluFormerPredictor(BaseGlucosePredictor):
    """
    Enhanced GluFormer with wavelet multi-scale features.
    Integrates wavelet tokens with LSTM+GRU+Cross-Attention.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 6,
                 dropout: float = 0.1, wavelet: str = 'db4', wavelet_levels: int = 3,
                 device: Optional[torch.device] = None):
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        super().__init__(input_dim, hidden_dim, output_dim, dropout, device)

    def _build_model(self) -> None:
        """Build Wavelet-enhanced GluFormer architecture."""
        # Import here to avoid circular dependency
        from data_processing.wavelet_features import WaveletTokenizer

        # Wavelet tokenizer
        self.wavelet_tokenizer = WaveletTokenizer(
            input_dim=self.input_dim,
            token_dim=self.hidden_dim,
            wavelet=self.wavelet,
            levels=self.wavelet_levels,
            device=self.device
        )

        # LSTM branch for sequence tokens
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, num_layers=2,
            batch_first=True, dropout=self.dropout if self.hidden_dim > 1 else 0
        )

        # GRU branch for sequence tokens
        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, num_layers=2,
            batch_first=True, dropout=self.dropout if self.hidden_dim > 1 else 0
        )

        # Wavelet feature processor
        self.wavelet_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 2,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )

        # Self-attention for each branch
        self.lstm_self_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )
        self.gru_self_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )

        # Cross-attention between branches
        self.lstm_gru_cross_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )

        # Cross-attention with wavelet features
        self.wavelet_cross_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=self.dropout, batch_first=True
        )

        # Layer normalization
        self.lstm_norm = nn.LayerNorm(self.hidden_dim)
        self.gru_norm = nn.LayerNorm(self.hidden_dim)
        self.wavelet_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # Multi-scale fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.Sigmoid()
        )

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        # Multi-step prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            ) for _ in range(self.output_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Wavelet-enhanced GluFormer."""
        # Tokenize input with wavelet features
        sequence_tokens, wavelet_tokens = self.wavelet_tokenizer(x)

        # Process sequence tokens through LSTM and GRU
        lstm_out, _ = self.lstm(sequence_tokens)
        lstm_attn, _ = self.lstm_self_attention(lstm_out, lstm_out, lstm_out)
        lstm_features = self.lstm_norm(lstm_attn + lstm_out)

        gru_out, _ = self.gru(sequence_tokens)
        gru_attn, _ = self.gru_self_attention(gru_out, gru_out, gru_out)
        gru_features = self.gru_norm(gru_attn + gru_out)

        # Process wavelet tokens
        wavelet_features = self.wavelet_processor(wavelet_tokens)
        wavelet_features = self.wavelet_norm(wavelet_features)

        # Cross-attention between LSTM and GRU
        lstm_gru_cross, _ = self.lstm_gru_cross_attention(
            query=lstm_features,
            key=gru_features,
            value=gru_features
        )

        # Cross-attention with wavelet features
        wavelet_cross, _ = self.wavelet_cross_attention(
            query=lstm_gru_cross,
            key=wavelet_features,
            value=wavelet_features
        )

        # Extract final features (last time step for sequences, mean for wavelets)
        lstm_final = lstm_features[:, -1, :]
        gru_final = gru_features[:, -1, :]
        wavelet_final = torch.mean(wavelet_cross, dim=1)  # Global average pooling

        # Multi-scale fusion
        combined = torch.cat([lstm_final, gru_final, wavelet_final], dim=1)
        fusion_weights = self.fusion_gate(combined)

        # Weighted combination
        fused_features = (
            fusion_weights * lstm_final +
            (1 - fusion_weights) * gru_final +
            fusion_weights * wavelet_final
        ) / 2

        fused_features = self.fusion_norm(fused_features)
        fused_features = self.dropout_layer(fused_features)

        # Multi-step predictions
        predictions = []
        for head in self.prediction_heads:
            pred = head(fused_features)
            predictions.append(pred)

        return torch.cat(predictions, dim=1)


class TransformerGlucosePredictor(BaseGlucosePredictor):
    """Transformer-based glucose prediction model."""

    def _build_model(self) -> None:
        """Build Transformer model architecture."""
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Transformer model."""
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)

        # Transformer processing
        x = self.transformer(x)

        # Use last time step for prediction
        x = x[:, -1, :]

        # Output projection
        return self.output_projection(x)
