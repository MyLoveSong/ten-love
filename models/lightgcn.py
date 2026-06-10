#!/usr/bin/env python3
"""
LightGCN model for collaborative filtering.

LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Adapted for general recommender system baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class LightGCNLayer(nn.Module):
    """
    Single LightGCN layer with simplified graph convolution.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        adj_matrix: torch.sparse.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LightGCN layer.

        Args:
            user_embeddings: [num_users, latent_dim] User embeddings
            item_embeddings: [num_items, latent_dim] Item embeddings
            adj_matrix: [num_users + num_items, num_users + num_items] Adjacency matrix

        Returns:
            updated_user_embeddings: [num_users, latent_dim] Updated user embeddings
            updated_item_embeddings: [num_items, latent_dim] Updated item embeddings
        """
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # [num_users + num_items, latent_dim]

        # Graph convolution: E^(k) = A_hat * E^(k-1)
        # Using sparse matrix multiplication for efficiency
        updated_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)  # [num_users + num_items, latent_dim]

        # Split back into users and items
        num_users = user_embeddings.size(0)
        updated_user_embeddings = updated_embeddings[:num_users]
        updated_item_embeddings = updated_embeddings[num_users:]

        return updated_user_embeddings, updated_item_embeddings


class LightGCNEncoder(nn.Module):
    """
    LightGCN encoder with multiple graph convolution layers.

    Based on the paper: "LightGCN: Simplifying and Powering Graph Convolution
    Network for Recommendation" (He et al., SIGIR 2020)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.item_embeddings = nn.Embedding(num_items, latent_dim)

        # LightGCN layers
        self.layers = nn.ModuleList([
            LightGCNLayer(latent_dim) for _ in range(num_layers)
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def build_adjacency_matrix(
        self,
        user_item_interactions: torch.Tensor,
        device: torch.device
    ) -> torch.sparse.Tensor:
        """
        Build normalized adjacency matrix from user-item interactions.

        Args:
            user_item_interactions: [num_interactions, 2] User-item interaction pairs
            device: Target device for the matrix

        Returns:
            adj_matrix: [num_users + num_items, num_users + num_items] Normalized adjacency matrix
        """
        num_nodes = self.num_users + self.num_items

        # Build adjacency matrix indices and values
        row_indices = []
        col_indices = []
        values = []

        # Add user-item interactions
        users = user_item_interactions[:, 0]
        items = user_item_interactions[:, 1] + self.num_users  # Offset item indices

        # Add both directions: (user, item) and (item, user)
        row_indices.extend(users.tolist())
        col_indices.extend(items.tolist())
        values.extend([1.0] * len(users))

        row_indices.extend(items.tolist())
        col_indices.extend(users.tolist())
        values.extend([1.0] * len(items))

        # Add self-loops
        for i in range(num_nodes):
            row_indices.append(i)
            col_indices.append(i)
            values.append(1.0)

        # Convert to sparse tensor
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
        values = torch.tensor(values, dtype=torch.float, device=device)
        adj_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

        # Normalize adjacency matrix (LightGCN normalization)
        degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()  # [num_nodes]
        degrees_sqrt = torch.sqrt(degrees + 1e-8)  # Add small epsilon to avoid division by zero

        # Create diagonal normalization matrix
        norm_matrix = torch.diag(1.0 / degrees_sqrt)

        # Normalize: A_hat = D^(-1/2) * A * D^(-1/2)
        adj_matrix = torch.sparse.mm(torch.sparse.mm(norm_matrix, adj_matrix.to_dense()), norm_matrix)
        adj_matrix = adj_matrix.to_sparse()

        return adj_matrix

    def forward(
        self,
        user_item_interactions: torch.Tensor,
        return_layer_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through LightGCN encoder.

        Args:
            user_item_interactions: [num_interactions, 2] User-item interaction pairs
            return_layer_embeddings: Whether to return embeddings from each layer

        Returns:
            user_embeddings: [num_users, latent_dim] Final user embeddings
            item_embeddings: [num_items, latent_dim] Final item embeddings
            layer_embeddings: List of (user_emb, item_emb) tuples from each layer (optional)
        """
        device = user_item_interactions.device

        # Build adjacency matrix
        adj_matrix = self.build_adjacency_matrix(user_item_interactions, device)

        # Initial embeddings
        user_emb = self.user_embeddings.weight  # [num_users, latent_dim]
        item_emb = self.item_embeddings.weight  # [num_items, latent_dim]

        all_layer_embeddings = [] if return_layer_embeddings else None

        # Apply LightGCN layers
        for layer in self.layers:
            if return_layer_embeddings:
                all_layer_embeddings.append((user_emb, item_emb))

            # Apply dropout to embeddings
            user_emb_dropped = self.dropout(user_emb)
            item_emb_dropped = self.dropout(item_emb)

            # Graph convolution
            user_emb, item_emb = layer(user_emb_dropped, item_emb_dropped, adj_matrix)

        # Final embeddings (average of all layers + initial layer)
        if return_layer_embeddings and all_layer_embeddings:
            all_user_emb = [user_emb] + [emb[0] for emb in all_layer_embeddings]
            all_item_emb = [item_emb] + [emb[1] for emb in all_layer_embeddings]

            final_user_emb = torch.stack(all_user_emb, dim=0).mean(dim=0)
            final_item_emb = torch.stack(all_item_emb, dim=0).mean(dim=0)
        else:
            # Simple case: just use final layer
            final_user_emb = user_emb
            final_item_emb = item_emb

        return final_user_emb, final_item_emb, all_layer_embeddings


class LightGCNRecommender(nn.Module):
    """
    LightGCN-based recommender for collaborative filtering.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

        # LightGCN encoder
        self.encoder = LightGCNEncoder(
            num_users=num_users,
            num_items=num_items,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_item_interactions: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for recommendation.

        Args:
            user_ids: [batch_size] User IDs to score
            item_ids: [batch_size, num_candidates] Item IDs to score
            user_item_interactions: [num_interactions, 2] All user-item interactions
            return_embeddings: Whether to return embeddings

        Returns:
            outputs: Dictionary containing predictions and scores
        """
        batch_size, num_candidates = item_ids.size()

        # Get final embeddings
        user_embeddings, item_embeddings, _ = self.encoder(user_item_interactions)

        # Get embeddings for the specific users and items
        user_emb_batch = user_embeddings[user_ids]  # [batch_size, latent_dim]

        # Compute scores for all user-item pairs
        scores = []
        for i in range(num_candidates):
            item_emb = item_embeddings[item_ids[:, i]]  # [batch_size, latent_dim]
            score = (user_emb_batch * item_emb).sum(dim=1)  # [batch_size]
            scores.append(score)

        recommendation_scores = torch.stack(scores, dim=1)  # [batch_size, num_candidates]

        outputs = {
            'recommendation_scores': recommendation_scores,
            'top_k_items': torch.topk(recommendation_scores, k=min(10, num_candidates), dim=1).indices,
        }

        if return_embeddings:
            outputs.update({
                'user_embeddings': user_embeddings,
                'item_embeddings': item_embeddings,
                'user_emb_batch': user_emb_batch,
            })

        return outputs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LightGCNRecommender':
        """Create model from configuration."""
        return cls(
            num_users=config.get('num_users', 1000),
            num_items=config.get('num_items', 1000),
            latent_dim=config.get('latent_dim', 64),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
        )
