"""
Merlin Two-Stage Recommendation Architecture - PyTorch Models
Includes:
- Matrix Factorization Baseline
- Two-Tower Dual-Encoder Model (Candidate Retrieval)
- Neural Collaborative Filtering (NCF / NeuMF)
- Deep & Cross Ranking Model (Candidate Scoring)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class MatrixFactorization(nn.Module):
    """Classic Matrix Factorization baseline with User and Item Embeddings and Biases."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embed(user_ids)
        i_emb = self.item_embed(item_ids)
        dot = (u_emb * i_emb).sum(dim=-1)
        u_b = self.user_bias(user_ids).squeeze(-1)
        i_b = self.item_bias(item_ids).squeeze(-1)
        logits = dot + u_b + i_b + self.global_bias
        return logits

class UserTower(nn.Module):
    """User Sub-Tower for Two-Tower Retrieval Model."""
    def __init__(self, num_users: int, embedding_dim: int = 32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim)
        self.gender_embed = nn.Embedding(2, 8)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + 8 + 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, user_ids: torch.Tensor, gender: torch.Tensor, age: torch.Tensor, user_activity: torch.Tensor) -> torch.Tensor:
        u = self.user_embed(user_ids)
        g = self.gender_embed(gender)
        cont = torch.stack([age, user_activity], dim=1)
        x = torch.cat([u, g, cont], dim=1)
        output = self.mlp(x)
        return F.normalize(output, p=2, dim=1)

class ItemTower(nn.Module):
    """Item Sub-Tower for Two-Tower Retrieval Model."""
    def __init__(self, num_items: int, num_categories: int = 10, embedding_dim: int = 32):
        super().__init__()
        self.item_embed = nn.Embedding(num_items + 1, embedding_dim)
        self.cat_embed = nn.Embedding(num_categories + 1, 16)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + 16 + 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, item_ids: torch.Tensor, category_ids: torch.Tensor, price: torch.Tensor, quality: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
        i = self.item_embed(item_ids)
        c = self.cat_embed(category_ids)
        cont = torch.stack([price, quality, pop], dim=1)
        x = torch.cat([i, c, cont], dim=1)
        output = self.mlp(x)
        return F.normalize(output, p=2, dim=1)

class TwoTowerModel(nn.Module):
    """Deep Two-Tower Dual-Encoder Model supporting In-Batch & Temperature Softmax Loss."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32, temperature: float = 0.07):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, 10, embedding_dim)
        self.temperature = temperature

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        u_emb = self.user_tower(batch['user_id'], batch['gender'], batch['age'], batch['log_user_count'])
        i_emb = self.item_tower(batch['item_id'], batch['category_id'], batch['price'], batch['quality_score'], batch['log_item_count'])
        return u_emb, i_emb

    def compute_inbatch_loss(self, u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        """In-batch Softmax Loss with Temperature."""
        sim_matrix = torch.matmul(u_emb, i_emb.T) / self.temperature
        labels = torch.arange(u_emb.size(0), device=u_emb.device)
        return F.cross_entropy(sim_matrix, labels)

class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering (NeuMF fusing Generalized Matrix Factorization & Deep MLP)."""
    def __init__(self, num_users: int, num_items: int, latent_dim_gmf: int = 32, latent_dim_mlp: int = 32):
        super().__init__()
        self.gmf_user_embed = nn.Embedding(num_users + 1, latent_dim_gmf)
        self.gmf_item_embed = nn.Embedding(num_items + 1, latent_dim_gmf)
        
        self.mlp_user_embed = nn.Embedding(num_users + 1, latent_dim_mlp)
        self.mlp_item_embed = nn.Embedding(num_items + 1, latent_dim_mlp)
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim_mlp * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.final_layer = nn.Linear(latent_dim_gmf + 16, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf_u = self.gmf_user_embed(user_ids)
        gmf_i = self.gmf_item_embed(item_ids)
        gmf_vector = gmf_u * gmf_i
        
        mlp_u = self.mlp_user_embed(user_ids)
        mlp_i = self.mlp_item_embed(item_ids)
        mlp_vector = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))
        
        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logits = self.final_layer(fusion).squeeze(-1)
        return logits

class CrossNetwork(nn.Module):
    """Explicit feature interaction cross network layer (DCN-v2 style)."""
    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, input_dim) * 0.01) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        x0_col = x0.unsqueeze(-1) # [B, D, 1]
        for l in range(self.num_layers):
            xl_col = xl.unsqueeze(-1)
            # x0 * (W * xl + b) + xl
            mat = torch.matmul(self.weights[l], xl_col) + self.biases[l]
            xl = x0 * mat.squeeze(-1) + xl
        return xl

class DeepRankingModel(nn.Module):
    """Stage 2 Ranking Model combining DCN Cross Network and Deep MLP for candidate fine-scoring."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embed = nn.Embedding(num_items + 1, embedding_dim)
        self.cat_embed = nn.Embedding(11, 16)
        
        input_dim = embedding_dim * 2 + 16 + 5 # embeddings + continuous features
        self.cross_net = CrossNetwork(input_dim=input_dim, num_layers=2)
        
        self.deep_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.head = nn.Linear(input_dim + 32, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        u_emb = self.user_embed(batch['user_id'])
        i_emb = self.item_embed(batch['item_id'])
        c_emb = self.cat_embed(batch['category_id'])
        
        cont = torch.stack([batch['age'], batch['gender'].float(), batch['price'], batch['quality_score'], batch['log_item_count']], dim=1)
        
        x0 = torch.cat([u_emb, i_emb, c_emb, cont], dim=1)
        
        cross_out = self.cross_net(x0)
        deep_out = self.deep_mlp(x0)
        
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.head(combined).squeeze(-1)
        return logits

if __name__ == '__main__':
    print("Recommendation Models PyTorch definitions initialized successfully.")
