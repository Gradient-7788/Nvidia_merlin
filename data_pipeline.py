"""
Merlin Recommendation Engine - Data & Feature Pipeline
Simulates scale User Events and NVTabular-style Feature Engineering.
"""

import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, Tuple, List

class SyntheticRecSysDataGenerator:
    """Generates synthetic User Events with long-tail power law dynamics, cold-start entities, and rich metadata."""
    
    def __init__(self, num_users: int = 5000, num_items: int = 10000, num_interactions: int = 250000, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Item features with power-law popularity
        item_categories = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=self.num_items, p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.04, 0.02])
        item_prices = np.random.exponential(scale=30.0, size=self.num_items) + 4.99
        item_quality = np.random.uniform(1.0, 5.0, size=self.num_items)
        
        # Power-law popularity alpha
        alpha = 1.2
        item_weights = (np.arange(1, self.num_items + 1) ** (-alpha))
        item_weights /= item_weights.sum()
        
        items_df = pd.DataFrame({
            'item_id': np.arange(self.num_items),
            'category_id': item_categories,
            'price': item_prices,
            'quality_score': item_quality,
            'pop_weight': item_weights
        })
        
        # User features
        user_age = np.random.randint(18, 65, size=self.num_users)
        user_gender = np.random.choice([0, 1], size=self.num_users) # 0: F, 1: M
        user_activity_level = np.random.choice(['low', 'medium', 'high'], size=self.num_users, p=[0.5, 0.35, 0.15])
        
        users_df = pd.DataFrame({
            'user_id': np.arange(self.num_users),
            'age': user_age,
            'gender': user_gender,
            'activity_level': user_activity_level
        })
        
        # Generate interactions (User Events)
        user_ids = np.random.randint(0, self.num_users, size=self.num_interactions)
        item_ids = np.random.choice(np.arange(self.num_items), size=self.num_interactions, p=item_weights)
        
        # Signal probabilities based on affinity matching + item quality + noise
        affinities = (user_ids % 10 == item_categories[item_ids]).astype(float) * 1.5
        qualities = item_quality[item_ids] / 5.0
        logits = affinities + qualities + np.random.normal(0, 0.8, size=self.num_interactions)
        probs = 1 / (1 + np.exp(-logits))
        
        events = (probs > 0.45).astype(int)
        timestamps = np.sort(np.random.randint(1600000000, 1700000000, size=self.num_interactions))
        
        events_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'label': events,
            'timestamp': timestamps
        })
        
        # Inject Cold-Start Items and Users (entities with zero interaction in history)
        cold_item_ids = np.arange(int(self.num_items * 0.95), self.num_items)
        cold_user_ids = np.arange(int(self.num_users * 0.95), self.num_users)
        
        events_df = events_df[~events_df['item_id'].isin(cold_item_ids)]
        events_df = events_df[~events_df['user_id'].isin(cold_user_ids)]
        
        return events_df.reset_index(drop=True), users_df, items_df

class MerlinFeaturePipeline:
    """NVTabular-inspired feature pipeline for categorical hashing, continuous normalization, and user interaction aggregations."""
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.user_stats = {}
        self.item_stats = {}
        self.category_map = {}
        
    def fit_transform(self, events_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        # Merge event metadata
        full_df = events_df.merge(users_df, on='user_id', how='left').merge(items_df, on='item_id', how='left')
        
        # Item popularity feature
        pop_counts = full_df['item_id'].value_counts().to_dict()
        full_df['item_count'] = full_df['item_id'].map(pop_counts).fillna(0)
        full_df['log_item_count'] = np.log1p(full_df['item_count'])
        
        # User activity count
        user_counts = full_df['user_id'].value_counts().to_dict()
        full_df['user_count'] = full_df['user_id'].map(user_counts).fillna(0)
        full_df['log_user_count'] = np.log1p(full_df['user_count'])
        
        # Continuous normalization (MinMax)
        full_df['norm_price'] = (full_df['price'] - full_df['price'].min()) / (full_df['price'].max() - full_df['price'].min() + 1e-6)
        full_df['norm_age'] = (full_df['age'] - full_df['age'].min()) / (full_df['age'].max() - full_df['age'].min() + 1e-6)
        
        # Convert to PyTorch Tensors
        tensors = {
            'user_id': torch.tensor(full_df['user_id'].values, dtype=torch.long),
            'item_id': torch.tensor(full_df['item_id'].values, dtype=torch.long),
            'category_id': torch.tensor(full_df['category_id'].values, dtype=torch.long),
            'age': torch.tensor(full_df['norm_age'].values, dtype=torch.float32),
            'gender': torch.tensor(full_df['gender'].values, dtype=torch.long),
            'price': torch.tensor(full_df['norm_price'].values, dtype=torch.float32),
            'quality_score': torch.tensor(full_df['quality_score'].values / 5.0, dtype=torch.float32),
            'log_item_count': torch.tensor(full_df['log_item_count'].values, dtype=torch.float32),
            'log_user_count': torch.tensor(full_df['log_user_count'].values, dtype=torch.float32),
            'label': torch.tensor(full_df['label'].values, dtype=torch.float32)
        }
        
        self.pop_counts = pop_counts
        return tensors, full_df

if __name__ == '__main__':
    generator = SyntheticRecSysDataGenerator()
    events_df, users_df, items_df = generator.generate()
    pipeline = MerlinFeaturePipeline()
    tensors, full_df = pipeline.fit_transform(events_df, users_df, items_df)
    print(f"Generated {len(events_df)} user events across {len(users_df)} users and {len(items_df)} items.")
    print("Tensor dictionary ready for Candidate Generation & Ranking stages!")
