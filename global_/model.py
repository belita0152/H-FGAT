import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

from meso.model import DiffPool


class GlobalFusion(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GlobalFusion, self).__init__()

        # Global Feature Aggregation Block
        # 1. Feature Fusion Layer: Local + Meso_Upsampled -> Combined
        self.fusion_proj = nn.Linear(hidden_channels * 2, hidden_channels)

        # 2. Global Refinement
        self.global_transformer = TransformerConv(hidden_channels, hidden_channels, heads=4, dropout=0.5, edge_dim=1)
        self.bn_global = nn.BatchNorm1d(hidden_channels * 4)

        # 3. Classifier
        self.fc1 = nn.Linear(hidden_channels * 4, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, z_local, z_meso, s, edge_index, edge_attr, batch, mask):
        # Upsampling (Meso -> Local scale)
        z_meso_up = torch.matmul(s, z_meso)  # (batch, 30, h)

        # Fusion
        z_fused = torch.cat([z_local, z_meso_up], dim=-1)  # (batch, 30, 2h)
        z_fused = self.fusion_proj(z_fused)  # (batch, 30, h)
        z_fused = F.relu(z_fused)

        # Global Refinement
        z_fused = z_fused[mask]  # (batch * 30, h)

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        out_global = self.global_transformer(z_fused, edge_index, edge_attr=edge_attr)

        # Classification
        out = global_mean_pool(out_global, batch)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)

        return out


# 전체 모델 조립
class HierarchicalModel(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32, out_channels=2, n_nodes=30):
        super(HierarchicalModel, self).__init__()

        # (1) Local + Meso: Clustering & Intermediate Classification
        self.meso_net = DiffPool(in_channels, hidden_channels, out_channels, n_nodes)

        # (2) Global: Fusion & Final Classification
        self.global_net = GlobalFusion(hidden_channels, out_channels)


    def forward(self, batch):
        x, edge_index, edge_attr, index = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        x_dense, mask = to_dense_batch(x, index)  # x_dense: [batch, 30, feature], mask: [batch, 30]
        adj_dense = to_dense_adj(edge_index, index, edge_attr)  # adj_dense: [batch, 30, 30]

        threshold = 0.2
        adj_dense = adj_dense * (adj_dense > threshold).float()

        # Local + Meso Flow
        z_local, z_meso, s, out_local, out_meso, link_loss, ent_loss = self.meso_net(x_dense, adj_dense)

        # Global Flow
        out_global = self.global_net(z_local, z_meso, s, edge_index, edge_attr, batch.batch, mask)

        return out_local, out_meso, out_global, link_loss, ent_loss
