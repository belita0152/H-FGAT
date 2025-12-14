"""
Automatically generate Meso-state graphs by DiffPool
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGATConv, DenseSAGEConv, dense_diff_pool

class DiffPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes, n_clusters=5):
        super(DiffPool, self).__init__()

        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.heads = 2  # GAT 멀티 헤드 개수

        # 1. Embedding GNN for extracting node features (Z)
        self.gat1_embed = DenseGATConv(in_channels, hidden_channels, heads=self.heads, dropout=0.5)
        self.gat2_embed = DenseGATConv(hidden_channels * self.heads, hidden_channels, heads=1, concat=False, dropout=0.5)
        self.bn1_embed = nn.BatchNorm1d(n_nodes)
        self.bn2_embed = nn.BatchNorm1d(n_nodes)

        # 2. Pooling GNN for learning assignment matrix (S)
        self.gnn1_pool = DenseGCNConv(in_channels, hidden_channels)
        self.gnn2_pool = DenseGCNConv(hidden_channels, n_clusters)
        self.bn1_pool = nn.BatchNorm1d(n_nodes)
        self.bn2_pool = nn.BatchNorm1d(n_nodes)

        # 3. GNN for coarsened graph (Meso-state)
        self.gat1_coarse = DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, dropout=0.5)
        self.gat2_coarse = DenseGATConv(hidden_channels * self.heads, hidden_channels, heads=1, concat=False, dropout=0.5)
        # self.sage1_coarse = DenseSAGEConv(hidden_channels, hidden_channels)
        # self.sage2_coarse = DenseSAGEConv(hidden_channels, hidden_channels)

        self.bn1_coarse = nn.BatchNorm1d(n_clusters)
        self.bn2_coarse = nn.BatchNorm1d(n_clusters)

        # 4. Final classifier
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dense, adj_dense):
        # Step 1: Embedding & Assignment 계산
        # Embedding Z
        z = self.gat1_embed(x_dense, adj_dense)
        z = self.bn1_embed(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = F.relu(self.gat2_embed(z, adj_dense))  # [batch, 30, hidden]
        z = self.bn2_embed(z)
        z = F.relu(z)
        z_local = F.dropout(z, p=0.5, training=self.training)

        # Assignment Matrix S
        s = self.gnn1_pool(x_dense, adj_dense)
        s = self.bn1_pool(s)
        s = F.relu(s)

        s = self.gnn2_pool(s, adj_dense)
        s = self.bn2_pool(s)
        s = F.softmax(s, dim=-1)  # [batch, 30, n_clusters]

        # Step 2: DiffPool (Meso-State 생성)
        x_coarse, adj_coarse, link_loss, ent_loss = dense_diff_pool(z, adj_dense, s)

        # Step 3: Coarsened Graph 학습
        z_meso = self.gat1_coarse(x_coarse, adj_coarse)
        # z_meso = self.sage1_coarse(x_coarse, adj_coarse)
        z_meso = self.bn1_coarse(z_meso)
        z_meso = F.relu(z_meso)
        z_meso = F.dropout(z_meso, p=0.5, training=self.training)

        z_meso = self.gat2_coarse(z_meso, adj_coarse)
        # z_meso = self.sage2_coarse(z_meso, adj_coarse)
        z_meso = self.bn2_coarse(z_meso)
        z_meso = F.relu(z_meso)

        # Step 4: Classification
        out_meso = torch.mean(z_meso, dim=1)
        out_meso = F.relu(self.fc1(out_meso))
        out_meso = F.dropout(out_meso, p=0.5, training=self.training)
        out_meso = self.fc2(out_meso)  # [batch, 2]

        out_local = torch.mean(z_local, dim=1)
        out_local = self.fc2(out_local)

        # return out, link_loss, ent_loss
        return z_local, z_meso, s, out_local, out_meso, link_loss, ent_loss
