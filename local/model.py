"""
This module functions exactly in the same way as the Embedding GNN of DiffPool in the Meso state.

Therefore, it is not used during the actual training process.

This module exists only to evaluate the training process of Local state.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

        self.dropout_rate = dropout_rate

    def forward(self, batch):
        x, edge_index, edge_weight, index = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)

        x = global_mean_pool(x, index)

        x = self.fc(x)

        return x
