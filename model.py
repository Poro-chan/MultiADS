# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torch import Tensor, nn
import torch
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens

class MLPLayerWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super().__init__()
        if 'ViT' in model:
            self.fc = nn.ModuleList([MLPLayer(dim_in, dim_out) for i in range(k)])
        else:
            self.fc = nn.ModuleList([MLPLayer(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens

class MLPLayer(nn.Module) :
    def __init__(self, dim_in, dim_out, hidden_dim=None, dropout=0.0) :
        super().__init__()
        hidden_dim = hidden_dim or dim_in * 2
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out)
        )
    
    def forward(self, tokens) :
        return self.net(tokens)