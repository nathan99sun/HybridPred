import torch
import numpy as np
import torch.nn as nn


class TransformerEncoderRegression(torch.nn.Module):

    def __init__(self, num_heads = 1, num_layers = 2, d_model = 1, cycles = 49):
        super(TransformerEncoderRegression, self).__init__()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=d_model, nhead=num_heads,
                                        dim_feedforward=int(d_model), dropout=0)
        
        self.transformer = nn.TransformerEncoder(
                                encoder_layer=self.transformer_encoder_layer,
                                num_layers=num_layers)
        self.linear = nn.Linear(d_model, 1)
        self.initialize_parameters()

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.transformer(x)
        x = torch.sum(x, dim = 0)
        x = self.linear(x)
        return x