import torch
from torch import nn
from torch.nn import functional as F
import math

def expand_to_planes(input, shape):
    return input[..., None].repeat([1, 1, shape[2]])


class TextReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)
      

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class RecurrentScore(torch.nn.Module):
    def __init__(self,n_in_channels,n_conditioning_channels) -> None:
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        self.hidden_size = 256

        self.reduced_text_embedding_size=16


        self.text_reducer = TextReducer(n_conditioning_channels, self.reduced_text_embedding_size)

        # MLP with skip connection
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_in_channels+16+self.reduced_text_embedding_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        self.gru = torch.nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        
        self.out_mlp=torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size*2, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, n_in_channels),
        )        

    def forward(self, x,t,text_embedding=None):  

        te = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        
        x = torch.cat([x, te], dim=1)

        if text_embedding is not None:
            reduced_text_embedding = self.text_reducer(text_embedding)
            reduced_text_embedding = reduced_text_embedding[:,:,None].repeat(1,1,x.shape[2])
            x = torch.cat([x, reduced_text_embedding], dim=1)

        x = x.permute(0, 2, 1)

        # Skip MLP
        x = self.mlp(x)

        # GRU
        x, _ = self.gru(x)

        # Skip MLP
        x = self.out_mlp(x)

        x = x.permute(0, 2, 1)

        return x