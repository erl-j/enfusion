import torch
from torch import nn
from torch.nn import functional as F
import math

def expand_to_planes(input, shape):
    return input[..., None].repeat([1, 1, shape[2]])

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
    def __init__(self,in_channels) -> None:
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        # MLP with skip connection
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels+16, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
        )

        # Bidirectional GRU model with a 2 layer skip MLP before and after the GRU
        self.gru = torch.nn.GRU(
            input_size=in_channels,
            hidden_size=in_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        self.out_mlp=torch.nn.Sequential(
            torch.nn.Linear(in_channels*2, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels),
        )        

    def forward(self, x,t):        

        te = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        
        x = torch.cat([x, te], dim=1)

        x = x.permute(0, 2, 1)

        # Skip MLP
        x = self.mlp(x)

        # GRU
        x, _ = self.gru(x)

        # Skip MLP
        x = self.out_mlp(x)

        x = x.permute(0, 2, 1)

        return x



class RecurrentScore2(torch.nn.Module):
    def __init__(self,in_channels) -> None:
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)
        
        self.in_channels = in_channels

        self.hidden_size = in_channels + 16

        # MLP with skip connection
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels+16, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.pre_gru_act = torch.nn.ReLU()

        # Bidirectional GRU model with a 2 layer skip MLP before and after the GRU
        self.gru = torch.nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.post_gru_act = torch.nn.ReLU()
        
        self.out_mlp=torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.in_channels),
        )

        self.pre_out_act = torch.nn.ReLU()
        
    def forward(self, x,t):        

        te = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        
        x = torch.cat([x, te], dim=1)

        x = x.permute(0, 2, 1) 

        # Skip MLP
        x = self.mlp(x) + x

        x = self.pre_gru_act(x)

        # GRU
        gru_out, _ = self.gru(x)

        # fold the bidirectional GRU output and add
        gru_out = gru_out[:,:, :self.hidden_size] + gru_out[:, :,self.hidden_size:]

        x = self.post_gru_act(gru_out + x )

        # Skip MLP
        x = self.out_mlp(x) + x[:, :, :self.in_channels]

        x = self.pre_out_act(x)

        x = x.permute(0, 2, 1)

        return x