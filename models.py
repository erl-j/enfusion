import torch
from torch import nn
from torch.nn import functional as F
import math
import einops
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
    def __init__(self,n_in_channels,n_conditioning_channels,hidden_size=256, reduced_text_embedding_size=16) -> None:
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        self.hidden_size = hidden_size

        self.reduced_text_embedding_size=reduced_text_embedding_size

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

class Block(torch.nn.Module): 
    def __init__(self, n_in_channels, n_hidden_channels, n_out_channels):
        super().__init__()
          # MLP with skip connection
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_in_channels, n_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_channels, n_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_channels, n_hidden_channels),
            torch.nn.ReLU(),
        )

        self.gru = torch.nn.GRU(
            input_size=n_hidden_channels,
            hidden_size=n_hidden_channels,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        
        self.out_mlp=torch.nn.Sequential(
            torch.nn.Linear(n_hidden_channels*2, n_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_channels, n_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_channels, n_out_channels)
        )

    def forward(self, x):
        # Skip MLP
        x = self.mlp(x)
        # GRU
        x, _ = self.gru(x)
        # Skip MLP
        x = self.out_mlp(x)
        return x
    

class MultiPitchRecurrentScore(torch.nn.Module):
    def __init__(self,n_in_channels,n_conditioning_channels, n_pitches) -> None:
        super().__init__()
        # input is n_pitches concatenated in the time dimension
        # so we need to split it up 
        self.n_pitches = n_pitches

        self.timestep_embed = FourierFeatures(1, 16)
        self.hidden_size = 200
        self.reduced_text_embedding_size=16
        self.text_reducer = TextReducer(n_conditioning_channels, self.reduced_text_embedding_size)

        self.global_context_size = 64
        self.global_context_block = Block(n_in_channels + 1  + 16 + self.reduced_text_embedding_size, n_hidden_channels=self.global_context_size, n_out_channels=self.global_context_size)
        self.main_block = Block(n_in_channels +1+ 16 + self.reduced_text_embedding_size + self.global_context_size, self.hidden_size, n_in_channels)
       
        

    def forward(self, x,t,text_embedding=None):  
        # batch, channel, time
        # batch, pitch, channel, time
        batch, channel, time = x.shape

        pitch_signal = torch.linspace(0, 1, self.n_pitches).repeat_interleave(time//self.n_pitches).to(x.device)[None,None,:].repeat(batch,1,1)

        x = torch.cat([x, pitch_signal], dim=1)

        te = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        x = torch.cat([x, te], dim=1)

        if text_embedding is not None:
            reduced_text_embedding = self.text_reducer(text_embedding)
            reduced_text_embedding = reduced_text_embedding[:,:,None].repeat(1,1,x.shape[2])
            x = torch.cat([x, reduced_text_embedding], dim=1)

        # reshape with einops
        x = einops.rearrange(x, 'b c (p t) -> (b p) c t', p=self.n_pitches)
       
        x = einops.rearrange(x, '(b p) c t -> (b p) t c', p=self.n_pitches)
        # global context
        global_context = self.global_context_block(x)

        global_context = einops.rearrange(global_context, '(b p) t c -> b p t c', p=self.n_pitches)

        global_context = torch.sum(global_context, dim=1, keepdim=True).repeat(1,self.n_pitches,1,1) 

        global_context = einops.rearrange(global_context, 'b p t c -> (b p) t c', p=self.n_pitches)

        x = torch.cat([x, global_context], dim=2)

        x = self.main_block(x)
        x = einops.rearrange(x, '(b p) t c -> (b p) c t', p=self.n_pitches)

        x = einops.rearrange(x, '(b p) c t -> b c (p t)',p=self.n_pitches)
        return x