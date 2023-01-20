# %%

import os
import noisereduce as nr
from scipy.io import wavfile
# GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import sys

#@title Imports and definitions
from prefigure.prefigure import get_all_args
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
#from google.colab import files

import os, signal, sys
import gc


import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
from einops import rearrange
import einops

import torchaudio
from models import RecurrentScore, MultiPitchRecurrentScore
import numpy as np

import random
import matplotlib.pyplot as plt
import IPython.display as ipd
from audio_diffusion_utils import Stereo, PadCrop
from glob import glob

from sampling import sample, resample, reverse_sample

from export_sfz import export_sfz
from datetime import datetime


from encodec_processor import EncodecProcessor
from text_embedder import TextEmbedder



device ="cuda" if torch.cuda.is_available() else "cpu"



import matplotlib.pyplot as plt
import IPython.display as ipd

def plot_and_hear(audio, sr):
    display(ipd.Audio(audio.cpu().clamp(-1, 1), rate=sr))
    plt.plot(audio.cpu().t().numpy())
  
def load_to_device(path, sr):
    audio, file_sr = torchaudio.load(path)
    if sr != file_sr:
      audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
    audio = audio.to(device)
    return audio


#@title Args
sample_size = 65536 
sample_rate = 48000   
latent_dim = 0             

class Object(object):
    pass

args = Object()
args.sample_size = sample_size
args.sample_rate = sample_rate
args.latent_dim = latent_dim

sampler_type = "v-iplms" #@param ["v-iplms", "k-heun", "k-dpmpp_2s_ancestral", "k-lms", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive"]

text_embedder = TextEmbedder()

# %% [markdown]
# Select the model you want to sample from
# ---
# Model name | Description |
# --- | --- |
# multipitch_parallel | multi-pitch, parallel
# multipitch_sequential | multi-pitch, sequential
# single_pitch | single pitch
# 

#%%
models_metadata = {
  "multipitch_parallel": {"checkpoint_path":"demo_assets/multiplenotes/parallel/epoch=4210-step=400000.ckpt", "clip_s":5, "n_pitches":9},
  "multipitch_sequential": {"checkpoint_path":"demo_assets/multiplenotes/sequential/epoch=2105-step=200000.ckpt", "clip_s":5, "n_pitches":9,"hidden_size":256,"reduced_text_embedding_size":16},
  "single_pitch": {"checkpoint_path":"demo_assets/single_note/epoch=6363-step=70000.ckpt", "clip_s":5, "n_pitches":1,"hidden_size":256, "reduced_text_embedding_size":16},
  "single_pitch_large":{"checkpoint_path":"demo_assets/single_note_large/epoch=4999-step=110000.ckpt", "clip_s":5, "n_pitches":1, "hidden_size":512, "reduced_text_embedding_size":32},
}

MODEL = "single_pitch_large"

model_metadata = models_metadata[MODEL]

if MODEL == "multipitch_parallel":
    denoising_model= MultiPitchRecurrentScore(n_in_channels=128,n_conditioning_channels=512, n_pitches=models_metadata[MODEL]["n_pitches"])
else:
    denoising_model= RecurrentScore(n_in_channels=128,n_conditioning_channels=512,hidden_size=model_metadata["hidden_size"],reduced_text_embedding_size=model_metadata["reduced_text_embedding_size"])

#@title Model code
class DiffusionModel(nn.Module):
    def __init__(self, global_args , denoising_model=None):
        super().__init__()
        self.diffusion = denoising_model
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


print(torch.load(model_metadata["checkpoint_path"],map_location=torch.device(device)))

ENCODEC_FRAME_RATE = 150
ENCODEC_CHANELS = 128

print("Creating the model...")
model = DiffusionModel(args, denoising_model=denoising_model)
state_dict=torch.load(model_metadata["checkpoint_path"],map_location=torch.device(device))["state_dict"]

print(state_dict.keys())
model.load_state_dict(state_dict, strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.requires_grad_(False).to(device)
print("Model created")

# # Remove non-EMA
del model.diffusion
model_fn = model.diffusion_ema

def midi_to_hz(note):
    a = 440 #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))

encodec_processor = EncodecProcessor(48000).to(device)

def sonify(fakes):
        embeddings = fakes
        # decode here
        fakes = encodec_processor.decode_embeddings(embeddings)

        # Put the demos together
        fakes = rearrange(fakes, "b d n -> d (b n)")
        fakes = fakes / torch.max(torch.abs(fakes) + 1e-8)
        return fakes

def text2sound(text,steps,batch_size):
  text_embedding = text_embedder.embed_text(text).to(device)[None,:].repeat(batch_size,1)

  torch.cuda.empty_cache()
  gc.collect()
  noise = torch.randn([batch_size, ENCODEC_CHANELS, ENCODEC_FRAME_RATE * model_metadata["clip_s"]* model_metadata["n_pitches"] ]).to(device)
  generated = sample(model_fn, noise, text_embedding, steps, sampler_type)
  generated = sonify(generated)
  generated = generated.cpu().detach()

  audio = generated.clamp(-1, 1)
  audio = torchaudio.functional.highpass_biquad(audio,args.sample_rate,15.0,Q=0.707)
  # audio_memap = nr.reduce_noise(y=audio.cpu().numpy()[0], sr=args.sample_rate, prop_decrease=0.5)
  # wavfile.write("artefacts/noise_reduced.wav", args.sample_rate, audio_memap)
  # audio,sr= torchaudio.load("artefacts/noise_reduced.wav")
  # convert to numpy array
  plot_and_hear(audio, args.sample_rate)
  return audio
