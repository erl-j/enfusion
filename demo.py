# %%
if False:
    !git clone --recursive https://github.com/crowsonkb/v-diffusion-pytorch
    !pip install sample-generator
    !pip install v-diffusion-pytorch
    !pip install ipywidgets==7.7.1 gradio
    !pip install k-diffusion

# %%
import os
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
  "multipitch_sequential": {"checkpoint_path":"demo_assets/multiplenotes/sequential/epoch=2105-step=200000.ckpt", "clip_s":5, "n_pitches":9},
  "single_pitch": {"checkpoint_path":"demo_assets/single_note/epoch=6363-step=70000.ckpt", "clip_s":5, "n_pitches":1},
}

MODEL = "single_pitch"

if MODEL == "multipitch_parallel":
    denoising_model= MultiPitchRecurrentScore(n_in_channels=128,n_conditioning_channels=512, n_pitches=models_metadata[MODEL]["n_pitches"])
else:
    denoising_model= RecurrentScore(n_in_channels=128,n_conditioning_channels=512)

#@title Model code
class DiffusionModel(nn.Module):
    def __init__(self, global_args , denoising_model=None):
        super().__init__()
        self.diffusion = denoising_model
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

model_metadata = models_metadata[MODEL]

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

def sonify(fakes):
        embeddings = fakes
        # decode here
        fakes = encodec_processor.decode_embeddings(embeddings)

        # Put the demos together
        fakes = rearrange(fakes, "b d n -> d (b n)")
        fakes = fakes / torch.max(torch.abs(fakes) + 1e-8)
        return fakes

# %% [markdown]
# Select the sampler you want to use
# ---
# Sampler name | Notes
# --- | ---
# v-iplms | This is what the model expects. Needs more steps, but more reliable.
# k-heun | Needs fewer steps, but ideal sigma_min and sigma_max need to be found. Doesn't work with all models.
# k-dpmpp_2s_ancestral | Fastest sampler, but you may have to find new sigmas. Recommended min & max sigmas: 0.01, 80
# k-lms | "
# k-dpm-2 | "
# k-dpm-fast | "
# k-dpm-adaptive | Takes in extra parameters for quality, step count is non-deterministic

# %%
#@title Sampler options

# %% [markdown]
# # Generate new sounds
# 
# Feeding white noise into the model to be denoised creates novel sounds in the "space" of the training data.

# %%

#@markdown How many audio clips to create
batch_size =  20#@param {type:"number"}
#@markdown Number of steps (100 is a good start, more steps trades off speed for quality)
steps = 300 #@param {type:"number"}
#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

text="Hard bass"
text_embedding = text_embedder.embed_text(text).to(device)[None,:].repeat(batch_size,1)
encodec_processor = EncodecProcessor(48000).to(device)

if not skip_for_run_all:
  torch.cuda.empty_cache()
  gc.collect()
  notes = 48 
  noise = torch.randn([batch_size, ENCODEC_CHANELS, ENCODEC_FRAME_RATE * model_metadata["clip_s"]* model_metadata["n_pitches"] ]).to(device)
  generated = sample(model_fn, noise, text_embedding, steps, sampler_type)
  generated = sonify(generated)
  generated = generated.cpu().detach()
  generated_all = generated.clamp(-1, 1)
  plot_and_hear(generated_all, args.sample_rate)
else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable")

EXPORT_SOUNDFONT = False
if EXPORT_SOUNDFONT:
  audio = generated_all

  audio = torchaudio.functional.highpass_biquad(audio,args.sample_rate,15.0,Q=0.707)

  plot_and_hear(audio, args.sample_rate)

  notes = torch.split(einops.rearrange(audio,"c (p t) -> p c t", p = model_metadata["n_pitches"]),1,0)

  min_midi_pitch = 36
  regions = []
  for note_idx, note in enumerate(notes):
      midi_pitch_nr = min_midi_pitch + note_idx*6
      regions.append({
        "midi_pitch_nr": midi_pitch_nr,
        "waveform": note[0].T,
        "lokey": midi_pitch_nr-3,
        "hikey": midi_pitch_nr+3,
      } 
      )

  timestamp =  datetime.now().strftime("%Y%m%d_%H%M%S")
  outpath = f"artefacts/instruments/{text}_{timestamp}/"

  export_sfz(outpath,regions, args.sample_rate)

#%%
# %%
#@title Generate new sounds from recording

batch_size=3
#@markdown Total number of steps (100 is a good start, more steps trades off speed for quality)
steps = 100#@param {type:"number"}

#@markdown How much (0-1) to re-noise the original sample. Adding more noise (a higher number) means a bigger change to the input audio
noise_level = 0.9#@param {type:"number"}

#@markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 1#@param {type:"number"}

#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

text="Hard bass"
text_embedding = text_embedder.embed_text(text).to(device)[None,:].repeat(batch_size,1)
encodec_processor = EncodecProcessor(48000).to(device)

if not skip_for_run_all:
  torch.cuda.empty_cache()
  gc.collect()

  hz = midi_to_hz(60)

  audio_sample = torch.sin(torch.linspace(0, 2 * math.pi * hz * model_metadata["clip_s"], args.sample_rate * model_metadata["clip_s"] )).unsqueeze(0)*0.9

  audio_sample =audio_sample.to(device) #+ torch.randn_like(audio_sample) * noise_level

  print(audio_sample.shape)

  start_embedding = encodec_processor.encode_wo_quantization(audio_sample, args.sample_rate)

  print(start_embedding.shape)

  audio_sample_hat = encodec_processor.decode_embeddings(start_embedding)[0]

  plot_and_hear(audio_sample, args.sample_rate)
  plot_and_hear(audio_sample_hat.detach().cpu(), args.sample_rate)

  start_embedding = start_embedding.repeat(batch_size,1,1)
  
  generated = resample(model_fn, start_embedding, text_embedding,steps, sampler_type, noise_level=noise_level)
  generated = sonify(generated)
  generated = generated.cpu().detach()
  generated_all = generated.clamp(-1, 1)
  plot_and_hear(generated_all, args.sample_rate)
else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable")

 # %% [markdown]
# # Regenerate your own sounds
# By adding noise to an audio file and running it through the model to be denoised, new details will be created, pulling the audio closer to the "sonic space" of the model. The more noise you add, the more the sound will change.
# 
# The effect of this is a kind of "style transfer" on the audio. For those familiar with image generation models, this is analogous to an "init image".

# %%
#@title Record audio or enter a filepath to a prerecorded audio file
import torch
import torchaudio
from typing import Iterable, Tuple

Audio = Tuple[int, np.ndarray]

#@markdown Check the box below to create an audio recording interface below
record_audio = True #@param {type: "boolean"}

#@markdown If you left "record_audio" blank, enter a path to an audio file you want to alter, or leave blank to upload a file (.wav or .flac).
file_path = "" #@param{type:"string"}

#@markdown Number of audio recordings to combine into one clip. Only applies if the "record_audio" box is checked.
n_audio_recordings = 1 #@param{type:"number"}

# this is a global variable to be filled in by the generate_from_audio callback
recording_file_path = ""


def combine_audio(*audio_iterable: Iterable[Audio]) -> Audio:
    """Combines an iterable of audio signals into one."""
    max_len = max([x.shape for _, x in audio_iterable])
    combined_audio = np.zeros(max_len, dtype=np.int32)
    for _, a in audio_iterable:
        combined_audio[:a.shape[0]] = combined_audio[:a.shape[0]] * .5 + a * .5
    return combined_audio


def generate_from_audio(file_path: str, *audio_iterable: Iterable[Audio]):
    sample_rate = audio_iterable[0][0]
    combined_audio = combine_audio(*audio_iterable)
    tensor = torch.from_numpy(
        np.concatenate(
            [
                combined_audio.reshape(1, -1),
                combined_audio.reshape(1, -1)
            ],
            axis=0,
        )
    )
    global recording_file_path
    recording_file_path = file_path
    torchaudio.save(
        file_path,
        tensor,
        sample_rate=sample_rate,
        format="wav"
    )
    return (sample_rate, combined_audio), file_path

if record_audio:
    recording_interface = gr.Interface(
        fn=generate_from_audio,
        inputs=[
            gr.Textbox(
                "/content/recording.wav",
                label="save recording to filepath",
            ),
            *[
                gr.Audio(source="microphone", label=f"audio clip {i}")
                for i in range(1, n_audio_recordings + 1)
            ]
        ],
        outputs=[
            gr.Audio(label="combined output audio"),
            gr.File(label="output file"),
        ],
        allow_flagging="never",
    )

    recording_interface.launch();
elif file_path == "":
    print("No file path provided, please upload a file")
    # uploaded = files.upload()
    file_path = list(uploaded.keys())[0]

if not record_audio:
    print(f"Using file_path: {file_path} to regenerate new sounds.")

# %%
#@title Generate new sounds from recording

#@markdown Total number of steps (100 is a good start, more steps trades off speed for quality)
steps = 10#@param {type:"number"}

#@markdown How much (0-1) to re-noise the original sample. Adding more noise (a higher number) means a bigger change to the input audio
noise_level = 0.3#@param {type:"number"}

#@markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 2#@param {type:"number"}

#@markdown How many variations to create
batch_size = 4 #@param {type:"number"}

#@markdown Check the box below to save your generated audio to [Weights & Biases](https://www.wandb.ai/site)
save_own_generations_to_wandb = False #@param {type: "boolean"}

#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

effective_length = args.sample_size * sample_length_mult

if not skip_for_run_all:
  torch.cuda.empty_cache()
  gc.collect()

  augs = torch.nn.Sequential(
    PadCrop(effective_length, randomize=True),
    Stereo()
  )

  fp = recording_file_path if record_audio else file_path

  audio_sample = load_to_device(fp, args.sample_rate)

  audio_sample = augs(audio_sample).unsqueeze(0).repeat([batch_size, 1, 1])

  print("Initial audio sample")
  plot_and_hear(audio_sample[0], args.sample_rate)
  
  generated = resample(model_fn, audio_sample, steps, sampler_type, noise_level=noise_level)

  print("Regenerated audio samples")
  plot_and_hear(rearrange(generated, 'b d n -> d (b n)'), args.sample_rate)

  for ix, gen_sample in enumerate(generated):
    print(f'sample #{ix + 1}')
    display(ipd.Audio(gen_sample.cpu(), rate=args.sample_rate))

else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable")

# %% [markdown]
# # Interpolate between sounds
# Diffusion models allow for interpolation between inputs through a process of deterministic noising and denoising. 
# 
# By deterministically noising two audio files, interpolating between the results, and deterministically denoising them, we can can create new sounds "between" the audio files provided.

# %%
# Interpolation code taken and modified from CRASH
def compute_interpolation_in_latent(latent1, latent2, lambd):
    '''
    Implementation of Spherical Linear Interpolation: https://en.wikipedia.org/wiki/Slerp
    latent1: tensor of shape (2, n)
    latent2: tensor of shape (2, n)
    lambd: list of floats between 0 and 1 representing the parameter t of the Slerp
    '''
    device = latent1.device
    lambd = torch.tensor(lambd)

    assert(latent1.shape[0] == latent2.shape[0])

    # get the number of channels
    nc = latent1.shape[0]
    interps = []
    for channel in range(nc):
    
      cos_omega = latent1[channel]@latent2[channel] / \
          (torch.linalg.norm(latent1[channel])*torch.linalg.norm(latent2[channel]))
      omega = torch.arccos(cos_omega).item()

      a = torch.sin((1-lambd)*omega) / np.sin(omega)
      b = torch.sin(lambd*omega) / np.sin(omega)
      a = a.unsqueeze(1).to(device)
      b = b.unsqueeze(1).to(device)
      interps.append(a * latent1[channel] + b * latent2[channel])
    return rearrange(torch.cat(interps), "(c b) n -> b c n", c=nc) 

#@markdown Enter the paths to two audio files to interpolate between (.wav or .flac)
source_audio_path = "" #@param{type:"string"}
target_audio_path = "" #@param{type:"string"}

#@markdown Total number of steps (100 is a good start, can go lower for more speed/less quality)
steps = 100#@param {type:"number"}

#@markdown Number of interpolated samples
n_interps = 12 #@param {type:"number"}

#@markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 1#@param {type:"number"}

#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

effective_length = args.sample_size * sample_length_mult

if not skip_for_run_all:

  augs = torch.nn.Sequential(
    PadCrop(effective_length, randomize=True),
    Stereo()
  )

  if source_audio_path == "":
    print("No file path provided for the source audio, please upload a file")
    uploaded = files.upload()
    source_audio_path = list(uploaded.keys())[0]

  audio_sample_1 = load_to_device(source_audio_path, args.sample_rate)

  print("Source audio sample loaded")

  if target_audio_path == "":
    print("No file path provided for the target audio, please upload a file")
    uploaded = files.upload()
    target_audio_path = list(uploaded.keys())[0]

  audio_sample_2 = load_to_device(target_audio_path, args.sample_rate)

  print("Target audio sample loaded")

  audio_samples = augs(audio_sample_1).unsqueeze(0).repeat([2, 1, 1])
  audio_samples[1] = augs(audio_sample_2)

  print("Initial audio samples")
  plot_and_hear(audio_samples[0], args.sample_rate)
  plot_and_hear(audio_samples[1], args.sample_rate)

  reversed = reverse_sample(model_fn, audio_samples, steps)

  latent_series = compute_interpolation_in_latent(reversed[0], reversed[1], [k/n_interps for k in range(n_interps + 2)])

  generated = sample(model_fn, latent_series, steps) 
  
  #sampling.iplms_sample(, latent_series, step_list.flip(0)[:-1], {})

  # Put the demos together
  generated_all = rearrange(generated, 'b d n -> d (b n)')

  print("Full interpolation")
  plot_and_hear(generated_all, args.sample_rate)
  for ix, gen_sample in enumerate(generated):
    print(f'sample #{ix + 1}')
    display(ipd.Audio(gen_sample.cpu(), rate=args.sample_rate))
else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable") 



