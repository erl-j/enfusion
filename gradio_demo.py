# %%
import gc
import os
from copy import deepcopy

import torch
import torchaudio
from einops import rearrange
#@title Imports and definitions
from torch import nn, optim

from encodec_processor import EncodecProcessor
from export_sfz import export_sfz
from models import MultiPitchRecurrentScore, RecurrentScore
from sampling import resample, reverse_sample, sample
from text_embedder import TextEmbedder

device ="cuda" if torch.cuda.is_available() else "cpu"

import gradio as gr

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

#%%
models_metadata = {
  "multipitch_parallel": {"checkpoint_path":"demo_assets/multiplenotes/parallel/epoch=4210-step=400000.ckpt", "clip_s":5, "n_pitches":9},
  "multipitch_sequential": {"checkpoint_path":"demo_assets/multiplenotes/sequential/epoch=2105-step=200000.ckpt", "clip_s":5, "n_pitches":9},
  "single_pitch": {"checkpoint_path":"demo_assets/single_note/epoch=6363-step=70000.ckpt", "clip_s":5, "n_pitches":1},
}

MODEL = "single_pitch"

model_metadata = models_metadata[MODEL]

## download model file
local_path = models_metadata[MODEL]["checkpoint_path"]
os.makedirs(os.path.dirname(local_path), exist_ok=True)
remote_path = model_metadata["checkpoint_path"].replace("demo_assets/","https://github.com/erl-j/enfusion-weights/raw/main")

# download model file and save it locally
if not os.path.exists(local_path):
    print(f"Downloading {remote_path} to {local_path}")
    os.system(f"wget -O {local_path} {remote_path}")
    print("Download complete")

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

def generate(text,steps,n_sounds=1):
    batch_size = n_sounds
    text_embedding = text_embedder.embed_text(text).to(device)[None,:].repeat(batch_size,1)
    torch.cuda.empty_cache()
    gc.collect()
    notes = 48 
    noise = torch.randn([batch_size, ENCODEC_CHANELS, ENCODEC_FRAME_RATE * model_metadata["clip_s"]* model_metadata["n_pitches"] ]).to(device)
    generated = sample(model_fn, noise, text_embedding, steps, sampler_type)
    generated = sonify(generated)
    generated = generated.cpu().detach()
    generated_all = generated.clamp(-1, 1)
    #plot_and_hear(generated_all, args.sample_rate)
    fp = "artefacts/generated.wav"
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    torchaudio.save(fp, generated_all, args.sample_rate)
    return fp

output = gr.Audio(label="Output")

inputs = [
        gr.Textbox(label="Prompt", value="Bright organ", max_lines=3), 
        gr.Slider(label="Denoising steps", minimum=1, maximum=1000, value=200, step=1),
        #gr.Slider(label="Number of outputs", min=1, max=10, default=1, step=1)
        ]

examples = [
    ["Hard Bass",100,1],
]

gr.Interface(generate, inputs, output, examples=examples).launch()