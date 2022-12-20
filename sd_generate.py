#%%
import sys

sys.path.append("./enfusion")

from diffusers import StableDiffusionPipeline
import torch
import IPython.display
from encodec_processor import EncodecProcessor,image2embedding,embedding2image
from audio_encodec_to_sd_dataset import DrumfusionDataset
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from PIL import Image

#%%

MODEL_NAME="sd-scaled-asdf"

model_path = f"./artefacts/{MODEL_NAME}"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# %%

#%% create dummy safety checker
def dummy(images, **kwargs):
    return images, False

pipe.safety_checker = dummy

ds = DrumfusionDataset()

#%%

for prompt in ["kick"]:#,"kick","hihat","crash"]:
    wavs=[]
    for i in range(10):
        image = pipe(prompt=prompt,guidance_scale=7.0,num_inference_steps=50).images[0]
        # image.show()
        im = torch.tensor(np.array(image)).to(torch.float32)[None, ...]


        # load image from 
        #filename = f"./artefacts/scaled_aesd_dataset/{i}.png"
        #im = torch.tensor(np.array(Image.open(filename))).to(torch.float32)[None, ...]

        embedding = ds.image2embedding(im)

        SAMPLE_RATE=48000
        encodec_processor = EncodecProcessor(SAMPLE_RATE)

        # plt.imshow(embedding[0], aspect="auto")
        # plt.show()

        wav = encodec_processor.decode_embeddings(embedding.to(torch.float32))

        wavs.append(wav)

    wav = torch.concat(wavs, dim=-1)[0]
    IPython.display.Audio(wav.detach().cpu().numpy(), rate=SAMPLE_RATE)

    # play audio
    IPython.display.Audio(wav.detach().cpu().numpy(), rate=SAMPLE_RATE)

    wav = wav/torch.max(torch.abs(wav)+1e-8)
    # save to file
    torchaudio.save(f"./artefacts/demos/{MODEL_NAME}_prompt={prompt}.wav", wav, SAMPLE_RATE)

# %%
