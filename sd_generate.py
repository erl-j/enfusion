#%%
import sys

sys.path.append("./enfusion")

from diffusers import StableDiffusionPipeline
import torch
import IPython.display
from encodec_processor import EncodecProcessor,image2embedding,embedding2image
from audio_encodec_to_sd_dataset import DrumfusionDataset
import numpy as np

#%%

model_path = "./enfusion/artefacts/sd-asdf"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# %%

#%% create dummy safety checker
def dummy(images, **kwargs):
    return images, False

pipe.safety_checker = dummy


#%%
image = pipe(prompt="snare", guidance_scale=7).images[0]
image.show()

im = torch.tensor(np.array(image))

embedding = image2embedding(im[None,...], min_value=-20, max_value=18)

SAMPLE_RATE=48000
encodec_processor = EncodecProcessor(SAMPLE_RATE)

wav = encodec_processor.decode_embeddings(embedding)

IPython.display.Audio(wav[0].detach().cpu().numpy(), rate=SAMPLE_RATE)


# %%
