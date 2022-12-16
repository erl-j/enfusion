#%%
from encodec_processor import EncodecProcessor
import torch
import pandas as pd
from utils import play_audio
from copy import deepcopy

SAMPLE_RATE = 48000
#%%
frames = torch.load("artefacts/drums_encoded_frames.pt")

# df = pd.read_csv("artefacts/drums_metadata.csv")

encodec_processor = EncodecProcessor(SAMPLE_RATE)
# %%
for i in range(10):
    print(frames[i][0])
#%%
quantized_frames = [
    encodec_processor.quantize(frames[i], 16) for i in range(len(frames))
]


print(quantized_frames[3][1])

#%%
for i in range(10):
    wav = encodec_processor.decode(quantized_frames[i])
    play_audio(wav, SAMPLE_RATE)

tweaked_frames = deepcopy(quantized_frames)


#%%
N = 10
tweaked_frames[0][0][0][:, :, :] = (tweaked_frames[0][0][0][:, :, :] - 2) % 1024

wav = encodec_processor.decode(tweaked_frames[1])

play_audio(wav, SAMPLE_RATE)

# %%
