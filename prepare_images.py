#%%
from encodec_processor import EncodecProcessor
import torch
import pandas as pd
from utils import play_audio
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import seaborn as sns

#%%
SAMPLE_RATE = 48000
#%%
data = torch.load("artefacts/drums_data.pt")

# embeddings
embeddings = torch.concat(
    [data[i]["encoded_frames_embeddings"] for i in range(len(data))], dim=0
)

min_embedding = torch.min(embeddings)
max_embedding = torch.max(embeddings)

new_embeddings = deepcopy(embeddings)
# standard scale
new_embeddings = (embeddings - min_embedding) / (max_embedding - min_embedding)

#%%
# upscale to batch by 512 by 512
new_embeddings = torch.nn.functional.interpolate(
    embeddings[:, None, ...], size=(512, 512), mode="bilinear"
)[:, 0, ...]


#%%
#%% downscaled embeddings to 128 by 150
downscaled_embeddings = torch.nn.functional.interpolate(
    embeddings[:, None, ...], size=(128, 150), mode="bilinear"
)[:, 0, ...]


#%%
print(embeddings[0].shape)

#%%
plt.imshow(embeddings[0])
plt.show()

plt.imshow(new_embeddings[0])
plt.show()

plt.imshow(downscaled_embeddings[0])
plt.show()

# %%
print(downscaled_embeddings - embeddings)

# %%
