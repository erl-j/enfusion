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
embeddings = torch.stack(
    [data[i]["encoded_frames_embeddings"] for i in range(len(data))]
)
# %%

# print min and max
print(torch.min(embeddings))
print(torch.max(embeddings))

#%%

#%%

# %%

# get min
maxs = torch.amax(embeddings, dim=[1, 2], keepdim=False)[0]
mins = torch.amin(embeddings, dim=[1, 2], keepdim=False)[0]

print(mins)
print(maxs)

print(torch.min(mins))
print(torch.max(maxs))


#%%
sns.histplot(mins, bins=100)
sns.histplot(maxs, bins=100)
plt.show()


#%%


# %%
plt.show()
