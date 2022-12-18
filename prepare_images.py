#%%
from encodec_processor import EncodecProcessor
import torch
from utils import play_audio
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import numpy as np
#%%
SAMPLE_RATE = 48000
#%%
data = torch.load("artefacts/drums_data.pt")

# embeddings
embeddings = torch.concat(
    [data[i]["encoded_frames_embeddings"] for i in range(len(data))], dim=0
)

#%%
sample_mins = torch.amin(embeddings, dim=[1,2])
sample_maxs = torch.amax(embeddings, dim=[1,2])

print(sample_mins.shape)
print(sample_maxs.shape)

#%%
T=15

print(torch.sum(sample_mins<-T))
print(torch.sum(sample_maxs>T))

#%%

min_embedding = torch.min(embeddings)
max_embedding = torch.max(embeddings)

print(min_embedding)
print(max_embedding)

MIN=-20
MAX=18

new_embeddings = deepcopy(embeddings)
# standard scale
new_embeddings = (embeddings - MIN) / (MAX - MIN)

#%%
# upscale to batch by 512 by 512
new_embeddings = torch.nn.functional.interpolate(
    embeddings[:, None, ...], size=(512, 512), mode="bilinear"
)[:, 0, ...]

#%%
# turn image into rgb according to inferno colormap

embedding = new_embeddings[0]
im_array =embedding.detach().numpy()
im_array=(im_array * 255).astype(np.uint8)

print(np.max(im_array))
print(np.min(im_array))
# turn into PIL image
im = PIL.Image.fromarray(im_array)
# save image
im.save("artefacts/images/test.png")
# load PIL image
im = PIL.Image.open("artefacts/images/test.png")

# turn into rgb array
im_array = np.array(im)[..., :3]

print(np.max(im_array))
print(np.min(im_array))

# turn into 0-1
im = im / 255

im = im[..., 0]

# invert colormap
embedding_hat = im

print(embedding_hat.shape)

print(embedding-embedding_hat)



#%% 
print(embedding.shape)
print(embedding_hat.shape)

#%%
print(embedding-embedding_hat)






#%%

#%% downscaled embeddings to 128 by 150
downscaled_embeddings = torch.nn.functional.interpolate(
    embeddings[:, None, ...], size=(128, 150), mode="bilinear"
)[:, 0, ...]


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
