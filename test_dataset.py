#%%
from dataset import ALVDataset, EnfusionDataset


ds = EnfusionDataset("artefacts/killerbee_dataset.pt")


# print(len(ds))

# print(ds[0])
# dataset = ALVDataset("../synth_text_dataset")


# dataset.save("artefacts/alv_dataset.pt")

# %%


# %%
# import torch
# import numpy as np

# class DrumfusionDataset(torch.utils.data.Dataset):
#         def __init__(self):
#             self.data = torch.load("artefacts/kb_data_with_text_embeddings.pt")
#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             fp = self.data[idx]["filepath"]
#             embedding = self.data[idx]["encoded_frames_embeddings"][0]
#             rms = self.data[idx]["frame_rms"]
#             text_embeddings = self.data[idx]["text_embeddings"]
#             text_embedding = text_embeddings[np.random.randint(0, len(text_embeddings))]
#             return (embedding, text_embedding)

#         def export(self):
#             out_data = [{"fp":self.data[idx]["filepath"],"audio_embeddings": [self.data[idx]["encoded_frames_embeddings"][0]], "text_embeddings": self.data[idx]["text_embeddings"],"texts":self.data[idx]["captions"]} for idx in range(len(self.data))]
#             torch.save(out_data, "artefacts/killerbee_dataset.pt")

# train_set = DrumfusionDataset()

# train_set.export()
# %%
