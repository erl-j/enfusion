#%%
from encodec_processor import EncodecProcessor
import pedalboard as pb
import glob
import pandas
import torchaudio
import torch
from tqdm import tqdm
from utils import play_audio
import matplotlib.pyplot as plt

#%%

AUDIO_FILEPATH_PATTERN = "data/drums/*/*.wav"

SAMPLE_RATE = 48000
CLIP_S = 1

encodec_processor = EncodecProcessor(SAMPLE_RATE)
#%%
frames = []

good_fps = []

fps = glob.glob(AUDIO_FILEPATH_PATTERN, recursive=True)

for i in range(10):

    fp = fps[100 + i]

    wav, sr = torchaudio.load(fp)

    good_fps.append(fp)
    # crop/pad to 1 second
    if wav.shape[1] > sr:
        wav = wav[:, :sr]

    if wav.shape[1] < sr:
        wav = torch.nn.functional.pad(wav, (0, sr - wav.shape[1]))

    # resample
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # normalize
    wav = wav / torch.max(torch.abs(wav) + 1e-8)

    play_audio(wav, SAMPLE_RATE)

    # # encode
    # encoded_frames = encodec_processor.encode(wav, SAMPLE_RATE)

    embeddings = encodec_processor.encode_wo_quantization(wav, SAMPLE_RATE)

    # # compute sums across time
    # sums = torch.sum(embeddings[0], dim=-1)

    # # # sort embeddings by sum
    # # sorted_embeddings = embeddings[:, torch.argsort(sums, dim=-1), :]

    plt.imshow(embeddings[0])
    plt.show()

    # plot some statistics of the embeddings

    # # show max and min
    print(torch.max(embeddings[0]))
    print(torch.min(embeddings[0]))


# %%
