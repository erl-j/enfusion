#%%
from encodec_processor import EncodecProcessor
import pedalboard as pb
import glob
import pandas
import torchaudio
import torch
from tqdm import tqdm
from utils import play_audio

AUDIO_FILEPATH_PATTERN = "data/drums/*/*.wav"

SAMPLE_RATE = 48000
CLIP_S = 1

#%%

encodec_processor = EncodecProcessor(SAMPLE_RATE)
#%%
data = []


fps = glob.glob(AUDIO_FILEPATH_PATTERN, recursive=True)
for fp in tqdm(fps):
    try:
        wav, sr = torchaudio.load(fp)
    except:
        print(fp)
        # if there is an error, skip the file
        continue

    # crop/pad to 1 second
    if wav.shape[1] > sr:
        wav = wav[:, :sr]

    if wav.shape[1] < sr:
        wav = torch.nn.functional.pad(wav, (0, sr - wav.shape[1]))

    # resample
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # normalize
    wav = wav / torch.max(torch.abs(wav) + 1e-8)

    # encode
    encoded_frames = encodec_processor.encode(wav, SAMPLE_RATE)

    # show max and min

    # decode
    # play_audio(wav, SAMPLE_RATE)
    # reconstructed_wav = encodec_processor.decode(encoded_frames)
    # play_audio(reconstructed_wav, SAMPLE_RATE)

    embeddings = encodec_processor.encode_wo_quantization(wav, SAMPLE_RATE)

    folder = fp.split("/")[2]
    # filenames
    filename = fp.split("/")[3]

    data.append(
        {
            "folder": folder,
            "filename": filename,
            "filepath": fp,
            "encoded_frames_codes": encoded_frames,
            "encoded_frames_embeddings": embeddings,
        }
    )

torch.save(data, "artefacts/drums_data.pt")
#%%
#%%


# %%
