#%%
from encodec_processor import EncodecProcessor
import glob
import pandas
import torchaudio
import torch
from tqdm import tqdm
from utils import play_audio

AUDIO_FILEPATH_PATTERN = "data/KillerBee samples/**/*.wav"

SAMPLE_RATE = 48000
CLIP_S = 1

#%%

encodec_processor = EncodecProcessor(SAMPLE_RATE)
#%%
data = []

fps = glob.glob(AUDIO_FILEPATH_PATTERN, recursive=True)

DISALLOWED = ["Loops","Loop","Break","bpm"]

for fp in tqdm(fps):
    try:
        wav, sr = torchaudio.load(fp)
    except:
        print(fp)
        # if there is an error, skip the file
        continue

    DURATION=1
    # crop/pad to 1 second
    if wav.shape[1] > DURATION*sr:
        wav = wav[:, :DURATION*sr]

    if wav.shape[1] < DURATION*sr:
        wav = torch.nn.functional.pad(wav, (0, DURATION*sr - wav.shape[1]))

    # resample
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # normalize
    wav = wav / torch.max(torch.abs(wav) + 1e-8)

    # encode
    encoded_frames = encodec_processor.encode(wav, SAMPLE_RATE)

    n_frames = encoded_frames[0][0].shape[-1]

    # get wav frames
    wav_frames = wav.reshape(1, n_frames ,-1)

    # compute rms per frames
    rms = torch.sqrt(torch.mean(wav_frames**2, dim=-1))

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
            "frame_rms": rms,
        }
    )

torch.save(data, "artefacts/kb_data.pt")
#%%
#%%


# %%
