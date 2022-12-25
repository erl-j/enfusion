#%%
from encodec_processor import EncodecProcessor
import glob
import pandas
import torchaudio
import torch
from tqdm import tqdm
from utils import play_audio
import numpy as np
from transformers import CLIPTokenizer, CLIPModel

AUDIO_FILEPATH_PATTERN = "data/KillerBee samples/**/*.wav"

SAMPLE_RATE = 48000
CLIP_S = 1

#%%

encodec_processor = EncodecProcessor(SAMPLE_RATE)
#%%
data = []

fps = glob.glob(AUDIO_FILEPATH_PATTERN, recursive=True)

DISALLOWED = ["Loops","Loop","Break","bpm"]

fps = [fp for fp in fps if not any([d in fp for d in DISALLOWED])]
#%%

# integer show the minimum of levels of subfolders (starting from the key) to include in caption 
killerbee_levels={
    "Instruments":1,
    "Instruments/Bass":2,
    "Kick":1,
    "808 Subs":1,
    "FX":1,
    "Instrument":1,
    "Instrument/Guitar":1,
    "Instrument/Synth":1,
    "Instrument/Bass":1,
    "HiHat":1,
    "Drum Machines":1,
    "Drum Machines/Roland TR-808":1,
    "Vox":1,
    "Snare":1,
    "Cymbals":1,
    "Percussion":1,
    "Toms":1,
}

def fp2captions(fp):
    fp=fp.replace("data/KillerBee samples/","")

    # sort keys by length
    keys = sorted(killerbee_levels.keys(), key=len, reverse=True)

    captions=[]
    for key in keys:
        if key in fp:
            # key levels
            n_key_levels = len(key.split("/"))
            # get the number of levels to include in caption
            n_levels = killerbee_levels[key]
            # split the filepath into levels
            levels = fp.split("/")
            # get the levels to include in caption
            levels = levels[:n_levels+n_key_levels]
            # join the levels into a caption
            captions+=levels
        # remove duplicates
    captions = list(set(captions))

    # add file name to caption
    #caption = caption + " " + fp.split("/")[-1].replace(".wav","")
    return captions


sample_fps = np.random.choice(fps, 10)
for fp in sample_fps:
    print(fp)
    print(fp2captions(fp))
    print("#")

#%%

class TextEmbedder():

    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_to_embeddings = {}
    
    def _embed_text(self,string):
        input_ids = self.tokenizer(string, return_tensors="pt",padding=True).input_ids
        with torch.no_grad():
            text_features= self.model.get_text_features(input_ids=input_ids).detach()[0]
        return text_features
        
    def embed_text(self,text):
        if text not in self.text_to_embeddings:
            self.text_to_embeddings[text] = self._embed_text(text)
        return self.text_to_embeddings[text]

text_embedder = TextEmbedder()


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

    captions = fp2captions(fp)

    text_embeddings = [text_embedder.embed_text(caption) for caption in captions]

    

    data.append(
        {
            "folder": folder,
            "filename": filename,
            "filepath": fp,
            "encoded_frames_codes": encoded_frames,
            "encoded_frames_embeddings": embeddings,
            "frame_rms": rms,
            "captions": captions,
            "text_embeddings": text_embeddings,
        }
    )
    print(data[-1])


torch.save(data, "artefacts/kb_data_with_text_embeddings.pt")
#%%
#%%


# %%
