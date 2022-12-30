import torch
from encodec_processor import EncodecProcessor
from text_embedder import TextEmbedder
import torchaudio
import json
from tqdm import tqdm
import numpy as np

class EnfusionDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def save(self, path):
        torch.save(self.data, path)

    def __getitem__(self, index):
        audio_index = np.random.randint(0, len(self.data[index]["audio_embeddings"]))
        text_index = np.random.randint(0, len(self.data[index]["text_embeddings"]))
        audio_embedding = self.data[index]["audio_embeddings"][audio_index]
        text_embedding =  self.data[index]["text_embeddings"][text_index]
        return {"audio_embedding":audio_embedding, "text_embedding":text_embedding}

class ALVDataset(EnfusionDataset):
    def __init__(self,dataset_path=None,preprocessed_path=None) -> None:
        if preprocessed_path is not None:
            super().__init__(preprocessed_path)
        else:
            DURATION=5
            SAMPLE_RATE = 48000
            self.metadata = json.load(open(f"{dataset_path}/patch_metadata.json"))

            encodec_processor = EncodecProcessor(SAMPLE_RATE)
            text_embedder = TextEmbedder()

            self.data=[]

            for i in tqdm(range(len(self.metadata))):

                pm = self.metadata[i]
                captions = self.get_augmented_text_attributes(pm)

                text_embeddings = [text_embedder.embed_text(caption) for caption in captions]

                timestamp = pm["timestamp"]

                embeddings=[]
                for j in range(0,16):
                    audio_path = f"{dataset_path}/cropped_audio/{timestamp}/6.wav"
                    wav, sr = torchaudio.load(audio_path)
                    # crop/pad to duration
                    if wav.shape[1] > DURATION*sr:
                        wav = wav[:, :DURATION*sr]
                    if wav.shape[1] < DURATION*sr:
                        wav = torch.nn.functional.pad(wav, (0, DURATION*sr - wav.shape[1]))
                    # resample
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
                    # normalize
                    wav = wav / torch.max(torch.abs(wav) + 1e-8)
                    # encode
                    embeddings.append(encodec_processor.encode_wo_quantization(wav, SAMPLE_RATE)[0])

                self.data.append({"audio_embeddings":embeddings, "text_embeddings":text_embeddings, "metadata":pm, "texts":captions})


    def get_augmented_text_attributes(self,pm):
        text = {
            "title": [{"string": pm["title"]}],
            "description": [{"string": pm["description"]}],
            "type": [{"string": pm["type"]}],
            "type_group": [{"string": pm["type_group"]}],
            "tags": [{"string": tag} for tag in pm["tags"]],
            "tag+type": [{"string": tag + " " + pm["type"]} for tag in pm["tags"]],
            "tag+type_group": [
                {"string": tag + " " + pm["type_group"]} for tag in pm["tags"]
            ],
        }
        # join all text attributes into one list
        texts = [item["string"] for sublist in text.values() for item in sublist]

        # remove "NA"
        texts = [t for t in texts if t != "NA"]
        return texts

    def __getitem__(self, index):
        # audio_index = 6#np.random.randint(0, len(self.data[index]["audio_embeddings"]))
        # audio_embedding = self.data[index]["audio_embeddings"][audio_index]
        audio_embedding = torch.concat(self.data[index]["audio_embeddings"], dim=-1)

        text_index = np.random.randint(0, len(self.data[index]["text_embeddings"]))
        text_embedding =  self.data[index]["text_embeddings"][text_index]
        return {"audio_embedding":audio_embedding, "text_embedding":text_embedding}
    
    
# class KillerBeeDataset():

#     def __init__(self) -> None:
        
#         AUDIO_FILEPATH_PATTERN = "data/KillerBee samples/**/*.wav"

#         SAMPLE_RATE = 48000
#         CLIP_S = 1

#         #%%
#         encodec_processor = EncodecProcessor(SAMPLE_RATE)
#         #%%
#         data = []

#         fps = glob.glob(AUDIO_FILEPATH_PATTERN, recursive=True)

#         DISALLOWED = ["Loops","Loop","Break","bpm"]

#         fps = [fp for fp in fps if not any([d in fp for d in DISALLOWED])]
#         #%%

#         # integer show the minimum of levels of subfolders (starting from the key) to include in caption 
#         killerbee_levels={
#             "Instruments":1,
#             "Instruments/Bass":2,
#             "Kick":1,
#             "808 Subs":1,
#             "FX":1,
#             "Instrument":1,
#             "Instrument/Guitar":1,
#             "Instrument/Synth":1,
#             "Instrument/Bass":1,
#             "HiHat":1,
#             "Drum Machines":1,
#             "Drum Machines/Roland TR-808":1,
#             "Vox":1,
#             "Snare":1,
#             "Cymbals":1,
#             "Percussion":1,
#             "Toms":1,
#         }

#         def fp2captions(fp):
#             fp=fp.replace("data/KillerBee samples/","")

#             # sort keys by length
#             keys = sorted(killerbee_levels.keys(), key=len, reverse=True)

#             captions=[]
#             for key in keys:
#                 if key in fp:
#                     # key levels
#                     n_key_levels = len(key.split("/"))
#                     # get the number of levels to include in caption
#                     n_levels = killerbee_levels[key]
#                     # split the filepath into levels
#                     levels = fp.split("/")
#                     # get the levels to include in caption
#                     levels = levels[:n_levels+n_key_levels]
#                     # join the levels into a caption
#                     captions+=levels
#                 # remove duplicates
#             captions = list(set(captions))

#             # add file name to caption
#             #caption = caption + " " + fp.split("/")[-1].replace(".wav","")
#             return captions


#         sample_fps = np.random.choice(fps, 10)
#         for fp in sample_fps:
#             print(fp)
#             print(fp2captions(fp))
#             print("#")

#         #%%

#         from text_embedder import TextEmbedder

#         text_embedder = TextEmbedder()


#         for fp in tqdm(fps):
#             try:
#                 wav, sr = torchaudio.load(fp)
#             except:
#                 print(fp)
#                 # if there is an error, skip the file
#                 continue

#             DURATION=1
#             # crop/pad to 1 second
#             if wav.shape[1] > DURATION*sr:
#                 wav = wav[:, :DURATION*sr]

#             if wav.shape[1] < DURATION*sr:
#                 wav = torch.nn.functional.pad(wav, (0, DURATION*sr - wav.shape[1]))

#             # resample
#             wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

#             # normalize
#             wav = wav / torch.max(torch.abs(wav) + 1e-8)

#             # encode
#             encoded_frames = encodec_processor.encode(wav, SAMPLE_RATE)

#             n_frames = encoded_frames[0][0].shape[-1]

#             # get wav frames
#             wav_frames = wav.reshape(1, n_frames ,-1)

#             # compute rms per frames
#             rms = torch.sqrt(torch.mean(wav_frames**2, dim=-1))

#             # decode
#             # play_audio(wav, SAMPLE_RATE)
#             # reconstructed_wav = encodec_processor.decode(encoded_frames)
#             # play_audio(reconstructed_wav, SAMPLE_RATE)

#             embeddings = encodec_processor.encode_wo_quantization(wav, SAMPLE_RATE)

#             folder = fp.split("/")[2]
#             # filenames
#             filename = fp.split("/")[3]

#             captions = fp2captions(fp)

#             text_embeddings = [text_embedder.embed_text(caption) for caption in captions]

            

#             data.append(
#                 {
#                     "folder": folder,
#                     "filename": filename,
#                     "filepath": fp,
#                     "encoded_frames_codes": encoded_frames,
#                     "encoded_frames_embeddings": embeddings,
#                     "frame_rms": rms,
#                     "captions": captions,
#                     "text_embeddings": text_embeddings,
#                 }
#             )
#             print(data[-1])


#         torch.save(data, "artefacts/kb_data_with_text_embeddings.pt")




