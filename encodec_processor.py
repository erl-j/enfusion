#%%
from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch


def encodec_to_audio_scale(codes):
    return (codes / 1024) * 2 - 1


def audio_to_encodec_scale(audio):
    return ((audio + 1) / 2) * 1024


def embedding2image( embeddings,min_value=-20,max_value=18):
        # standard scale
        embeddings = (embeddings - min_value) / (max_value - min_value)
        # upscale to batch by 512 by 512
        embeddings = torch.nn.functional.interpolate(
            embeddings[:, None, ...], size=(512, 512), mode="bilinear"
        )[:, 0, ...]
        embeddings = (embeddings*255)[..., None].repeat([1, 1, 1, 3])
        return embeddings

def image2embedding( images,min_value=-20,max_value=18):
    # turn into 0-1
    images = images / 255
    embeddings = torch.mean(images, dim=-1)
    embeddings = torch.nn.functional.interpolate(
        embeddings[:, None, ...], size=(128, 150), mode="bilinear"
    )[:, 0, ...]
    # turn into original scale
    embeddings_hat = embeddings * (max_value - min_value) + min_value
    return embeddings_hat

class EncodecProcessor:
    def __init__(self, sample_rate):

        self.sample_rate = sample_rate

        self.model = EncodecModel.encodec_model_48khz()
        self.model.set_target_bandwidth(24.0)

    def encode(self, wav, input_sample_rate):
        # Load and pre-process the audio waveform
        wav = wav.unsqueeze(0)
        wav = convert_audio(
            wav, input_sample_rate, self.sample_rate, self.model.channels
        )
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        return encoded_frames

    def decode(self, encoded_frames):
        reconstructed_wav = self.model.decode(encoded_frames)
        return reconstructed_wav

    def encode_wo_quantization(self, wav, input_sample_rate):
        # Load and pre-process the audio waveform
        wav = wav.unsqueeze(0)
        wav = convert_audio(
            wav, input_sample_rate, self.sample_rate, self.model.channels
        )
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            embeddings = self.model.encoder(wav)
        return embeddings

    def decode_embeddings(self, embeddings):
        reconstructed_wav = self.model.decoder(embeddings)
        return reconstructed_wav

    # def crop(self, encoded_frames, n):
    #     for frame_index in range(len(encoded_frames)):
    #         encoded_frames[frame_index] = list(encoded_frames[frame_index])
    #         encoded_frames[frame_index][0][:, :, n:] = 0
    #     return encoded_frames

    def quantize(self, encoded_frames, n):
        for frame_index in range(len(encoded_frames)):
            encoded_frames[frame_index] = list(encoded_frames[frame_index])
            encoded_frames[frame_index][0] = encoded_frames[frame_index][0][:, :n, :]
        return encoded_frames


# play_audio(wav, model.sample_rate)
# play_audio(reconstructed_wav, model.sample_rate)

# %%
