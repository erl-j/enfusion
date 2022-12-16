#%%
from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch


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
