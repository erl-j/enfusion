#%%

from encodec_processor import EncodecProcessor
import torch
from IPython.display import Audio
import torchaudio
import glob
# 

SAMPLE_RATE = 48000


fp=glob.glob("../synth_text_dataset/**/*.wav", recursive=True)[15]
sample,sr = torchaudio.load(fp)

sample=sample*0


# play
Audio(sample.detach().numpy(), rate=SAMPLE_RATE)

DURATION=4
#sample = torch.randn(1, SAMPLE_RATE*DURATION)

print(sample.shape)

encodec_processor = EncodecProcessor(SAMPLE_RATE)
embedding = encodec_processor.encode_wo_quantization(sample, SAMPLE_RATE)

print(embedding[0].shape)

waveform = encodec_processor.decode_embeddings(embedding)

print(waveform.shape)

# play
Audio(waveform.detach().numpy()[0], rate=SAMPLE_RATE)
# %%
