import IPython.display as ipd

#%%
def play_audio(wav, sr):
    wav = wav.squeeze(0).detach().cpu().numpy()
    ipd.display(ipd.Audio(wav, rate=sr))

