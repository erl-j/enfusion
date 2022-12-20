# credit to @caillonantoine from https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py
import torch


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def safe_log(x):
    return torch.log(x + 1e-7)


def multiscale_loss(s, y, scales, overlap, lin_weight=1, log_weight=1):
    ori_stft = multiscale_fft(
        s,
        scales=scales,
        overlap=overlap,
    )
    rec_stft = multiscale_fft(
        y,
        scales=scales,
        overlap=overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss * lin_weight + log_loss * log_weight
    return loss