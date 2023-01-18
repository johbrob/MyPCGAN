import torch
import torch.nn.functional as F
from librosa.filters import mel


def get_components(raw_speech, sampling_rate, n_fft, hop_length, center, n_mels, win_length):
    if raw_speech.dim() == 1:
        raw_speech = raw_speech.unsqueeze(dim=0)

    window = torch.hann_window(win_length).float()
    mel_basis = mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float()
    fft = torch.stft(raw_speech, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                     window=window, center=center, return_complex=True)
    magnitude = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
    mel_output = torch.matmul(mel_basis, magnitude)

    frames = None

    return window, frames, fft, magnitude, mel_basis, mel_output


def melgan_audio2mel(raw_speech, sampling_rate, n_fft, hop_length, center, n_mels, win_length):

    window = torch.hann_window(win_length).float()
    mel_basis = mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()

    if raw_speech.dim() == 1:
        raw_speech = raw_speech.unsqueeze(dim=0)

    # p = _get_pad_length()
    # raw_speech = F.pad(raw_speech, (p, p), "reflect")  # raw_speech: bsz, seq_len

    # bsz x L -> bsz x new_L x frames x 2 where
    # new_L = L/2 + 1
    # frames = ceil((L - (window_length - 1) - 1) / hop_length)
    fft = torch.stft(raw_speech, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                     window=window, center=center, return_complex=True)
    # tmp_window = self.window.numpy()
    # tmp_stft = fft.numpy()
    # tmp_abs_magnitude = np.abs(tmp_stft)
    # tmp_square_abs_magnitude = tmp_abs_magnitude ** 2
    magnitude = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
    mel_output = torch.matmul(mel_basis, magnitude)
    log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
    # return log_mel_spec  # , mel_output
    print('WARNING: MELGAN-AUDIO2MEL outputs mel_output instead of log_mel_spec')
    return mel_output
    # return input_features, mel_spec, pre_final_conversions
