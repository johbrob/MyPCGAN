import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


# class Audio2Mel(nn.Module):
#     def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, n_mel_channels=80,
#                  mel_fmin=0.0, mel_fmax=None):
#         super().__init__()
#         ##############################################
#         # FFT Parameters                             #
#         ##############################################
#         window = torch.hann_window(win_length).float()
#         mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
#         mel_basis = torch.from_numpy(mel_basis).float()
#         self.register_buffer("mel_basis", mel_basis)
#         self.register_buffer("window", window)
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.sampling_rate = sampling_rate
#         self.n_mel_channels = n_mel_channels
#
#     def forward(self, audio):
#         p = self._get_pad_length()
#         audio = F.pad(audio, (p, p), "reflect").squeeze(1)
#
#         # bsz x L -> bsz x new_L x frames x 2 where
#         # new_L = L/2 + 1
#         # frames = ceil((L - (window_length - 1) - 1) / hop_length)
#         fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
#                          window=self.window, center=False, return_complex=True)
#         magnitude = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
#         mel_output = torch.matmul(self.mel_basis, magnitude)
#         log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
#         return log_mel_spec
#
#     def _get_pad_length(self):
#         return (self.n_fft - self.hop_length) // 2
#
#     def output_shape(self, audio):
#         padded_audio_length = audio.shape[-1] + 2 * self._get_pad_length()
#         return self.n_mel_channels, int(np.ceil((padded_audio_length - self.win_length) // self.hop_length) + 1)


# class MelGanGenerator(nn.Module):
#     def __init__(self, input_size, ngf, n_residual_layers):
#         super().__init__()
#         ratios = [8, 8, 2, 2]
#         self.hop_length = np.prod(ratios)
#         mult = int(2 ** len(ratios))
#
#         model = [nn.ReflectionPad1d(3), WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0)]
#
#         # Upsample to raw audio scale
#         for i, r in enumerate(ratios):
#             model += [nn.LeakyReLU(0.2), WNConvTranspose1d(mult * ngf, mult * ngf // 2, kernel_size=r * 2,
#                                                            stride=r, padding=r // 2 + r % 2, output_padding=r % 2, )]
#
#             for j in range(n_residual_layers):
#                 model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]
#
#             mult //= 2
#
#         model += [nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), WNConv1d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#         self.apply(weights_init)
#
#     def forward(self, x):
#         # return self.model(x)
#         for m in self.model:
#             x = m(x)
#
#         return x


from torchaudio import transforms
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_audio


# waveform, sample_rate = torchaudio.load("test.wav", normalize=True)

class AudioMelConverter():
    def __init__(self, audio_mel_config):
        self.n_fft = audio_mel_config.n_fft
        self.sample_rate = audio_mel_config.sample_rate
        self.hop_length = audio_mel_config.hop_length
        self.win_length = audio_mel_config.win_length
        self.n_mels = audio_mel_config.n_mels
        self.center = audio_mel_config.center

        # self.audio2mel = lambda audio: melspectrogram(y=audio, sr=self.sample_rate, n_fft=self.n_fft,
        #                                               hop_length=self.hop_length,
        #                                               win_length=self.win_length,
        #                                               center=False, n_mels=self.n_mels)
        #
        # self.mel2audio = lambda mel: mel_to_audio(M=mel, sr=self.sample_rate, n_fft=self.n_fft,
        #                                           hop_length=self.hop_length, win_length=self.win_length, center=False,
        #                                           n_mels=self.n_mels)

    def audio2mel(self, audio):
        device = 'cpu'
        is_tensor = isinstance(audio, torch.Tensor)
        if is_tensor:
            device = audio.device
            audio = audio.cpu().numpy()

        mel = melspectrogram(y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length, center=False, n_mels=self.n_mels)
        if is_tensor:
            return torch.Tensor(mel).to(device)
        else:
            return mel

    def mel2audio(self, mel):
        device = 'cpu'
        is_tensor = isinstance(mel, torch.Tensor)
        if is_tensor:
            device = mel.device
            mel = mel.cpu().numpy()
        audio = mel_to_audio(M=mel, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length, center=False)
        if is_tensor:
            return torch.Tensor(audio).to(device)
        else:
            return audio

    def _get_pad_length(self):
        return (self.n_fft - self.hop_length) // 2

    def output_shape(self, audio):
        padded_audio_length = audio.shape[-1] + 2 * self._get_pad_length()
        return self.n_mels, int(np.ceil((padded_audio_length - self.win_length) // self.hop_length) + 1) - 4

# class Mel2Audio(nn.Module):
#     def __init__(self, n_fft, sample_rate, win_length, hop_length, window_fn, normalized, center):
#         super().__init__()
#         inverse_melscale = transforms.InverseMelScale(n_stft=n_fft // 2 + 1, sample_rate=sample_rate)
#         inverse_spectrogram = transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
#                                                             window_fn=window_fn, normalized=normalized,
#                                                             center=center)
#
#     def forward(self, mel_spectrogram):
#         spectrogram = self.inverse_melscale(mel_spectrogram)
#         return self.inverse_spectrogram(spectrogram)
