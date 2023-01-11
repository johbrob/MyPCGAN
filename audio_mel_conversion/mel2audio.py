from neural_networks.modules import weights_init, ResnetBlock, WNConv1d, WNConvTranspose1d
from librosa.filters import mel as librosa_mel_fn
from librosa.feature.inverse import mel_to_audio
from librosa.feature import melspectrogram
import torch.nn.functional as F
from abc import abstractmethod
from typing import Union
import torch.nn as nn
import numpy as np
import local_vars
import torch


class Mel2Audio:
    def __init__(self, audio_mel_config):
        self.n_fft = audio_mel_config.n_fft
        self.sample_rate = audio_mel_config.sample_rate
        self.hop_length = audio_mel_config.hop_length
        self.win_length = audio_mel_config.win_length
        self.n_mels = audio_mel_config.n_mels
        self.center = audio_mel_config.center

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    # TODO: revert these
    # def _get_pad_length(self):
    #     return (self.n_fft - self.hop_length) // 2
    #
    # def output_shape(self, audio):
    #     padded_audio_length = audio.shape[-1] + 2 * self._get_pad_length()
    #     return self.n_mels, int(np.ceil((padded_audio_length - self.win_length) // self.hop_length) + 1) - 4


class LibRosaMel2Audio(Mel2Audio):
    def __init__(self, config):
        super().__init__(config)


    def __call__(self, mel):
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


class CustomMel2Audio(Mel2Audio, nn.Module):
    def __init__(self, config):
        super().__init__(config)
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [nn.ReflectionPad1d(3), WNConv1d(self.input_size, mult * self.ngf, kernel_size=7, padding=0)]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [nn.LeakyReLU(0.2), WNConvTranspose1d(mult * self.ngf, mult * self.ngf // 2, kernel_size=r * 2,
                                                           stride=r, padding=r // 2 + r % 2, output_padding=r % 2, )]

            for j in range(self.n_residual_layers):
                model += [ResnetBlock(mult * self.ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), WNConv1d(self.ngf, 1, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        # return self.model(x)
        for m in self.model:
            x = m(x)

        return x