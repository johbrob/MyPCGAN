from nn.modules import weights_init, ResnetBlock, WNConv1d, WNConvTranspose1d
from librosa.filters import mel as librosa_mel_fn
from librosa.feature.inverse import mel_to_audio
from librosa.feature import melspectrogram
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import local_vars
import torch


class AudioMelConfig:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sample_rate=8000, n_mels=80, center=False,
                 mel_fmin=0.0, mel_fmax=None):
        self.n_fft = n_fft
        self.sample_rate = sample_rate          # 22050
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.center = center
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax


class AudioMelConverter():
    def __init__(self, audio_mel_config):
        self.n_fft = audio_mel_config.n_fft
        self.sample_rate = audio_mel_config.sample_rate
        self.hop_length = audio_mel_config.hop_length
        self.win_length = audio_mel_config.win_length
        self.n_mels = audio_mel_config.n_mels
        self.center = audio_mel_config.center

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


class CustomAudio2Mel(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, n_mel_channels=80,
                 mel_fmin=0.0, mel_fmax=None):
        super().__init__()
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = self._get_pad_length()
        audio = F.pad(audio, (p, p), "reflect")         # audio: bsz, seq_len

        # bsz x L -> bsz x new_L x frames x 2 where
        # new_L = L/2 + 1
        # frames = ceil((L - (window_length - 1) - 1) / hop_length)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                         window=self.window, center=False, return_complex=True)
        magnitude = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

    def _get_pad_length(self):
        return (self.n_fft - self.hop_length) // 2

    def output_shape(self, audio):
        padded_audio_length = audio.shape[-1] + 2 * self._get_pad_length()
        return self.n_mel_channels, int(np.ceil((padded_audio_length - self.win_length) // self.hop_length) + 1)


class CustomMel2Audio(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [nn.ReflectionPad1d(3), WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0)]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [nn.LeakyReLU(0.2), WNConvTranspose1d(mult * ngf, mult * ngf // 2, kernel_size=r * 2,
                                                           stride=r, padding=r // 2 + r % 2, output_padding=r % 2, )]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), WNConv1d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        # return self.model(x)
        for m in self.model:
            x = m(x)

        return x


class CustomAudioMelConverter():
    def __init__(self, audio_mel_config):
        self.n_fft = audio_mel_config.n_fft
        self.sample_rate = audio_mel_config.sample_rate
        self.hop_length = audio_mel_config.hop_length
        self.win_length = audio_mel_config.win_length
        self.n_mels = audio_mel_config.n_mels
        self.center = audio_mel_config.center
        self.mel_fmin = audio_mel_config.mel_fmin
        self.mel_fmax = audio_mel_config.mel_fmax

        self.audio2mel_func = CustomAudio2Mel(n_fft=audio_mel_config.n_fft, hop_length=audio_mel_config.hop_length,
                                              win_length=audio_mel_config.win_length,
                                              sampling_rate=audio_mel_config.sample_rate,
                                              n_mel_channels=audio_mel_config.n_mels,
                                              mel_fmin=audio_mel_config.mel_fmin, mel_fmax=audio_mel_config.mel_fmax)
        self.mel2audio_func = CustomMel2Audio(audio_mel_config.n_mels, ngf=32, n_residual_layers=3)
        self.mel2audio_func.load_state_dict(
            torch.load(local_vars.PWD + 'nn/pretrained_weights/best_netG_epoch_2120.pt', map_location=torch.device('cpu')))

    def audio2mel(self, audio):
        return self.audio2mel_func(audio)

    def mel2audio(self, mel):
        return self.mel2audio_func(mel)

    def output_shape(self, audio):
        return self.audio2mel_func.output_shape(audio)
