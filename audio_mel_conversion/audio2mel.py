import pandas
from transformers import AutoFeatureExtractor
from librosa.filters import mel as librosa_mel_fn
from librosa.feature import melspectrogram
import torch.nn.functional as F
from abc import abstractmethod
from typing import Union
import numpy as np
import torch


class Audio2Mel:
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

    def _get_pad_length(self):
        return (self.n_fft - self.hop_length) // 2

    def output_shape(self, audio):
        padded_audio_length = audio.shape[-1] + 2 * self._get_pad_length()
        return self.n_mels, int(np.ceil((padded_audio_length - self.win_length) // self.hop_length) + 1) - 4


class LibRosaAudio2Mel(Audio2Mel):
    def __init__(self, audio_mel_config):
        super().__init__(audio_mel_config)

    def __call__(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
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


class LibRosaAudio2Mel2(Audio2Mel, torch.nn.Module):
    def __init__(self, config):
        super().__init__(config)

        window = torch.hann_window(config.win_length).float()
        mel_basis = librosa_mel_fn(sr=config.sampling_rate, n_fft=config.n_fft, n_mels=config.n_mel_channels,
                                   fmin=config.mel_fmin, fmax=config.mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

    def __call__(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        p = self._get_pad_length()
        audio = F.pad(audio, (p, p), "reflect")  # audio: bsz, seq_len

        # bsz x L -> bsz x new_L x frames x 2 where
        # new_L = L/2 + 1
        # frames = ceil((L - (window_length - 1) - 1) / hop_length)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                         window=self.window, center=False, return_complex=True)
        magnitude = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


class WhisperAudio2Mel:
    def __init__(self, config):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_size.value)
        self.sampling_rate = config.sampling_rate

    def __call__(self, data):
        data = [audio.squeeze().cpu().numpy() for audio in data]
        return self.feature_extractor(data, return_tensors="pt", sampling_rate=self.sampling_rate).input_features

    def output_shape(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        shape = self.feature_extractor(audio, return_tensors="pt",
                                       sampling_rate=self.sampling_rate).input_features.shape
        return shape[1], shape[2]
        # TODO: fix this so its not just hard coded
        # return [80, 3000]
