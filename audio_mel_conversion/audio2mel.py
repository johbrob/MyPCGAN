import pandas
from neural_networks.whisper_encoder import WhisperSize
from transformers.models.whisper import WhisperFeatureExtractor
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
        self.sampling_rate = audio_mel_config.sampling_rate
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

        mel = melspectrogram(y=audio, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length, center=False, n_mels=self.n_mels)
        if is_tensor:
            return torch.Tensor(mel).to(device)
        else:
            return mel


class MelGanAudio2Mel(Audio2Mel, torch.nn.Module):
    def __init__(self, config):
        # Audio2Mel.__init__(self, config)
        # torch.nn.Module.__init__(self)
        super().__init__(config)  # calls all parent classes up to Audio2Mel
        super(Audio2Mel, self).__init__()  # calls all parent classes from Audio2Mel to torch.nn.Module

        window = torch.hann_window(config.win_length).float()
        mel_basis = librosa_mel_fn(sr=config.sampling_rate, n_fft=config.n_fft, n_mels=config.n_mels,
                                   fmin=config.mel_fmin, fmax=config.mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

    def __call__(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        # print('WARNING: NO PADDING IN LIBROSA AUDIO2MEL')
        if audio.dim() == 1:
            audio = audio.unsqueeze(dim=0)

        p = self._get_pad_length()
        audio = F.pad(audio, (p, p), "reflect")  # audio: bsz, seq_len

        # bsz x L -> bsz x new_L x frames x 2 where
        # new_L = L/2 + 1
        # frames = ceil((L - (window_length - 1) - 1) / hop_length)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                         window=self.window, center=False, return_complex=True)
        # tmp_window = self.window.numpy()
        # tmp_stft = fft.numpy()
        # tmp_abs_magnitude = np.abs(tmp_stft)
        # tmp_square_abs_magnitude = tmp_abs_magnitude ** 2
        magnitude = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec  # , mel_output


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

    def not_padded_output_shape(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        shape = self.feature_extractor(audio, return_tensors="pt", padding=False,
                                       sampling_rate=self.sampling_rate).input_features.shape
        return shape[1], shape[2]


class CustomWhisperFeatureExtractor(WhisperFeatureExtractor):

    def __init__(self):
        super().__init__()

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
        implementation with 1e-5 tolerance.
        """
        window = np.hanning(self.n_fft + 1)[:-1]

        frames = self.fram_wave(waveform)
        stft = self.stft(frames, window=window)
        magnitudes = np.abs(stft[:, :-1]) ** 2

        filters = self.mel_filters
        mel_spec = filters @ magnitudes

        return mel_spec


class CustomWhisperAudio2Mel(WhisperAudio2Mel):
    def __init__(self, config):
        self.feature_extractor = CustomWhisperFeatureExtractor()
        self.sampling_rate = config.sampling_rate

    def __call__(self, data):
        data = [audio.squeeze().cpu().numpy() for audio in data]
        return self.feature_extractor(data, return_tensors="pt", padding=False,
                                      sampling_rate=self.sampling_rate).input_features


    @staticmethod
    def log_mels(mel):
        log_spec = np.log10(np.clip(mel, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
