import torch.nn.functional
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, WhisperFeatureExtractor
import enum


class WhisperSize(enum.Enum):
    TINY = "openai/whisper-tiny"
    BASE = "openai/whisper-base"
    SMALL = "openai/whisper-small"
    MEDIUM = "openai/whisper-medium"
    LARGE = "openai/whisper-large"


class WhisperEncoderConfig:
    def __init__(self, model_size, sampling_rate):
        self.model_size = model_size
        self.sampling_rate = sampling_rate


class WhisperEncoder:
    def __init__(self, config, device):
        # self.processor = AutoProcessor.from_pretrained(config.model_size.value)
        self.whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_size.value).get_encoder().to(device)
        self.whisper_encoder._freeze_parameters()
        self.embedding_dim = self.whisper_encoder.embed_positions.embedding_dim
        self.sampling_rate = config.sampling_rate
        self.device = device

    def __call__(self, data, *args, **kwargs):
        if data.shape[0] == 1 and data.dim() == 3:
            data = data.repeat(2, 1, 1)
            output = self.whisper_encoder(data.to(self.device), *args, **kwargs).last_hidden_state
            output = output[0]
        else:
            output = self.whisper_encoder(data.to(self.device), *args, **kwargs).last_hidden_state
        return output

import GPUtil
class WhisperEncoderForMelGanMels(WhisperEncoder):
    def __init__(self, config, device):
        super().__init__(config, device)


    def __call__(self, data, *args, **kwargs):
        GPUtil.showUtilization()
        data = self._pad(data)
        GPUtil.showUtilization()
        data = self._preprocess(data)
        GPUtil.showUtilization()
        if data.shape[0] == 1 and data.dim() == 3:
            data = data.repeat(2, 1, 1)
            output = self.whisper_encoder(data.to(self.device), *args, **kwargs).last_hidden_state[:, 0, :]
            output = output[0]
        else:
            # output = self.whisper_encoder(data.to(self.device), *args, **kwargs).last_hidden_state[:, 0, :]
            self.whisper_encoder(data.to(self.device), *args, **kwargs).last_hidden_state[:, 0, :]
            GPUtil.showUtilization()
            output = None
        return output

    def _pad(self, data):
        lth = data.shape[-1]
        p = 3000 - lth
        output = torch.nn.functional.pad(data, (0, p), "constant", 0)
        return output

    def _preprocess(self, data):
        log_spec = torch.maximum(data, data.amax() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec