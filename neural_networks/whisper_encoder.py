from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoFeatureExtractor
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