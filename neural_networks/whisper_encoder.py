from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
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
        self.processor = AutoProcessor.from_pretrained(config.model_size.value)
        self.whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_size.value).get_encoder().to(device)
        self.whisper_encoder._freeze_parameters()
        self.embedding_dim = self.whisper_encoder.embed_positions.embedding_dim
        self.sampling_rate = config.sampling_rate
        self.device = device

    def __call__(self, data):
        input_features = [audio.squeeze().cpu().numpy() for audio in data]

        print(data.shape)
        print(len(input_features), input_features[0].shape)
        input_features = self.processor(input_features, return_tensors="pt",
                                        sampling_rate=self.sampling_rate).input_features
        return self.whisper_encoder(input_features.to(self.device)).last_hidden_state
