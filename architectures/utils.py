import torch


def _normalize(spectrogram):
    mean = torch.mean(spectrogram, dim=(1, 2), keepdim=True)
    std = torch.std(spectrogram, dim=(1, 2), keepdim=True)
    return (spectrogram - mean) / (3 * std + 1e-6), mean, std


def preprocess_spectrograms(spectrograms):
    spectrogram, mean, std = _normalize(spectrograms)
    spectrogram = torch.clamp(spectrogram, -1, 1)

    return spectrogram, mean, std


def create_model_from_config(config):
    return config.model(**vars(config.args))


