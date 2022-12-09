import torch


def preprocess_spectrograms(spectrograms):
    means = torch.mean(spectrograms, dim=(1, 2), keepdim=True)
    stds = torch.std(spectrograms, dim=(1, 2), keepdim=True)
    normalized_spectrograms = (spectrograms - means) / (3 * stds + 1e-6)
    clipped_spectrograms = torch.clamp(normalized_spectrograms, -1, 1)

    return clipped_spectrograms, means, stds