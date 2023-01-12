import enum
from datasets.audiomnist import AudioMNIST, create_audiomnist
from datasets.crema_d import CremaD, create_crema_d
from datasets.audio_dataset import AudioDataset


class AvailableDatasets(enum.Enum):
    AudioMNIST = 1
    CremaD = 2


def get_dataset(dataset_name: AvailableDatasets, n_train_samples: int = None, n_test_samples: int = None) -> AudioDataset:
    if dataset_name == AvailableDatasets.AudioMNIST:
        return AudioMNIST.load(n_train_samples=n_train_samples, n_test_samples=n_test_samples)
    elif dataset_name == AvailableDatasets.CremaD:
        return CremaD.load(n_train_samples=n_train_samples, n_test_samples=n_test_samples)
