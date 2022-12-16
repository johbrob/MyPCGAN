import enum
from datasets.audiomnist import AudioMNIST, create_audiomnist
from datasets.crema_d import CremaD, create_crema_d


class AvailableDatasets(enum.Enum):
    AudioMNIST = 1
    CremaD = 2


