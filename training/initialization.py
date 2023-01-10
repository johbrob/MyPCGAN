from datasets import AvailableDatasets, AudioMNIST, CremaD
from training.training import training_loop
from neural_networks.audio_mel import AudioMelConverter, CustomAudioMelConverter
from torch.utils.data import DataLoader
import numpy as np
import torch
import utils
import log

dataset_to_name = {
    AvailableDatasets.AudioMNIST: 'DAIS - AudioMNIST',
    AvailableDatasets.CremaD: 'DAIS - Crema-D'
}


class TrainingConfig:
    def __init__(self, run_name='tmp', dataset=AvailableDatasets.CremaD, train_batch_size=128, test_batch_size=128, do_train_shuffle=True,
                 do_test_shuffle=True, train_num_workers=2, test_num_workers=2, save_interval=5, checkpoint_interval=1,
                 updates_per_evaluation=50, updates_per_train_log_commit=10, gradient_accumulation=1, epochs=2,
                 n_samples=1, do_log=True, librosa_audio_mel=False, deterministic=False):
        self.run_name = run_name + '_' + self.random_id()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.do_train_shuffle = do_train_shuffle
        self.do_test_shuffle = do_test_shuffle
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers
        self.save_interval = save_interval
        self.checkpoint_interval = checkpoint_interval
        self.updates_per_evaluation = updates_per_evaluation
        self.updates_per_train_log_commit = updates_per_train_log_commit
        self.gradient_accumulation = gradient_accumulation

        self.epochs = epochs
        self.n_samples = n_samples
        self.do_log = do_log
        self.librosa_audio_mel = librosa_audio_mel
        self.deterministic = deterministic

    def random_id(self):
        return str(np.random.randint(0, 9, 7))[1:-1].replace(' ', '')


def set_seed(seed: int = 42) -> None:
    import random
    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def create_architecture_from_config(config, device):
    return config.architecture(config, device)


def create_model_from_config(config):
    return config.model(**vars(config.args))


def get_dataset(dataset_name):
    if dataset_name == AvailableDatasets.AudioMNIST:
        return AudioMNIST.load()
    elif dataset_name == AvailableDatasets.CremaD:
        return CremaD.load()


def init_training(experiment_setup, device):
    # load dataset

    train_data, test_data = get_dataset(experiment_setup.training.dataset)

    # print split ratios
    train_female_speakar_ratio = sum(1 - train_data.gender_idx) / len(train_data.gender_idx)
    test_female_speakar_ratio = sum(1 - test_data.gender_idx) / len(test_data.gender_idx)
    print(f'Training set contains {train_data.n_speakers} speakers with {int(100 * train_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(train_data.gender_idx)}')
    print(f'Test set contains {test_data.n_speakers} speakers with {int(100 * test_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(test_data.gender_idx)}')

    # init dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=experiment_setup.training.train_batch_size,
                              num_workers=experiment_setup.training.train_num_workers,
                              shuffle=experiment_setup.training.do_train_shuffle)
    test_loader = DataLoader(dataset=test_data, batch_size=experiment_setup.training.test_batch_size,
                             num_workers=experiment_setup.training.test_num_workers,
                             shuffle=experiment_setup.training.do_test_shuffle)

    # init Audio/Mel-converter
    if experiment_setup.training.librosa_audio_mel:
        audio_mel_converter = AudioMelConverter(experiment_setup.audio_mel)
    else:
        audio_mel_converter = CustomAudioMelConverter(experiment_setup.audio_mel)

    # match model sizes with mel sizes and number of labels and secrets
    image_width, image_height = audio_mel_converter.output_shape(train_data[0][0])
    experiment_setup.architecture.image_width = image_width
    experiment_setup.architecture.image_height = image_height
    experiment_setup.architecture.n_genders = train_data.n_genders
    experiment_setup.architecture.n_labels = train_data.n_labels

    if experiment_setup.training.deterministic:
        set_seed(0)

    # init models etc etc
    architecture = create_architecture_from_config(experiment_setup.architecture, device)

    # init wandb
    if experiment_setup.training.do_log:
        log.init(utils.nestedConfigs2dict(experiment_setup), project=dataset_to_name[experiment_setup.training.dataset],
                 run_name=experiment_setup.training.run_name)

    # start training loop
    training_loop(train_loader, test_loader, experiment_setup.training, architecture, audio_mel_converter,
                  experiment_setup.audio_mel.sample_rate, device)
