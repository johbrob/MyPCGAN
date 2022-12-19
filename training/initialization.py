from datasets import AvailableDatasets, AudioMNIST, CremaD
from training.training import training_loop
from neural_networks.audio_mel import AudioMelConverter, CustomAudioMelConverter
from torch.utils.data import DataLoader
from loss_compiling import HLoss
import numpy as np
import torch
import utils
import log

dataset_to_name = {
    AvailableDatasets.AudioMNIST: 'DAIS - AudioMNIST',
    AvailableDatasets.CremaD: 'DAIS - Crema-D'
}


class TrainingConfig:
    def __init__(self, lr=None, run_name='tmp', train_batch_size=128, test_batch_size=128, do_train_shuffle=True,
                 do_test_shuffle=True, train_num_workers=2, test_num_workers=2, save_interval=5, checkpoint_interval=1,
                 updates_per_evaluation=50, updates_per_train_log_commit=10, gradient_accumulation=1, epochs=2,
                 n_generated_samples=1, do_log=True, librosa_audio_mel=False, deterministic=False):
        self.run_name = run_name + '_' + self.random_id()
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
        if lr is None:
            self.lr = {'filter_gen': 0.0001, 'filter_disc': 0.0004, 'secret_gen': 0.0001, 'secret_disc': 0.0004}
        else:
            self.lr = lr
        self.epochs = epochs
        self.n_generated_samples = n_generated_samples
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


def create_model_from_config(config):
    return config.model(**vars(config.args))


def init_models(experiment_setup, device):
    if experiment_setup.training.deterministic:
        set_seed(0)

    # gender_classifier = ResNet18(2, 'relu').to(device)
    # gender_classifier.model.load_state_dict(torch.load("neural_networks/pretrained_weights/best_gender_model.pt", map_location=torch.device('cpu')))  # privacy

    label_classifier = create_model_from_config(experiment_setup.label_classifier).to(device)
    if experiment_setup.label_classifier.pretrained_path:
        label_classifier.model.load_state_dict(
            torch.load(experiment_setup.label_classifier.pretrained_path, map_location=torch.device('cpu')))

    secret_classifier = create_model_from_config(experiment_setup.secret_classifier).to(device)
    if experiment_setup.secret_classifier.pretrained_path:
        secret_classifier.model.load_state_dict(
            torch.load(experiment_setup.secret_classifier.pretrained_path, map_location=torch.device('cpu')))

    loss_funcs = {'distortion': torch.nn.L1Loss(), 'entropy': HLoss(), 'adversarial': torch.nn.CrossEntropyLoss(),
                  'adversarial_rf': torch.nn.CrossEntropyLoss()}

    models = {
        'filter_gen': create_model_from_config(experiment_setup.filter_gen).to(device),
        'filter_disc': create_model_from_config(experiment_setup.filter_disc).to(device),
        'secret_gen': create_model_from_config(experiment_setup.secret_gen).to(device),
        'secret_disc': create_model_from_config(experiment_setup.secret_disc).to(device),
        'label_classifier': label_classifier,
        'secret_classifier': secret_classifier
    }

    lr = experiment_setup.training.lr

    optimizers = {
        'filter_gen': torch.optim.Adam(models['filter_gen'].parameters(), lr['filter_gen'], betas=(0.5, 0.9)),
        'filter_disc': torch.optim.Adam(models['filter_disc'].parameters(), lr['filter_disc'], betas=(0.5, 0.9)),
        'secret_gen': torch.optim.Adam(models['secret_gen'].parameters(), lr['secret_gen'], betas=(0.5, 0.9)),
        'secret_disc': torch.optim.Adam(models['secret_disc'].parameters(), lr['secret_disc'], betas=(0.5, 0.9))
    }
    return loss_funcs, models, optimizers


def get_dataset(dataset_name):
    if dataset_name == AvailableDatasets.AudioMNIST:
        return AudioMNIST.load()
    elif dataset_name == AvailableDatasets.CremaD:
        return CremaD.load()


def init_training(dataset, experiment_setup, device):
    train_data, test_data = get_dataset(dataset)

    train_female_speakar_ratio = sum(1 - train_data.gender_idx) / len(train_data.gender_idx)
    test_female_speakar_ratio = sum(1 - test_data.gender_idx) / len(test_data.gender_idx)
    print(f'Training set contains {train_data.n_speakers} speakers with {int(100 * train_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(train_data.gender_idx)}')
    print(f'Test set contains {test_data.n_speakers} speakers with {int(100 * test_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(test_data.gender_idx)}')

    train_loader = DataLoader(dataset=train_data, batch_size=experiment_setup.training.train_batch_size,
                              num_workers=experiment_setup.training.train_num_workers,
                              shuffle=experiment_setup.training.do_train_shuffle)
    test_loader = DataLoader(dataset=test_data, batch_size=experiment_setup.training.test_batch_size,
                             num_workers=experiment_setup.training.test_num_workers,
                             shuffle=experiment_setup.training.do_test_shuffle)

    if experiment_setup.training.librosa_audio_mel:
        audio_mel_converter = AudioMelConverter(experiment_setup.audio_mel)
    else:
        audio_mel_converter = CustomAudioMelConverter(experiment_setup.audio_mel)
    image_width, image_height = audio_mel_converter.output_shape(train_data[0][0])

    experiment_setup.filter_gen.args.image_width = image_width
    experiment_setup.filter_gen.args.image_height = image_height
    experiment_setup.filter_gen.args.n_classes = train_data.n_genders
    experiment_setup.filter_disc.args.n_classes = train_data.n_genders
    experiment_setup.secret_gen.args.image_width = image_width
    experiment_setup.secret_gen.args.image_height = image_height
    experiment_setup.secret_gen.args.n_classes = train_data.n_genders
    experiment_setup.secret_disc.args.n_classes = train_data.n_genders + 1
    experiment_setup.label_classifier.args.n_classes = train_data.n_genders
    experiment_setup.secret_classifier.args.n_classes = train_data.n_labels

    loss_funcs, models, optimizers = init_models(experiment_setup, device)

    if experiment_setup.training.do_log:
        log.init(utils.nestedConfigs2dict(experiment_setup), project=dataset_to_name[dataset],
                 run_name=experiment_setup.training.run_name)

    training_loop(train_loader, test_loader, experiment_setup.training, models, optimizers, audio_mel_converter,
                  loss_funcs, experiment_setup.loss, experiment_setup.audio_mel.sample_rate, device)
