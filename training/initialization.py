from nn.models import load_modified_AlexNet, load_modified_ResNet, UNetFilter
from DataManaging.AudioDatasets import AudioDataset
from training.training import training_loop
# from nn.modules import AudioMelConverter, CustomAudioMelConverter
from nn.audio_mel import AudioMelConverter, CustomAudioMelConverter
from torch.utils.data import DataLoader
from loss_compiling import HLoss
import numpy as np
import local_vars
import torch
import utils
import log


class TrainingConfig:
    def __init__(self, lr, run_name='tmp', train_batch_size=128, test_batch_size=128, train_num_workers=2,
                 test_num_workers=2, save_interval=1, checkpoint_interval=1, updates_per_evaluation=5,
                 gradient_accumulation=1, epochs=2, n_samples=5, do_log=True, librosa_audio_mel=False):
        self.run_name = run_name + '_' + self.random_id()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers
        self.save_interval = save_interval
        self.checkpoint_interval = checkpoint_interval
        self.updates_per_evaluation = updates_per_evaluation
        self.gradient_accumulation = gradient_accumulation
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.do_log = do_log
        self.librosa_audio_mel = librosa_audio_mel

    def random_id(self):
        return str(np.random.randint(0, 9, 7))[1:-1].replace(' ', '')


def init_models(experiment_config, image_width, image_height, n_labels, n_genders, device):
    training_config = experiment_config.training_config
    unet_config = experiment_config.unet_config

    label_classifier = load_modified_ResNet(n_labels).to(device).eval()
    secret_classifier = load_modified_ResNet(n_genders).to(device).eval()

    # TODO: Fix loading state dicts
    # tmp_lc = torch.load(local_vars.PWD + 'nn/pretrained_weights/best_digit_alexnet_spectrograms_epoch_26.pt',
    #                     map_location=torch.device('cpu'))
    # label_classifier.load_state_dict(tmp_lc)
    #
    # secret_classifier.load_state_dict(
    #     torch.load(local_vars.PWD + 'nn/pretrained_weights/best_gender_alexnet_epoch_29.pt'),
    #     map_location=torch.device('cpu'))

    loss_funcs = {'distortion': torch.nn.L1Loss(), 'entropy': HLoss(), 'adversarial': torch.nn.CrossEntropyLoss(),
                  'adversarial_rf': torch.nn.CrossEntropyLoss()}

    filter_gen = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=unet_config.kernel_size,
                            image_width=image_width, image_height=image_height, noise_dim=unet_config.noise_dim,
                            n_classes=n_genders, embedding_dim=unet_config.embedding_dim, use_cond=False).to(
        device)
    filter_disc = load_modified_AlexNet(n_genders).to(device)
    secret_gen = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=unet_config.kernel_size,
                            image_width=image_width, image_height=image_height, noise_dim=unet_config.noise_dim,
                            n_classes=n_genders, embedding_dim=unet_config.embedding_dim, use_cond=False).to(
        device)
    secret_disc = load_modified_AlexNet(n_genders + 1).to(device)

    models = {
        'filter_gen': filter_gen, 'filter_disc': filter_disc, 'secret_gen': secret_gen, 'secret_disc': secret_disc,
        'label_classifier': label_classifier, 'secret_classifier': secret_classifier
    }

    optimizers = {
        'filter_gen': torch.optim.Adam(filter_gen.parameters(), training_config.lr['filter_gen'], betas=(0.5, 0.9)),
        'filter_disc': torch.optim.Adam(filter_disc.parameters(), training_config.lr['filter_gen'], betas=(0.5, 0.9)),
        'secret_gen': torch.optim.Adam(secret_gen.parameters(), training_config.lr['secret_gen'], betas=(0.5, 0.9)),
        'secret_disc': torch.optim.Adam(secret_disc.parameters(), training_config.lr['secret_disc'], betas=(0.5, 0.9))
    }
    return loss_funcs, models, optimizers


def init_training(dataset_name, experiment_settings, device):
    training_config, audio_mel_config, unet_config, loss_config = experiment_settings.get_configs()

    train_data, test_data = AudioDataset.load()
    train_loader = DataLoader(train_data, training_config.train_batch_size,
                              num_workers=training_config.train_num_workers, shuffle=True)
    test_loader = DataLoader(test_data, training_config.test_batch_size,
                             num_workers=training_config.test_num_workers, shuffle=True)

    if training_config.librosa_audio_mel:
        audio_mel_converter = AudioMelConverter(audio_mel_config)
    else:
        audio_mel_converter = CustomAudioMelConverter(audio_mel_config)
    image_width, image_height = audio_mel_converter.output_shape(train_data[0][0])

    loss_funcs, models, optimizers = init_models(experiment_settings, image_width, image_height, train_data.n_labels,
                                                 train_data.n_genders, device)

    if training_config.do_log:
        log.init(utils.nestedConfigs2dict(experiment_settings), run_name=training_config.run_name)

    training_loop(train_loader, test_loader, training_config, models, optimizers, audio_mel_converter, loss_funcs,
                  loss_config.gamma, loss_config.use_entropy_loss, audio_mel_config.sample_rate, device)
