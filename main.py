from nn.models import load_modified_AlexNet, load_modified_ResNet, UNetFilter, AudioNet
from DataManaging.AudioDatasets import AudioDataset
from nn.modules import Audio2Mel, MelGanGenerator
from metrics_compiling import compute_metrics, compile_metrics, aggregate_metrics
from loss_compiling import compute_losses, HLoss
from plotting import comparison_plot_pcgan
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import local_vars
import configs
import torch
import utils
import tqdm
import glob
import time
import log
import os

from torch.autograd import Variable
from torch import LongTensor, FloatTensor


def preprocess_spectrograms(spectrograms):
    """
    Preprocess spectrograms according to approach in WaveGAN paper
    Args:
        spectrograms (torch.FloatTensor) of size (batch_size, mel_bins, time_frames)

    """
    # Remove last time segment
    # spectrograms = spectrograms[:,:,:-1]
    # Normalize to zero mean unit variance, clip above 3 std and rescale to [-1,1]
    means = torch.mean(spectrograms, dim=(1, 2), keepdim=True)
    stds = torch.std(spectrograms, dim=(1, 2), keepdim=True)
    normalized_spectrograms = (spectrograms - means) / (3 * stds + 1e-6)
    clipped_spectrograms = torch.clamp(normalized_spectrograms, -1, 1)

    return clipped_spectrograms, means, stds


def forward_pass(models, input, secrets):
    noise_dim = models['filter_gen'].noise_dim

    # filter_gen
    filter_z = torch.randn(input.shape[0], noise_dim).to(input.device)
    filtered_mel = models['filter_gen'](input, filter_z, secrets.long())
    filtered_secret_preds_gen = models['filter_disc'](filtered_mel)
    filter_gen_output = {'filtered_mel': filtered_mel, 'filtered_secret_score': filtered_secret_preds_gen}

    # filter_disc
    filtered_secret_preds_disc = models['filter_disc'](filtered_mel.detach().clone())
    filter_disc_output = {'filtered_secret_score': filtered_secret_preds_disc}

    # secret_gen
    secret_z = torch.randn(input.shape[0], noise_dim).to(input.device)
    fake_secret_gen = Variable(LongTensor(np.random.choice([0.0, 1.0], input.shape[0]))).to(input.device)
    faked_mel = models['secret_gen'](filtered_mel.detach().clone(), secret_z, fake_secret_gen)
    fake_secret_preds_gen = models['secret_disc'](faked_mel)
    secret_gen_output = {'fake_secret': fake_secret_gen, 'faked_mel': faked_mel,
                         'fake_secret_score': fake_secret_preds_gen}

    # secret_disc
    fake_secret_preds_disc = models['secret_disc'](faked_mel.detach().clone())
    real_secret_preds_disc = models['secret_disc'](input)
    fake_secret_disc = Variable(LongTensor(fake_secret_preds_disc.size(0)).fill_(2.0), requires_grad=False).to(
        input.device)

    label_preds = models['label_classifier'](faked_mel)
    secret_preds = models['secret_classifier'](faked_mel)
    secret_disc_output = {'fake_secret_score': fake_secret_preds_disc, 'real_secret_score': real_secret_preds_disc,
                          'label_score': label_preds, 'secret_score': secret_preds, 'fake_secret': fake_secret_disc}

    return filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output


def evaluate_on_dataset(data_loader, audio2mel, models, loss_funcs, loss_compute_config, device):
    utils.set_mode(models, utils.Mode.EVAL)

    metrics = {}

    for i, (input, secret, label, _) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        label, secret = label.to(device), secret.to(device)
        input = torch.unsqueeze(input, 1)
        spectrograms = audio2mel(input).detach()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

        filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output = forward_pass(models,
                                                                                                    spectrograms,
                                                                                                    secret)
        losses = compute_losses(loss_funcs, spectrograms, secret, filter_gen_output, filter_disc_output,
                                secret_gen_output, secret_disc_output, loss_compute_config)
        utils.backward(losses)

        batch_metrics = compute_metrics(secret, label, filter_gen_output, filter_disc_output, secret_gen_output,
                                        secret_disc_output, losses)
        batch_metrics = compile_metrics(batch_metrics)
        metrics = aggregate_metrics(batch_metrics, metrics)

    return metrics


def save_test_samples(data_loader, audio2mel, mel2audio, models, loss_func, example_dirs, epoch, sampling_rate, device):
    utils.set_mode(models, utils.Mode.EVAL)
    noise_dim = models['filter_gen'].noise_dim

    for i, (input, secret, label, id) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        input, secret, label, id = input[:1], secret[:1], label[:1], id[:1]

        label, secret = label.to(device), secret.to(device)
        input = torch.unsqueeze(input, 1)
        spectrograms = audio2mel(input).detach()
        original_spectrograms = spectrograms.clone()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

        # filter_gen
        filter_z = torch.randn(spectrograms.shape[0], noise_dim).to(device)
        filtered = models['filter_gen'](spectrograms, filter_z, secret.long()).detach()

        # secret_gen
        secret_z = torch.randn(spectrograms.shape[0], noise_dim).to(device)
        fake_secret_male = Variable(LongTensor(spectrograms.size(0)).fill_(1.0), requires_grad=False).to(device)
        fake_secret_female = Variable(LongTensor(spectrograms.size(0)).fill_(0.0), requires_grad=False).to(device)
        fake_mel_male = models['secret_gen'](filtered, secret_z, fake_secret_male).detach()
        fake_mel_female = models['secret_gen'](filtered, secret_z, fake_secret_female).detach()

        # predict label
        pred_label_male = torch.argmax(models['label_classifier'](fake_mel_male).data, 1)
        pred_label_female = torch.argmax(models['label_classifier'](fake_mel_female).data, 1)

        # predict secret
        pred_secret_male = torch.argmax(models['secret_classifier'](fake_mel_male).data, 1)
        pred_secret_female = torch.argmax(models['secret_classifier'](fake_mel_female).data, 1)

        # distortions
        filtered_distortion = loss_func['distortion'](spectrograms, filtered)
        male_distortion = loss_func['distortion'](spectrograms, fake_mel_male).item()
        female_distortion = loss_func['distortion'](spectrograms, fake_mel_female).item()
        sample_distortion = loss_func['distortion'](fake_mel_male, fake_mel_female).item()

        unnormalized_filtered_mel = torch.squeeze(filtered, 1).to(device) * 3 * stds.to(device) + means.to(device)
        unnormalized_fake_mel_male = torch.squeeze(fake_mel_male, 1).to(device) * 3 * stds.to(device) + means.to(device)
        unnormalized_fake_mel_female = torch.squeeze(fake_mel_female, 1).to(device) * 3 * stds.to(device) + means.to(
            device)
        unnormalized_spectrograms = torch.squeeze(spectrograms.to(device) * 3 * stds.to(device) + means.to(device))

        filtered_audio = mel2audio(unnormalized_filtered_mel).squeeze().detach().cpu()
        audio_male = mel2audio(unnormalized_fake_mel_male).squeeze().detach().cpu()
        audio_female = mel2audio(unnormalized_fake_mel_female).squeeze().detach().cpu()

        utils.save_sample(example_dirs, id, label, epoch, pred_label_male, pred_label_female, filtered_audio,
                          audio_male, audio_female, unnormalized_spectrograms, sampling_rate)

        comparison_plot_pcgan(original_spectrograms, unnormalized_filtered_mel, unnormalized_fake_mel_male,
                              unnormalized_fake_mel_female, secret, label, pred_secret_male, pred_secret_female,
                              pred_label_male, pred_label_female, male_distortion, female_distortion, sample_distortion,
                              example_dirs, epoch, id)
    print("Success!")


def init_training(experiment_config, device):
    training_config = experiment_config.training_config
    audio2mel_config = experiment_config.audio2mel_config
    mel2audio_config = experiment_config.mel2audio_config
    unet_config = experiment_config.unet_config

    train_data, test_data = AudioDataset.load()
    train_loader = DataLoader(train_data, training_config.train_batch_size,
                              num_workers=training_config.train_num_workers, shuffle=True)
    test_loader = DataLoader(test_data, training_config.test_batch_size, num_workers=training_config.test_num_workers,
                             shuffle=True)

    audio2mel = Audio2Mel(audio2mel_config.n_fft, audio2mel_config.hop_length, audio2mel_config.win_length,
                          audio2mel_config.sampling_rate, audio2mel_config.n_mels)
    mel2audio = MelGanGenerator(mel2audio_config.input_size, mel2audio_config.ngf,
                                mel2audio_config.n_residual_layers).to(device)

    label_classifier = load_modified_ResNet(train_data.n_labels).to(device)
    secret_classifier = load_modified_ResNet(train_data.n_genders).to(device)

    label_classifier.eval()
    secret_classifier.eval()

    image_width, image_height = audio2mel.output_shape(train_data[0][0])

    loss_funcs = {'distortion': torch.nn.L1Loss(), 'entropy': HLoss(), 'adversarial': torch.nn.CrossEntropyLoss(),
                  'adversarial_rf': torch.nn.CrossEntropyLoss()}

    filter_gen = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=unet_config.kernel_size,
                            image_width=image_width, image_height=image_height, noise_dim=unet_config.noise_dim,
                            n_classes=train_data.n_genders, embedding_dim=unet_config.embedding_dim, use_cond=False).to(
        device)
    filter_disc = load_modified_AlexNet(train_data.n_genders).to(device)
    secret_gen = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=unet_config.kernel_size,
                            image_width=image_width, image_height=image_height, noise_dim=unet_config.noise_dim,
                            n_classes=train_data.n_genders, embedding_dim=unet_config.embedding_dim, use_cond=False).to(
        device)
    secret_disc = load_modified_AlexNet(train_data.n_genders + 1).to(device)

    models = {'filter_gen': filter_gen, 'filter_disc': filter_disc, 'secret_gen': secret_gen,
              'secret_disc': secret_disc, 'label_classifier': label_classifier, 'secret_classifier': secret_classifier}

    optimizers = {
        'filter_gen': torch.optim.Adam(filter_gen.parameters(), training_config.lr['filter_gen'], betas=(0.5, 0.9)),
        'filter_disc': torch.optim.Adam(filter_disc.parameters(), training_config.lr['filter_gen'], betas=(0.5, 0.9)),
        'secret_gen': torch.optim.Adam(secret_gen.parameters(), training_config.lr['secret_gen'], betas=(0.5, 0.9)),
        'secret_disc': torch.optim.Adam(secret_disc.parameters(), training_config.lr['secret_disc'], betas=(0.5, 0.9))
    }
    return train_loader, test_loader, audio2mel, mel2audio, loss_funcs, models, optimizers


def init_dirs(run_name):
    run_dir = local_vars.PWD + 'runs/audioMNIST/' + run_name
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    samples_dir = os.path.join(run_dir, 'samples')

    example_dir = os.path.join(run_dir, 'examples')
    example_audio_dir = os.path.join(example_dir, 'audio')
    example_spec_dir = os.path.join(example_dir, 'spectrograms')
    example_dirs = {'audio': example_audio_dir, 'spec': example_spec_dir}

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=False)
    Path(samples_dir).mkdir(parents=True, exist_ok=False)
    Path(example_audio_dir).mkdir(parents=True, exist_ok=False)
    Path(example_spec_dir).mkdir(parents=True, exist_ok=False)

    return run_dir, checkpoint_dir, samples_dir, example_dirs


def main():
    # device = 'cpu'
    # experiment_config = configs.get_experiment_config_debug()

    device = 'cuda:0'
    # experiment_config = configs.get_experiment_config_fast_run()
    experiment_config = configs.get_experiment_config_pcgan()

    device = torch.device(device)
    training_config, audio2mel_config, mel2audio_config, unet_config, loss_compute_config = experiment_config.get_configs()
    train_loader, test_loader, audio2mel, mel2audio, loss_funcs, models, optimizers = init_training(experiment_config,
                                                                                                    device)
    log.init(utils.nestedConfigs2dict(experiment_config), run_name=training_config.run_name)
    run_dir, checkpoint_dir, samples_dir, example_dirs = init_dirs(training_config.run_name)

    utils.zero_grad(optimizers)
    save_epoch = 0
    for epoch in range(0, training_config.epochs):
        epoch = epoch + 1
        epoch_start = time.time()

        utils.set_mode(models, utils.Mode.TRAIN)
        step_counter = 0
        for i, (input, secret, label, _) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            step_counter += 1

            label, secret = label.to(device), secret.to(device)
            input = torch.unsqueeze(input, 1)
            spectrograms = audio2mel(input).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

            filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output = forward_pass(models,
                                                                                                        spectrograms,
                                                                                                        secret)
            losses = compute_losses(loss_funcs, spectrograms, secret, filter_gen_output, filter_disc_output,
                                    secret_gen_output, secret_disc_output, loss_compute_config)
            utils.backward(losses)

            metrics = compute_metrics(secret, label, filter_gen_output, filter_disc_output, secret_gen_output,
                                      secret_disc_output, losses)
            metrics = compile_metrics(metrics)
            log.metrics(metrics, suffix='train', commit=True)

            if step_counter % training_config.updates_per_evaluation == 0:
                val_metrics = evaluate_on_dataset(test_loader, audio2mel, models, loss_funcs, loss_compute_config,
                                                  device)
                log.metrics(val_metrics, suffix='val', aggregation=np.mean, commit=True)

            if step_counter % training_config.gradient_accumulation == 0:
                utils.step(optimizers)
                utils.zero_grad(optimizers)

        if epoch % training_config.save_interval == 0:
            print("Saving audio and spectrogram samples.")
            save_test_samples(test_loader, audio2mel, mel2audio, models, loss_funcs, example_dirs, epoch,
                              audio2mel_config.sampling_rate, device)

        if epoch % training_config.checkpoint_interval == 0:
            utils.save_models_and_optimizers(checkpoint_dir, epoch, models, optimizers)
            print('make_checkpoints', epoch, step_counter)


if __name__ == '__main__':
    main()
