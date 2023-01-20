# from training.utils import preprocess_spectrograms
# from training.sampling import save_test_samples, generate_samples
import librosa
from training.optimizer_updating import OptimizerUpdater
import numpy as np
import torch
import utils
import tqdm
import time
import log
import os


def training_loop(train_loader, test_loader, training_config, architecture, device):
    utils.zero_grad(architecture.optimizers)
    total_steps = 0
    optimizer_updater = OptimizerUpdater(training_config.disc_graident_accumulation, training_config.gen_gradient_accumulation,
                                         architecture.disc_optimizers, architecture.gen_optimizers)

    save_samples(utils.create_run_subdir(test_loader.dataset.get_name(), training_config.run_name, 'samples'),
                 test_loader, architecture, 0, device, training_config.n_samples)

    for epoch in range(0, training_config.epochs):
        epoch = epoch + 1
        epoch_start = time.time()

        architecture.set_mode(utils.Mode.TRAIN)
        step_counter = 0
        metrics = {}
        for i, (audio, secrets, labels, _, _) in tqdm.tqdm(enumerate(train_loader), 'Epoch {}: Training'.format(epoch),
                                                           total=len(train_loader)):
            # data: (bsz x seq_len), secrets: (bsz,), labels: (bsz,)
            step_counter += 1
            total_steps += 1
            labels, secrets = labels.to(device), secrets.to(device)

            batch_metrics, losses = architecture.forward_pass(audio, secrets, labels)
            batch_metrics = utils.compile_metrics(batch_metrics)
            metrics = utils.aggregate_metrics(batch_metrics, metrics)

            do_log_eval = total_steps % training_config.updates_per_evaluation == 0
            do_log_train = total_steps % training_config.updates_per_train_log_commit == 0
            if training_config.do_log and (do_log_train or do_log_eval):
                if do_log_eval:
                    val_metrics = evaluate_on_dataset(test_loader, architecture, device)
                    log.metrics(val_metrics, total_steps, suffix='val', aggregation=np.mean, commit=False)
                if do_log_train:
                    log.metrics(metrics, total_steps, suffix='train', commit=False)
                log.metrics({'Epoch': epoch + (i / len(train_loader))}, total_steps, commit=True)

            optimizer_updater.step(total_steps, losses)
            # if total_steps % training_config.gradient_accumulation == 0:
            #     utils.backward(losses)
            #     utils.step(architecture.optimizers)
            #     utils.zero_grad(architecture.optimizers)

        if epoch % training_config.save_interval == 0:
            print("Saving data and mels samples.")
            save_samples(utils.create_run_subdir(test_loader.dataset.get_name(), training_config.run_name, 'samples'),
                         test_loader, architecture, epoch, device, training_config.n_samples)

        if epoch % training_config.checkpoint_interval == 0:
            architecture.save(
                utils.create_run_subdir(train_loader.dataset.get_name(), training_config.run_name, 'checkpoints'),
                epoch)


def evaluate_on_dataset(data_loader, architecture, device):
    architecture.set_mode(utils.Mode.EVAL)

    metrics = {}

    with torch.no_grad():
        for i, (audio, secrets, labels, _, _) in tqdm.tqdm(enumerate(data_loader), 'Evaluation',
                                                           total=len(data_loader)):
            # data: (bsz x seq_len), secrets: (bsz,), labels: (bsz,)
            labels, secrets = labels.to(device), secrets.to(device)
            # mels = audio_mel_converter.audio2mel(data).detach()  # spectrogram: (bsz, n_mels, frames)
            # mels, means, stds = preprocess_spectrograms(mels)
            # mels = mels.unsqueeze(dim=1).to(device)  # spectrogram: (bsz, 1, n_mels, frames)

            batch_metrics, losses = architecture.forward_pass(audio, secrets, labels)
            batch_metrics = utils.compile_metrics(batch_metrics)
            metrics = utils.aggregate_metrics(batch_metrics, metrics)

    architecture.set_mode(utils.Mode.TRAIN)
    return metrics


def save_samples(example_dir, data_loader, architecture, epoch, device, n_samples):
    architecture.set_mode(utils.Mode.EVAL)
    save_dir = utils.create_subdir(example_dir, 'audio')

    with torch.no_grad():
        for i, (audio, secret, label, id, _) in tqdm.tqdm(enumerate(data_loader), 'Generating Samples',
                                                          total=len(data_loader)):
            if i >= n_samples:
                break

            audio, secret, label, id = audio[:1], secret[:1], label[:1], id[:1]
            # data: (1 x seq_len), secret: (1,), label: (1,), id: (1,)

            label, secret = label.to(device), secret.to(device)
            # original_mel = audio_mel_converter.audio2mel(data).detach()
            # mel, mean, std = preprocess_spectrograms(original_mel)
            # mel = mel.unsqueeze(dim=1).to(device)

            sample = architecture.generate_sample(audio, secret, label, id, epoch)
            [utils.save_audio_file(os.path.join(save_dir, k), architecture.sampling_rate, v) for k, v in sample.items()]
