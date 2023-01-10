from training.utils import preprocess_spectrograms
# from training.sampling import save_test_samples, generate_samples
import numpy as np
import torch
import utils
import tqdm
import time
import log
import os


def training_loop(train_loader, test_loader, training_config, architecture, audio_mel_converter, sample_rate, device):
    utils.zero_grad(architecture.optimizers)
    total_steps = 0
    for epoch in range(0, training_config.epochs):
        epoch = epoch + 1
        epoch_start = time.time()

        architecture.set_mode(utils.Mode.TRAIN)
        step_counter = 0
        metrics = {}
        for i, (data, secrets, labels, _, _) in tqdm.tqdm(enumerate(train_loader), 'Epoch {}: Training'.format(epoch),
                                                          total=len(train_loader)):

            # data: (bsz x seq_len), secrets: (bsz,), labels: (bsz,)
            step_counter += 1
            total_steps += 1
            labels, secrets = labels.to(device), secrets.to(device)
            mels = audio_mel_converter.audio2mel(data).detach()  # mels: (bsz, n_mels, frames)
            mels, means, stds = preprocess_spectrograms(mels)
            mels = mels.unsqueeze(dim=1).to(device)  # mels: (bsz, 1, n_mels, frames)

            batch_metrics, losses = architecture.forward_pass(mels, secrets, labels)
            batch_metrics = utils.compile_metrics(batch_metrics)
            metrics = utils.aggregate_metrics(batch_metrics, metrics)

            do_log_eval = total_steps % training_config.updates_per_evaluation == 0
            do_log_train = total_steps % training_config.updates_per_train_log_commit == 0
            if training_config.do_log and (do_log_train or do_log_eval):
                if do_log_eval:
                    val_metrics = evaluate_on_dataset(test_loader, audio_mel_converter, architecture, device)
                    log.metrics(val_metrics, total_steps, suffix='val', aggregation=np.mean, commit=False)
                if do_log_train:
                    log.metrics(metrics, total_steps, suffix='train', commit=False)
                log.metrics({'Epoch': epoch + (i / len(train_loader))}, total_steps, commit=True)

            if total_steps % training_config.gradient_accumulation == 0:
                utils.backward(losses)
                utils.step(architecture.optimizers)
                utils.zero_grad(architecture.optimizers)

        if epoch % training_config.save_interval == 0:
            print("Saving data and mels samples.")
            save_samples(utils.create_run_subdir(test_loader.dataset.get_name(), training_config.run_name, 'samples'),
                         test_loader, audio_mel_converter, architecture, epoch, sample_rate, device,
                         training_config.n_samples)

        if epoch % training_config.checkpoint_interval == 0:
            utils.save_models_and_optimizers(
                utils.create_run_subdir(train_loader.dataset.get_name(), training_config.run_name, 'checkpoints'),
                epoch, architecture)


def evaluate_on_dataset(data_loader, audio_mel_converter, architecture, device):
    architecture.set_mode(utils.Mode.EVAL)

    metrics = {}

    with torch.no_grad():
        for i, (data, secrets, labels, _, _) in tqdm.tqdm(enumerate(data_loader), 'Evaluation', total=len(data_loader)):
            # data: (bsz x seq_len), secrets: (bsz,), labels: (bsz,)
            labels, secrets = labels.to(device), secrets.to(device)
            mels = audio_mel_converter.audio2mel(data).detach()  # spectrogram: (bsz, n_mels, frames)
            mels, means, stds = preprocess_spectrograms(mels)
            mels = mels.unsqueeze(dim=1).to(device)  # spectrogram: (bsz, 1, n_mels, frames)

            batch_metrics, losses = architecture.forward_pass(mels, secrets, labels)
            batch_metrics = utils.compile_metrics(batch_metrics)
            metrics = utils.aggregate_metrics(batch_metrics, metrics)

    architecture.set_mode(utils.Mode.TRAIN)
    return metrics


def save_samples(example_dir, data_loader, audio_mel_converter, architecture, epoch, sampling_rate, device, n_samples):
    architecture.set_mode(utils.Mode.EVAL)
    save_dir = utils.create_subdir(example_dir, 'audio')

    with torch.no_grad():
        for i, (data, secret, label, id, _) in tqdm.tqdm(enumerate(data_loader), 'Generating Samples',
                                                         total=len(data_loader)):
            if i >= n_samples:
                break

            data, secret, label, id = data[:1], secret[:1], label[:1], id[:1]
            # data: (1 x seq_len), secret: (1,), label: (1,), id: (1,)

            label, secret = label.to(device), secret.to(device)
            original_mel = audio_mel_converter.audio2mel(data).detach()
            mel, mean, std = preprocess_spectrograms(original_mel)
            mel = mel.unsqueeze(dim=1).to(device)

            sample = architecture.generate_sample(data, mel, std, mean, secret, label, id, audio_mel_converter, epoch)
            [utils.save_audio_file(os.path.join(save_dir, k), sampling_rate, v) for k, v in sample.items()]
