from metrics_compiling import compute_metrics, compile_metrics, aggregate_metrics
from loss_compiling import compute_losses
from training.utils import preprocess_spectrograms
from training.sampling import save_test_samples
import numpy as np
import torch
import utils
import tqdm
import time
import log

from torch.autograd import Variable
from torch import LongTensor


def forward_pass(models, audio, secrets):
    noise_dim = models['filter_gen'].noise_dim

    # filter_gen
    filter_z = torch.randn(audio.shape[0], noise_dim).to(audio.device)
    filtered_mel = models['filter_gen'](audio, filter_z, secrets.long())
    filtered_secret_preds_gen = models['filter_disc'](filtered_mel)
    filter_gen_output = {'filtered_mel': filtered_mel, 'filtered_secret_score': filtered_secret_preds_gen}

    # filter_disc
    filtered_secret_preds_disc = models['filter_disc'](filtered_mel.detach().clone())
    filter_disc_output = {'filtered_secret_score': filtered_secret_preds_disc}

    # secret_gen
    secret_z = torch.randn(audio.shape[0], noise_dim).to(audio.device)
    fake_secret_gen = Variable(LongTensor(np.random.choice([0.0, 1.0], audio.shape[0]))).to(audio.device)
    faked_mel = models['secret_gen'](filtered_mel.detach().clone(), secret_z, fake_secret_gen)
    fake_secret_preds_gen = models['secret_disc'](faked_mel)
    secret_gen_output = {'fake_secret': fake_secret_gen, 'faked_mel': faked_mel,
                         'fake_secret_score': fake_secret_preds_gen}

    # secret_disc
    fake_secret_preds_disc = models['secret_disc'](faked_mel.detach().clone())
    real_secret_preds_disc = models['secret_disc'](audio)
    fake_secret_disc = Variable(LongTensor(fake_secret_preds_disc.size(0)).fill_(2.0), requires_grad=False).to(
        audio.device)

    label_preds = models['label_classifier'](faked_mel)
    secret_preds = models['secret_classifier'](faked_mel)
    secret_disc_output = {'fake_secret_score': fake_secret_preds_disc, 'real_secret_score': real_secret_preds_disc,
                          'label_score': label_preds, 'secret_score': secret_preds, 'fake_secret': fake_secret_disc}

    return filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output


def training_loop(train_loader, test_loader, training_config, models, optimizers, audio_mel_converter, loss_funcs,
                  gamma, use_entropy_loss, sample_rate, device):
    utils.zero_grad(optimizers)
    save_epoch = 0
    for epoch in range(0, training_config.epochs):
        epoch = epoch + 1
        epoch_start = time.time()

        utils.set_mode(models, utils.Mode.TRAIN)
        step_counter = 0
        for i, (audio, secret, label, _, _) in tqdm.tqdm(enumerate(train_loader), 'Epoch {}: Training'.format(epoch),
                                                         total=len(train_loader)):
            step_counter += 1

            label, secret = label.to(device), secret.to(device)
            audio = torch.unsqueeze(audio, 1)
            spectrograms = audio_mel_converter.audio2mel(audio).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = spectrograms.to(device)
            spectrograms = spectrograms.unsqueeze(dim=1) if spectrograms.dim() == 3 else spectrograms

            filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output = forward_pass(models,
                                                                                                        spectrograms,
                                                                                                        secret)
            losses = compute_losses(loss_funcs, spectrograms, secret, filter_gen_output, filter_disc_output,
                                    secret_gen_output, secret_disc_output, gamma, use_entropy_loss)
            utils.backward(losses)

            metrics = compute_metrics(secret, label, filter_gen_output, filter_disc_output, secret_gen_output,
                                      secret_disc_output, losses)
            metrics = compile_metrics(metrics)
            if training_config.do_log:
                log.metrics(metrics, suffix='train', commit=True)

            if step_counter % training_config.updates_per_evaluation == 0:
                val_metrics = evaluate_on_dataset(test_loader, audio_mel_converter, models, loss_funcs, gamma,
                                                  use_entropy_loss,
                                                  device)
                if training_config.do_log:
                    log.metrics(val_metrics, suffix='val', aggregation=np.mean, commit=True)

            if step_counter % training_config.gradient_accumulation == 0:
                utils.step(optimizers)
                utils.zero_grad(optimizers)

        if epoch % training_config.save_interval == 0:
            print("Saving audio and spectrogram samples.")
            save_test_samples(utils.create_run_subdir(training_config.run_name, 'samples'), test_loader,
                              audio_mel_converter,
                              models, loss_funcs, epoch, sample_rate, device)

        if epoch % training_config.checkpoint_interval == 0:
            utils.save_models_and_optimizers(utils.create_run_subdir(training_config.run_name, 'checkpoints'),
                                             epoch, models, optimizers)


def evaluate_on_dataset(data_loader, audio_mel_converter, models, loss_funcs, gamma, use_entropy_loss, device):
    utils.set_mode(models, utils.Mode.EVAL)

    metrics = {}

    for i, (input, secret, label, _, _) in tqdm.tqdm(enumerate(data_loader), 'Evaluation', total=len(data_loader)):
        label, secret = label.to(device), secret.to(device)
        input = torch.unsqueeze(input, 1)
        spectrograms = audio_mel_converter.audio2mel(input).detach()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = spectrograms.to(device)
        spectrograms = spectrograms.unsqueeze(dim=1) if spectrograms.dim() == 3 else spectrograms

        filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output = forward_pass(models,
                                                                                                    spectrograms,
                                                                                                    secret)
        losses = compute_losses(loss_funcs, spectrograms, secret, filter_gen_output, filter_disc_output,
                                secret_gen_output, secret_disc_output, gamma, use_entropy_loss)
        utils.backward(losses)

        batch_metrics = compute_metrics(secret, label, filter_gen_output, filter_disc_output, secret_gen_output,
                                        secret_disc_output, losses)
        batch_metrics = compile_metrics(batch_metrics)
        metrics = aggregate_metrics(batch_metrics, metrics)

    return metrics
