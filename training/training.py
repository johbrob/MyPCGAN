import loss_compiling
from metrics_compiling import compute_metrics, compile_metrics, aggregate_metrics
from training.utils import preprocess_spectrograms
from training.sampling import save_test_samples, generate_samples
import numpy as np
import torch
import utils
import tqdm
import time
import log


def filter_gen_forward_pass(filter_gen, filter_disc, mels, secrets):
    filter_z = torch.randn(mels.shape[0], filter_gen.noise_dim).to(mels.device)
    filtered_mels = filter_gen(mels, filter_z, secrets.long())  # (bsz, 1, n_mels, frames)
    filtered_secret_preds_gen = filter_disc(filtered_mels, frozen=True)  # (bsz, n_secret)
    return {'filtered_mel': filtered_mels, 'filtered_secret_score': filtered_secret_preds_gen}


def secret_gen_forward_pass(secret_gen, secret_disc, mels, filtered_mel):
    secret_z = torch.randn(mels.shape[0], secret_gen.noise_dim).to(mels.device)
    fake_secret_gen = torch.randint(0, 1, mels.shape[0:1]).to(mels.device)  # (bsz,)
    fake_mel = secret_gen(filtered_mel.detach(), secret_z, fake_secret_gen)  # (bsz, 1, n_mels, frames)
    fake_secret_preds_gen = secret_disc(fake_mel, frozen=True)  # (bsz, n_secrets + 1)
    secret_gen_output = {'fake_secret': fake_secret_gen, 'faked_mel': fake_mel,
                         'fake_secret_score': fake_secret_preds_gen}

    generate_both_genders = True
    if generate_both_genders:
        alt_fake_mel = secret_gen(filtered_mel.detach(), secret_z, 1 - fake_secret_gen,
                                  frozen=True)  # (bsz, 1, n_mels, frames)
        alt_fake_secret_preds_gen = secret_disc(alt_fake_mel, frozen=True)  # (bsz, n_secrets + 1)
        secret_gen_output.update({'alt_faked_mel': alt_fake_mel, 'alt_fake_secret_score': alt_fake_secret_preds_gen})

    return secret_gen_output


def filter_disc_forward_pass(filter_disc, mels, filtered_mels):
    filtered_secret_preds_disc = filter_disc(filtered_mels.detach())  # (bsz, n_secret)
    unfiltered_secret_preds_disc = filter_disc(mels.detach(), frozen=True)  # (bsz, n_secret)
    return {'filtered_secret_score': filtered_secret_preds_disc,
            'unfiltered_secret_score': unfiltered_secret_preds_disc}


def secret_disc_forward_pass(secret_disc, mels, fake_mel):
    fake_secret_preds_disc = secret_disc(fake_mel.detach().clone())  # (bsz, n_secrets + 1)
    real_secret_preds_disc = secret_disc(mels)  # (bsz, n_secrets + 1)
    fake_secret_disc = 2 * torch.ones(mels.size(0), requires_grad=False, dtype=torch.int64).to(mels.device)  # (bsz,)
    return {'fake_secret_score': fake_secret_preds_disc, 'real_secret_score': real_secret_preds_disc,
            'fake_secret': fake_secret_disc}


def forward_pass(models, mels, secrets):
    assert mels.dim() == 4
    filter_gen_output = filter_gen_forward_pass(models['filter_gen'], models['filter_disc'], mels, secrets)
    secret_gen_output = secret_gen_forward_pass(models['secret_gen'], models['secret_disc'], mels,
                                                filter_gen_output['filtered_mel'])
    filter_disc_output = filter_disc_forward_pass(models['filter_disc'], mels, filter_gen_output['filtered_mel'])
    secret_disc_output = secret_disc_forward_pass(models['secret_disc'], mels, secret_gen_output['faked_mel'])

    label_preds = models['label_classifier'](secret_gen_output['faked_mel'])
    secret_preds = models['secret_classifier'](secret_gen_output['faked_mel'])
    secret_disc_output.update({'label_score': label_preds, 'secret_score': secret_preds})

    return filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output


def training_loop(train_loader, test_loader, training_config, models, optimizers, audio_mel_converter, loss_funcs,
                  loss_config, sample_rate, device):
    utils.zero_grad(optimizers)
    total_steps = 0
    for epoch in range(0, training_config.epochs):
        epoch = epoch + 1
        epoch_start = time.time()

        utils.set_mode(models, utils.Mode.TRAIN)
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

            filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output = forward_pass(models,
                                                                                                        mels,
                                                                                                        secrets)

            losses = loss_compiling.compute_losses(loss_funcs, mels, secrets, filter_gen_output, filter_disc_output,
                                                   secret_gen_output, secret_disc_output, loss_config)

            batch_metrics = compute_metrics(mels, secrets, labels, filter_gen_output, filter_disc_output,
                                            secret_gen_output,
                                            secret_disc_output, losses, loss_funcs)
            batch_metrics = compile_metrics(batch_metrics)
            metrics = aggregate_metrics(batch_metrics, metrics)

            do_log_eval = total_steps % training_config.updates_per_evaluation == 0
            do_log_train = total_steps % training_config.updates_per_train_log_commit == 0
            if training_config.do_log and (do_log_train or do_log_eval):
                if do_log_eval:
                    val_metrics = evaluate_on_dataset(test_loader, audio_mel_converter, models, loss_funcs, loss_config,
                                                      device)
                    log.metrics(val_metrics, total_steps, suffix='val', aggregation=np.mean, commit=False)
                if do_log_train:
                    log.metrics(metrics, total_steps, suffix='train', commit=False)
                log.metrics({'Epoch': epoch + (i / len(train_loader))}, total_steps, commit=True)

            if total_steps % training_config.gradient_accumulation == 0:
                utils.backward(losses)
                utils.step(optimizers)
                utils.zero_grad(optimizers)

        if epoch % training_config.save_interval == 0:
            print("Saving data and mels samples.")
            save_test_samples(
                utils.create_run_subdir(train_loader.dataset.get_name(), training_config.run_name, 'samples'),
                test_loader, audio_mel_converter, models, loss_funcs, epoch, sample_rate, device,
                training_config.n_generated_samples)

        if epoch % training_config.checkpoint_interval == 0:
            utils.save_models_and_optimizers(
                utils.create_run_subdir(train_loader.dataset.get_name(), training_config.run_name, 'checkpoints'),
                epoch, models, optimizers)


def evaluate_on_dataset(data_loader, audio_mel_converter, models, loss_funcs, loss_config, device):
    utils.set_mode(models, utils.Mode.EVAL)

    metrics = {}

    with torch.no_grad():
        for i, (data, secrets, labels, _, _) in tqdm.tqdm(enumerate(data_loader), 'Evaluation', total=len(data_loader)):
            # data: (bsz x seq_len), secrets: (bsz,), labels: (bsz,)
            labels, secrets = labels.to(device), secrets.to(device)
            mels = audio_mel_converter.audio2mel(data).detach()  # spectrogram: (bsz, n_mels, frames)
            mels, means, stds = preprocess_spectrograms(mels)
            mels = mels.unsqueeze(dim=1).to(device)  # spectrogram: (bsz, 1, n_mels, frames)

            filter_gen_output, filter_disc_output, secret_gen_output, secret_disc_output = forward_pass(models, mels,
                                                                                                        secrets)
            losses = loss_compiling.compute_losses(loss_funcs, mels, secrets, filter_gen_output, filter_disc_output,
                                                   secret_gen_output, secret_disc_output, loss_config)
            batch_metrics = compute_metrics(mels, secrets, labels, filter_gen_output, filter_disc_output,
                                            secret_gen_output,
                                            secret_disc_output, losses, loss_funcs)
            batch_metrics = compile_metrics(batch_metrics)
            metrics = aggregate_metrics(batch_metrics, metrics)

    return metrics
