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
        label = label.to(device)
        secret = secret.to(device)
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


def save_test_samples(test_loader, audio2mel, mel2audio, models, losses, run_dir, epoch, sampling_rate, device):
    example_dir = os.path.join(run_dir, 'examples')
    example_audio_dir = os.path.join(example_dir, 'audio')
    example_spec_dir = os.path.join(example_dir, 'spectrograms')

    utils.set_mode(models, utils.Mode.EVAL)

    for i, (x, gender, digit, speaker_id) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        if i % 50 == 0:
            x = torch.unsqueeze(x, 1)
            spectrograms = audio2mel(x).detach()
            spec_original = spectrograms
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms, 1).to(device)
            gender = gender.to(device)
            digit = digit.to(device)

            z1 = torch.randn(spectrograms.shape[0], 10).to(device)
            filtered = models['filter_gen'](spectrograms, z1, gender.long()).detach()

            z2 = torch.randn(spectrograms.shape[0], 10).to(device)
            male = Variable(LongTensor(spectrograms.size(0)).fill_(1.0), requires_grad=False).to(device)
            female = Variable(LongTensor(spectrograms.size(0)).fill_(0.0), requires_grad=False).to(device)
            generated_male = models['secret_gen'](filtered, z2, male).detach()
            generated_female = models['secret_gen'](filtered, z2, female).detach()

            # Predict digit
            digit_male = models['label_classifier'](generated_male)
            pred_digit_male = torch.argmax(digit_male.data, 1)
            digit_female = models['label_classifier'](generated_female)
            pred_digit_female = torch.argmax(digit_female.data, 1)

            # Predict gender
            gender_male = models['secret_classifier'](generated_male)
            pred_gender_male = torch.argmax(gender_male.data, 1)
            gender_female = models['secret_classifier'](generated_female)
            pred_gender_female = torch.argmax(gender_female.data, 1)

            if pred_gender_male == 0:
                pred_gender_male = 'female'
            else:
                pred_gender_male = 'male'

            if pred_gender_female == 0:
                pred_gender_female = 'female'
            else:
                pred_gender_female = 'male'

            # Distortions
            filtered_distortion = losses['distortion'](spectrograms, filtered)
            male_distortion = losses['distortion'](spectrograms, generated_male).item()
            female_distortion = losses['distortion'](spectrograms, generated_female).item()
            sample_distortion = losses['distortion'](generated_male, generated_female).item()

            filtered = torch.squeeze(filtered, 1).to(device) * 3 * stds.to(device) + means.to(device)
            generated_male = torch.squeeze(generated_male, 1).to(device) * 3 * stds.to(device) + means.to(
                device)
            generated_female = torch.squeeze(generated_female, 1).to(device) * 3 * stds.to(
                device) + means.to(device)
            spectrograms = spectrograms.to(device) * 3 * stds.to(device) + means.to(device)

            inverted_filtered = mel2audio(filtered).squeeze().detach().cpu()
            inverted_male = mel2audio(generated_male).squeeze().detach().cpu()
            inverted_female = mel2audio(generated_female).squeeze().detach().cpu()

            f_name_filtered_audio = os.path.join(example_audio_dir, 'speaker_{}_digit_{}_epoch_{}_filtered.wav'.format(
                speaker_id.item(), digit.item(), epoch + 1))
            f_name_male_audio = os.path.join(example_audio_dir,
                                             'speaker_{}_digit_{}_epoch_{}_sampled_gender_male_predicted_digit_{}.wav'.format(
                                                 speaker_id.item(), digit.item(), epoch + 1, pred_digit_male.item()))
            f_name_female_audio = os.path.join(example_audio_dir,
                                               'speaker_{}_digit_{}_epoch_{}_sampled_gender_female_predicted_digit_{}.wav'.format(
                                                   speaker_id.item(), digit.item(), epoch + 1,
                                                   pred_digit_female.item()))
            f_name_original_audio = os.path.join(example_audio_dir,
                                                 'speaker_{}_digit_{}_.wav'.format(speaker_id.item(), digit.item()))

            utils.save_sample(f_name_filtered_audio, sampling_rate, inverted_filtered)
            utils.save_sample(f_name_male_audio, sampling_rate, inverted_male)
            utils.save_sample(f_name_female_audio, sampling_rate, inverted_female)
            utils.save_sample(f_name_original_audio, sampling_rate, torch.squeeze(x))

            if gender == 0:
                gender_title = 'female'
            else:
                gender_title = 'male'
            orig_title = 'Original spectrogram - Gender: {} - Digit: {}'.format(gender_title, digit.item())
            filtered_title = 'Filtered spectrogram'
            male_title = 'Sampled/predicted gender: male / {} | Predicted digit: {} \n Distortion loss: {:5.5f} (original) | {:5.5f} (female) ({}_loss)'.format(
                pred_gender_male, pred_digit_male.item(), male_distortion, sample_distortion, 'l1')
            female_title = 'Sampled/predicted gender: female / {} | Predicted digit: {} \n Distortion loss: {:5.5f} (original) | {:5.5f} (male) ({}_loss)'.format(
                pred_gender_female, pred_digit_female.item(), female_distortion, sample_distortion,
                'l1')
            f_name = os.path.join(example_spec_dir, 'speaker_{}_digit_{}_epoch_{}'.format(
                speaker_id.item(), digit.item(), epoch + 1
            ))

            Path(f_name).parent.mkdir(parents=True, exist_ok=True)
            comparison_plot_pcgan(f_name, spec_original, filtered, generated_male, generated_female,
                                  orig_title, filtered_title, male_title, female_title)
    print("Success!")


def init_training(training_config, device):
    train_data, test_data = AudioDataset.load()
    train_loader = DataLoader(train_data, training_config.train_batch_size,
                              num_workers=training_config.train_num_workers, shuffle=True)
    test_loader = DataLoader(test_data, training_config.test_batch_size,
                             num_workers=training_config.test_num_workers)

    audio2mel = Audio2Mel(training_config.n_fft, training_config.hop_length, training_config.win_length,
                          training_config.sampling_rate, training_config.n_mels)
    mel2audio = MelGanGenerator(training_config.n_mels, training_config.ngf, training_config.n_resnet_layers).to(device)

    label_classifier = load_modified_ResNet(train_data.n_genders).to(device)
    secret_classifier = load_modified_ResNet(train_data.n_genders).to(device)

    label_classifier.eval()
    secret_classifier.eval()

    image_width, image_height = audio2mel.output_shape(train_data[0][0])

    loss_funcs = {'distortion': torch.nn.L1Loss(), 'entropy': HLoss(), 'adversarial': torch.nn.CrossEntropyLoss(),
                  'adversarial_rf': torch.nn.CrossEntropyLoss()}

    filter_gen = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=training_config.kernel_size,
                            image_width=image_width, image_height=image_height, noise_dim=training_config.noise_dim,
                            n_classes=train_data.n_genders, embedding_dim=training_config.embedding_dim,
                            use_cond=False).to(device)
    filter_disc = load_modified_AlexNet(train_data.n_genders).to(device)
    secret_gen = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=training_config.kernel_size,
                            image_width=image_width, image_height=image_height, noise_dim=training_config.noise_dim,
                            n_classes=train_data.n_genders, embedding_dim=training_config.embedding_dim,
                            use_cond=False).to(device)
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


def main():
    sampling_rate = 8000
    segment_length = 8192
    device = 'cpu'

    gradient_accumulation = 1
    updates_per_evaluation = 1

    save_interval = 1
    checkpoint_interval = 1

    training_config = configs.get_training_config_mini()
    loss_compute_config = training_config.loss_compute_config
    train_loader, test_loader, audio2mel, mel2audio, loss_funcs, models, optimizers = init_training(training_config,
                                                                                                    device)

    ####################################
    # Dump arguments and create logger #
    ####################################
    # with open(Path(run_dir) / "args.yml", "w") as f:
    #     yaml.dump(args, f)
    #     yaml.dump({'Seed used': manual_seed}, f)
    #     yaml.dump({'Run number': run}, f)
    from torch.utils.tensorboard import SummaryWriter
    run_id = str(np.random.randint(0, 9, 7))[1:-1].replace(' ', '')
    run_dir = local_vars.PWD + 'runs/audioMNIST/' + run_id
    writer = SummaryWriter(run_dir)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    visuals_dir = os.path.join(run_dir, 'visuals')

    log.init(vars(training_config))

    utils.zero_grad(optimizers)
    for epoch in range(0, 10):
        epoch_start = time.time()

        utils.set_mode(models, utils.Mode.TRAIN)
        step_counter = 0
        for i, (input, secret, label, _) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            step_counter += 1
            # if i > 0:
            #     print('breaking news')
            #     break
            # print('its going ok')
            label = label.to(device)
            secret = secret.to(device)
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

            # if step_counter % updates_per_evaluation == 0:
            #     # val_metrics = evaluate_on_dataset(test_loader, audio2mel, models, loss_funcs, loss_compute_config,
            #     #                                   device)
            #     # print(val_metrics)
            #     # log.metrics(val_metrics, suffix='val', aggregation=np.mean, commit=True)

            if step_counter % gradient_accumulation == 0:
                utils.step(optimizers)
                utils.zero_grad(optimizers)
            # print('__________________________________________________________________________')
            # print("Epoch {} completed | Time: {:5.2f} s ".format(epoch + 1, time.time() - epoch_start))
            # print("filterGen    | Adversarial loss: {:5.5f} | Distortion loss: {:5.5f}".format(
            #     F_adversary_loss_accum / (i + 1), F_distortion_loss_accum / (i + 1)))
            # print("filterDisc   | Filtered sample accuracy: {} %".format(FD_accuracy))
            # print(
            #     "secretGen    | Advsarial loss: {:5.5f} | Distortion loss: {:5.5f}".format(
            #         G_adversary_loss_accum / (i + 1),
            #         G_distortion_loss_accum / (
            #                 i + 1)))
            # print("secretDisc   | Real samples: {} % | Fake samples: {} % | Sampled gender accuracy: {} % ".format(
            #     GD_accuracy_real, GD_accuracy_fake, GD_accuracy_gender_fake))
            # print("Fix Digit accuracy: {} % | Fix gender accuracy: {} %".format(fix_digit_spec_classfier_accuracy,
            #                                                                     fix_gender_spec_classfier_accuracy))
            # ----------------------------------------------
            #   Compute test accuracy
            # ----------------------------------------------
            # if epoch % 10 == 0:
            #     test_correct_digit, test_fixed_original_gender, test_fixed_sampled_gender = validate_on_dataset(
            #         test_loader, audio2mel, models, device)
            #
            # test_digit_accuracy = 100 * test_correct_digit / len(test_loader.dataset)
            # test_fixed_original_gender_accuracy_fake = 100 * test_fixed_original_gender / len(test_loader.dataset)
            # test_fixed_sampled_gender_accuracy_fake = 100 * test_fixed_sampled_gender / len(test_loader.dataset)
            # writer.add_scalar("test_set_digit_accuracy", test_digit_accuracy, epoch + 1)
            # writer.add_scalar("test_set_fixed_original_gender_accuracy_fake",
            #                   test_fixed_original_gender_accuracy_fake, epoch + 1)
            # writer.add_scalar("test_set_fixed_sampled_gender_accuracy_fake",
            #                   test_fixed_sampled_gender_accuracy_fake, epoch + 1)
            #
            # print('__________________________________________________________________________')
            # print("## Test set statistics ##")
            # print(
            #     "Utility | Digit accuracy: {} % | Fixed sampled gender accuracy: {} % | Fixed original gender accuracy: {} % ".format(
            #         test_digit_accuracy, test_fixed_sampled_gender_accuracy_fake,
            #         test_fixed_original_gender_accuracy_fake))
            #
            # # ----------------------------------------------
            # #   Save test samples
            # # ----------------------------------------------
            #
            # Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            #
            # if (epoch + 1) % save_interval == 0:
            #     print("Saving audio and spectrogram samples.")
            # save_test_samples(test_loader, audio2mel, mel2audio, models, losses, run_dir, epoch, sampling_rate,
            #                   device)
            #
            # if (epoch + 1) % checkpoint_interval == 0:
            #     save_epoch = epoch + 1
            # old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
            # if old_checkpoints:
            #     for i, _ in enumerate(old_checkpoints):
            #         os.remove(old_checkpoints[i])

        for name, model in models:
            if name == 'label_classifier' or name == 'secret_classifier':
                continue
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, name + '_epoch_{}.pt'.format(save_epoch)))
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, name + '_latest_epoch_{}.pt'.format(save_epoch)))

        for name, model in optimizers:
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, 'optimizer_' + name + '_epoch_{}.pt'.format(save_epoch)))
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir,
                                    'optimizer_' + name + '_latest_epoch_{}.pt'.format(save_epoch)))

    run = 1
    print("Run number {} completed.".format(run + 1))
    print('__________________________________________________________________________')


if __name__ == '__main__':
    main()
