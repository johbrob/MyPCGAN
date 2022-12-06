from nn.models import load_modified_AlexNet, load_modified_ResNet, UNetFilter, AudioNet
from DataManaging.AudioDatasets import AudioDataset
from nn.modules import Audio2Mel, MelGanGenerator
from metrics_compiling import MetricCompiler
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


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


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


def compute_losses(spectrograms, filtered_mel, gender, pred_filtered_secret, fake_secret, faked_mel,
                   pred_fake_secret, pred_real_secret, use_entropy_loss, lamb, eps,
                   device):
    import loss_compiling
    losses = {}
    losses['filter_gen'] = loss_computation.for_filter_gen(spectrograms, filtered_mel, gender, pred_filtered_secret,
                                                           use_entropy_loss, lamb, eps, device)
    losses['secret_gen'] = loss_computation.for_secret_gen(spectrograms, fake_secret, faked_mel, pred_fake_secret, lamb, eps)
    losses['filter_disc'] = loss_computation.for_filter_disc(pred_filtered_secret, gender)
    losses['secret_disc'] = loss_computation.for_secret_disc(pred_fake_secret, pred_real_secret, gender, device)

    return losses


def forward_pass(models, spectrograms, gender, device):
    noise_dim = models['filter_gen'].noise_dim

    filter_z = torch.randn(spectrograms.shape[0], noise_dim).to(device)
    filtered_mel = models['filter_gen'](spectrograms, filter_z, gender.long())

    pred_filtered_secret = models['filter_disc'](filtered_mel)

    secret_z = torch.randn(spectrograms.shape[0], noise_dim).to(device)
    fake_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], spectrograms.shape[0]))).to(device)
    faked_mel = models['secret_gen'](filtered_mel, secret_z, fake_secret)
    pred_fake_secret = models['secret_disc'](faked_mel)
    pred_real_secret = models['secret_disc'](spectrograms)
    pred_digit = models['digit_classifier'](faked_mel)
    fixed_pred_secret = models['gender_classifier'](faked_mel)

    fake_secret = Variable(LongTensor(pred_fake_secret.size(0)).fill_(2.0), requires_grad=False).to(device)

    return filtered_mel, pred_filtered_secret, fake_secret, faked_mel, pred_fake_secret, pred_real_secret, pred_digit, \
           fixed_pred_secret, fake_secret

def training_forward_pass(optimizers, models, spectrograms, gender, loss_compiler, device):
    utils.zero_grad(optimizers)

    filtered_mel, pred_filtered_secret, fake_secret, faked_mel, pred_fake_secret, pred_real_secret, pred_digit, \
    fixed_pred_secret, fake_secret = forward_pass(models, spectrograms, gender, device)

    losses = loss_compiler(spectrograms, filtered_mel, gender, pred_filtered_secret, fake_secret, faked_mel,
                            pred_fake_secret, pred_real_secret)

    utils.backward(losses)
    utils.step(optimizers)


def filter_gen_forward_pass(optimizers, models, spectrograms, gender, losses, use_entropy_loss, lamb,
                            eps, device):
    optimizers['filter_gen'].zero_grad()

    z = torch.randn(spectrograms.shape[0], 10).to(device)
    filter_mel = models['filter_gen'](spectrograms, z, gender.long())
    pred_secret = models['filter_disc'](filter_mel)

    ones = Variable(FloatTensor(gender.shape).fill_(1.0), requires_grad=True).to(device)
    target = ones - gender.float()
    target = target.view(target.size(0))
    distortion_loss = losses['distortion'](filter_mel, spectrograms)

    if use_entropy_loss:
        adversary_loss = losses['entropy'](pred_secret)
    else:
        adversary_loss = losses['adversarial'](pred_secret, target.long())

    filter_gen_loss = adversary_loss + lamb * torch.pow(torch.relu(distortion_loss - eps), 2)
    filter_gen_loss.backward()
    optimizers['filter_gen'].step()

    return filter_mel, pred_secret, distortion_loss, adversary_loss


def secret_gen_forward_pass(optimizers, models, spectrograms, gender, losses, lamb, eps, device):
    optimizers['secret_gen'].zero_grad()
    z1 = torch.randn(spectrograms.shape[0], 10).to(device)
    filter_mel = models['filter_gen'](spectrograms, z1, gender.long())

    z2 = torch.randn(spectrograms.shape[0], 10).to(device)
    gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], spectrograms.shape[0]))).to(device)
    gen_mel = models['secret_gen'](filter_mel, z2, gen_secret)
    pred_secret = models['secret_disc'](gen_mel)
    pred_digit = models['digit_classifier'](gen_mel)
    fixed_pred_secret = models['gender_classifier'](gen_mel)

    distortion_loss = losses['distortion'](gen_mel, spectrograms)
    adversary_loss = losses['adversarial'](pred_secret, gen_secret)
    secret_gen_loss = adversary_loss + lamb * torch.pow(torch.relu(distortion_loss - eps), 2)
    secret_gen_loss.backward()
    optimizers['secret_gen'].step()

    return gen_mel, gen_secret, pred_digit, fixed_pred_secret, distortion_loss, adversary_loss


def filter_disc_forward_pass(optimizers, models, filter_mels, gender, losses):
    optimizers['filter_disc'].zero_grad()

    pred_secret = models['filter_disc'](filter_mels.detach())
    filter_disc_loss = losses['adversarial'](pred_secret, gender.long())
    filter_disc_loss.backward()
    optimizers['filter_disc'].step()

    return pred_secret, filter_disc_loss


def secret_disc_forward_pass(optimizers, models, spectrograms, gen_mel, gender, losses, device):
    optimizers['secret_disc'].zero_grad()

    real_pred_secret = models['secret_disc'](spectrograms)
    fake_pred_secret = models['secret_disc'](gen_mel.detach())

    fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(2.0), requires_grad=False).to(device)

    real_loss = losses['adversarial_rf'](real_pred_secret, gender.long().to(device)).to(device)
    fake_loss = losses['adversarial_rf'](fake_pred_secret, fake_secret).to(device)
    secret_disc_loss = (real_loss + fake_loss) / 2
    secret_disc_loss.backward()
    optimizers['secret_disc'].step()

    return real_pred_secret, fake_pred_secret, fake_secret, real_loss, fake_loss, secret_disc_loss


def validate_on_dataset(test_loader, audio2mel, models, device):
    test_correct_digit = 0
    test_fixed_original_gender = 0
    test_fixed_sampled_gender = 0

    for i, (x, gender, digit, speaker_id) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        x = torch.unsqueeze(x, 1)
        spectrograms = audio2mel(x).detach()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = torch.unsqueeze(spectrograms, 1).to(device)
        gender = gender.to(device)
        digit = digit.to(device)

        z1 = torch.randn(spectrograms.shape[0], 10).to(device)
        filter_mel = models['filter_gen'](spectrograms, z1, gender.long())
        z2 = torch.randn(filter_mel.shape[0], 10).to(device)
        gen_secret = torch.Tensor(np.random.choice([0.0, 1.0], filter_mel.shape[0])).to(device)
        gen_mel = models['secret_gen'](filter_mel, z2, gen_secret)

        pred_digit = models['digit_classifier'](gen_mel)
        fixed_pred_secret = models['gender_classifier'](gen_mel)

        # Calculate utility accuracy
        predicted = torch.argmax(pred_digit.data, 1)
        test_correct_digit += (predicted == digit).sum()

        # Calculate gender accuracy for fixed net
        fixed_predicted = torch.argmax(fixed_pred_secret.data, 1)
        test_fixed_original_gender += (fixed_predicted == gender.long()).sum()
        test_fixed_sampled_gender += (fixed_predicted == gen_secret).sum()

    return test_correct_digit, test_fixed_original_gender, test_fixed_sampled_gender


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
            digit_male = models['digit_classifier'](generated_male)
            pred_digit_male = torch.argmax(digit_male.data, 1)
            digit_female = models['digit_classifier'](generated_female)
            pred_digit_female = torch.argmax(digit_female.data, 1)

            # Predict gender
            gender_male = models['gender_classifier'](generated_male)
            pred_gender_male = torch.argmax(gender_male.data, 1)
            gender_female = models['gender_classifier'](generated_female)
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

    digit_classifier = load_modified_ResNet(train_data.n_genders).to(device)
    gender_classifier = load_modified_ResNet(train_data.n_genders).to(device)

    digit_classifier.eval()
    gender_classifier.eval()

    image_width, image_height = audio2mel.output_shape(train_data[0][0])

    losses = {'distortion': torch.nn.L1Loss(), 'entropy': HLoss(), 'adversarial': torch.nn.CrossEntropyLoss(),
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
              'secret_disc': secret_disc, 'digit_classifier': digit_classifier, 'gender_classifier': gender_classifier}

    optimizers = {
        'filter_gen': torch.optim.Adam(filter_gen.parameters(), training_config.lr['filter_gen'], betas=(0.5, 0.9)),
        'filter_disc': torch.optim.Adam(filter_disc.parameters(), training_config.lr['filter_gen'], betas=(0.5, 0.9)),
        'secret_gen': torch.optim.Adam(secret_gen.parameters(), training_config.lr['secret_gen'], betas=(0.5, 0.9)),
        'secret_disc': torch.optim.Adam(secret_disc.parameters(), training_config.lr['secret_disc'], betas=(0.5, 0.9))
    }
    return train_loader, test_loader, audio2mel, mel2audio, losses, models, optimizers


def main():
    sampling_rate = 8000
    segment_length = 8192
    device = 'cpu'

    lamb = 100
    eps = 1e-3
    use_entropy_loss = False

    save_interval = 1
    checkpoint_interval = 1

    training_config = configs.get_training_config_mini()
    loss_compiling_config = training_config.loss_compiling_config
    train_loader, test_loader, audio2mel, mel2audio, losses, models, optimizers = init_training(training_config, device)

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

    metrics_compiler = MetricCompiler()

    for epoch in range(0, 10):
        epoch_start = time.time()
        metrics_compiler.reset()

        # Add variables to add batch losses to
        filter_gen_distortion_loss = 0
        filter_gen_adversary_loss = 0
        filter_disc_adversary_loss = 0
        secret_gen_distortion_loss = 0
        secret_gen_adversary_loss = 0
        secret_disc_real_loss = 0
        secret_disc_fake_loss = 0

        utils.set_mode(models, utils.Mode.TRAIN)

        for i, (audio, gender, digit, _) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            # if i > 0:
            #     print('breaking news')
            #     break
            # print('its going ok')
            digit = digit.to(device)
            gender = gender.to(device)
            audio = torch.unsqueeze(audio, 1)
            spectrograms = audio2mel(audio).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

            # # Train filter_gen
            # filtered_mel, pred_filtered_secret, filter_gen_distortion_loss, filter_gen_adversary_loss = \
            #     filter_gen_forward_pass(optimizers, models, spectrograms, gender, losses, use_entropy_loss, lamb, eps,
            #                             device)
            # # Train secret_gen
            # faked_mel, fake_secret, pred_digit, fixed_pred_secret, secret_distortion_loss, secret_adversary_loss = \
            #     secret_gen_forward_pass(optimizers, models, spectrograms, gender, losses, lamb, eps, device)
            # # Train filter_gen
            # pred_secret, filter_disc_loss = filter_disc_forward_pass(optimizers, models, filtered_mel, gender, losses)
            # # Train secret_disc
            # real_pred_secret, fake_pred_secret, fake_secret, real_loss, fake_loss, secret_disc_loss = \
            #     secret_disc_forward_pass(optimizers, models, spectrograms, gen_mel, gender, losses, device)

            training_forward_pass(optimizers, models, spectrograms, gender, loss_compiling_config, device)

            # Compute accuracies
            metrics_compiler.compute_metrics(pred_secret, gender, fake_pred_secret, real_pred_secret,
                                             fake_secret, gen_secret, pred_digit, digit, fixed_pred_secret)

            # ----------------------------------------------
            #   Record losses
            # ----------------------------------------------

            filter_gen_distortion_loss = 0
            filter_gen_adversary_loss = 0
            filter_disc_adversary_loss = 0
            secret_gen_distortion_loss = 0
            secret_gen_adversary_loss = 0
            secret_disc_real_loss = 0
            secret_disc_fake_loss = 0
            filter_gen_distortion_loss += filter_distortion_loss.item() / (i + 1)
            F_adversary_loss_accum += filter_adversary_loss.item() / (i + 1)
            FD_adversary_loss_accum += filter_disc_loss.item() / (i + 1)
            G_distortion_loss_accum += secret_distortion_loss.item() / (i + 1)
            G_adversary_loss_accum += secret_adversary_loss.item() / (i + 1)
            GD_real_loss_accum += real_loss.item() / (i + 1)
            GD_fake_loss_accum += fake_loss.item() / (i + 1)

        # writer.add_scalar("F_adversary_loss", F_adversary_loss_accum / (i + 1), epoch + 1)
        # writer.add_scalar("G_distortion_loss", G_distortion_loss_accum / (i + 1), epoch + 1)
        # writer.add_scalar("G_adversary_loss", G_adversary_loss_accum / (i + 1), epoch + 1)
        # writer.add_scalar("FD_adversary_loss", FD_adversary_loss_accum / (i + 1), epoch + 1)
        # writer.add_scalar("GD_real_loss", GD_real_loss_accum / (i + 1), epoch + 1)
        # writer.add_scalar("GD_fake_loss", GD_fake_loss_accum / (i + 1), epoch + 1)

        metrics2log = {'F_distortion_loss_accum': F_distortion_loss_accum}
        log.metrics(metrics2log, 'train')

        # ----------------------------------------------
        #   Record accuracies
        # ----------------------------------------------

        FD_accuracy = 100 * metrics_compiler.correct_FD / len(train_loader.dataset)
        GD_accuracy_fake = 100 * metrics_compiler.correct_fake_GD / len(train_loader.dataset)
        GD_accuracy_real = 100 * metrics_compiler.correct_real_GD / len(train_loader.dataset)
        GD_accuracy_gender_fake = 100 * metrics_compiler.correct_gender_fake_GD / len(train_loader.dataset)
        fix_digit_spec_classfier_accuracy = 100 * metrics_compiler.correct_digit / len(train_loader.dataset)
        fix_gender_spec_classfier_accuracy = 100 * metrics_compiler.fixed_correct_gender / len(train_loader.dataset)

        writer.add_scalar("FD_accuracy", FD_accuracy, epoch + 1)
        writer.add_scalar("GD_accuracy_fake", GD_accuracy_fake, epoch + 1)
        writer.add_scalar("GD_accuracy_real", GD_accuracy_real, epoch + 1)
        writer.add_scalar("GD_accuracy_gender_fake", GD_accuracy_gender_fake, epoch + 1)
        writer.add_scalar("digit_accuracy", fix_digit_spec_classfier_accuracy, epoch + 1)
        writer.add_scalar("fixed_gender_accuracy_fake", fix_gender_spec_classfier_accuracy, epoch + 1)

        print('__________________________________________________________________________')
        print("Epoch {} completed | Time: {:5.2f} s ".format(epoch + 1, time.time() - epoch_start))
        print("filterGen    | Adversarial loss: {:5.5f} | Distortion loss: {:5.5f}".format(
            F_adversary_loss_accum / (i + 1), F_distortion_loss_accum / (i + 1)))
        print("filterDisc   | Filtered sample accuracy: {} %".format(FD_accuracy))
        print(
            "secretGen    | Advsarial loss: {:5.5f} | Distortion loss: {:5.5f}".format(G_adversary_loss_accum / (i + 1),
                                                                                       G_distortion_loss_accum / (
                                                                                               i + 1)))
        print("secretDisc   | Real samples: {} % | Fake samples: {} % | Sampled gender accuracy: {} % ".format(
            GD_accuracy_real, GD_accuracy_fake, GD_accuracy_gender_fake))
        print("Fix Digit accuracy: {} % | Fix gender accuracy: {} %".format(fix_digit_spec_classfier_accuracy,
                                                                            fix_gender_spec_classfier_accuracy))
        # ----------------------------------------------
        #   Compute test accuracy
        # ----------------------------------------------
        if epoch % 10 == 0:
            test_correct_digit, test_fixed_original_gender, test_fixed_sampled_gender = validate_on_dataset(
                test_loader, audio2mel, models, device)

            test_digit_accuracy = 100 * test_correct_digit / len(test_loader.dataset)
            test_fixed_original_gender_accuracy_fake = 100 * test_fixed_original_gender / len(test_loader.dataset)
            test_fixed_sampled_gender_accuracy_fake = 100 * test_fixed_sampled_gender / len(test_loader.dataset)
            writer.add_scalar("test_set_digit_accuracy", test_digit_accuracy, epoch + 1)
            writer.add_scalar("test_set_fixed_original_gender_accuracy_fake",
                              test_fixed_original_gender_accuracy_fake, epoch + 1)
            writer.add_scalar("test_set_fixed_sampled_gender_accuracy_fake",
                              test_fixed_sampled_gender_accuracy_fake, epoch + 1)

            print('__________________________________________________________________________')
            print("## Test set statistics ##")
            print(
                "Utility | Digit accuracy: {} % | Fixed sampled gender accuracy: {} % | Fixed original gender accuracy: {} % ".format(
                    test_digit_accuracy, test_fixed_sampled_gender_accuracy_fake,
                    test_fixed_original_gender_accuracy_fake))

            # ----------------------------------------------
            #   Save test samples
            # ----------------------------------------------

            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

            if (epoch + 1) % save_interval == 0:
                print("Saving audio and spectrogram samples.")
                save_test_samples(test_loader, audio2mel, mel2audio, models, losses, run_dir, epoch, sampling_rate,
                                  device)

            if (epoch + 1) % checkpoint_interval == 0:
                save_epoch = epoch + 1
                old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
                if old_checkpoints:
                    for i, _ in enumerate(old_checkpoints):
                        os.remove(old_checkpoints[i])
                for name, model in models:
                    if name == 'digit_classifier' or name == 'gender_classifier':
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
