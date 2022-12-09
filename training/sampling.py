from training.training import preprocess_spectrograms
from plotting import comparison_plot_pcgan
import torch
import utils
import tqdm

from torch.autograd import Variable
from torch import LongTensor


def generate_samples(data_loader, audio_mel_converter, models, device):
    utils.set_mode(models, utils.Mode.EVAL)
    noise_dim = models['filter_gen'].noise_dim
    models['filter_gen'].to(device)

    samples = []

    for i, (audio, secret, label, id, _) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        audio, secret, label, id = audio[:1], secret[:1], label[:1], id[:1]

        label, secret = label.to(device), secret.to(device)
        audio = torch.unsqueeze(audio, 1)
        spectrograms = audio_mel_converter.audio2mel(audio).detach()
        original_spectrograms = spectrograms.clone()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = spectrograms.to(device)
        spectrograms = spectrograms.unsqueeze(dim=1) if spectrograms.dim() == 3 else spectrograms

        # filter_gen
        filter_z = torch.randn(spectrograms.shape[0], noise_dim).to(device)
        filtered = models['filter_gen'](spectrograms, filter_z, secret.long()).detach()

        # secret_gen
        secret_z = torch.randn(spectrograms.shape[0], noise_dim).to(device)
        fake_secret_male = Variable(LongTensor(spectrograms.size(0)).fill_(1.0), requires_grad=False).to(device)
        fake_secret_female = Variable(LongTensor(spectrograms.size(0)).fill_(0.0), requires_grad=False).to(device)
        fake_mel_male = models['secret_gen'](filtered, secret_z, fake_secret_male).detach()
        fake_mel_female = models['secret_gen'](filtered, secret_z, fake_secret_female).detach()

        samples.append({'audio': audio, 'secret': secret, 'label': label, 'id': id, 'spectrogram': spectrograms,
                        'filter_z': filter_z, 'filtered_spectrogram': filtered, 'secret_z': secret_z,
                        'fake_secret_male': fake_secret_male, 'fake_secret_female': fake_secret_female,
                        'fake_mel_male': fake_mel_male, 'fake_mel_female': fake_mel_female})

    return samples


def save_test_samples(example_dir, data_loader, audio_mel_converter, models, loss_func, epoch, sampling_rate, device):
    utils.set_mode(models, utils.Mode.EVAL)
    noise_dim = models['filter_gen'].noise_dim
    models['filter_gen'].to(device)

    for i, (input, secret, label, id, _) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        input, secret, label, id = input[:1], secret[:1], label[:1], id[:1]

        label, secret = label.to(device), secret.to(device)
        input = torch.unsqueeze(input, 1)
        spectrograms = audio_mel_converter.audio2mel(input).detach()
        original_spectrograms = spectrograms.clone()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = spectrograms.to(device)
        spectrograms = spectrograms.unsqueeze(dim=1) if spectrograms.dim() == 3 else spectrograms
        original_spectrograms = original_spectrograms.unsqueeze(dim=1) if original_spectrograms.dim() == 3 else original_spectrograms
        # spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

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

        filtered_audio = audio_mel_converter.mel2audio(unnormalized_filtered_mel).squeeze().detach().cpu()
        audio_male = audio_mel_converter.mel2audio(unnormalized_fake_mel_male).squeeze().detach().cpu()
        audio_female = audio_mel_converter.mel2audio(unnormalized_fake_mel_female).squeeze().detach().cpu()

        utils.save_sample(utils.create_run_subdir(example_dir, 'audio'), id, label, epoch, pred_label_male,
                          pred_label_female, filtered_audio, audio_male, audio_female, input.squeeze(), sampling_rate)

        comparison_plot_pcgan(original_spectrograms, unnormalized_filtered_mel, unnormalized_fake_mel_male,
                              unnormalized_fake_mel_female, secret, label, pred_secret_male, pred_secret_female,
                              pred_label_male, pred_label_female, male_distortion, female_distortion, sample_distortion,
                              utils.create_run_subdir(example_dir, 'spectrograms'), epoch, id)
    print("Success!")
