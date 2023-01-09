from architectures.alt_gen.metrics_compiling import compute_metrics
from architectures.utils import create_model_from_config
from architectures.alt_gen import loss_compiling
from utils import Mode
import torch
import tqdm


class OneStepGanConfig:
    def __init__(self, gen_config, fake_disc_config, secret_disc_config, lr=None, betas=None,
                 generate_both_secrets=True, fake_disc_label_smoothing=0, secret_disc_label_smoothing=0):
        self.architecture = OneStepGAN

        self.gen = gen_config
        self.fake_disc = fake_disc_config
        self.secret_disc = secret_disc_config

        if lr is None:
            self.lr = {'gen': 0.0001, 'fake_disc': 0.0004, 'secret_disc': 0.0004}
        else:
            self.lr = lr

        if betas is None:
            self.betas = {'gen': (0.5, 0.999), 'fake_disc': (0.5, 0.999), 'secret_disc': (0.5, 0.999)}
        else:
            self.betas = betas

        self.generate_both_secrets = generate_both_secrets
        self.fake_disc_label_smoothing = fake_disc_label_smoothing
        self.secret_disc_label_smoothing = secret_disc_label_smoothing


class OneStepGAN:

    def __init__(self, config, device):

        config.gen.args.image_width = config.image_width
        config.gen.args.image_height = config.image_height
        config.gen.args.n_classes = config.n_genders
        config.fake_disc.args.n_classes = config.n_genders
        config.secret_disc.args.n_classes = config.n_genders

        # models
        self.gen = create_model_from_config(config.gen).to(device)
        self.fake_disc = create_model_from_config(config.fake_disc).to(device)
        self.secret_disc = create_model_from_config(config.secret_disc).to(device)

        # optimizers
        self.lr = config.lr
        betas = config.betas
        self.optimizers = {
            'gen': torch.optim.Adam(self.gen.parameters(), self.lr['gen'], betas['gen']),
            'fake_disc': torch.optim.Adam(self.fake_disc.parameters(), self.lr['fake_disc'], betas['fake_disc']),
            'secret_disc': torch.optim.Adam(self.secret_disc.parameters(), self.lr['secret_disc'], betas['secret_disc'])
        }

        self.loss_funcs = {
            'gen_fake_ce': torch.nn.CrossEntropyLoss(), 'gen_secret_ce': torch.nn.CrossEntropyLoss(),
            'fake_disc': torch.nn.CrossEntropyLoss(label_smoothing=config.fake_disc_label_smoothing),
            'secret_disc': torch.nn.CrossEntropyLoss(label_smoothing=config.secret_disc_label_smoothing)
        }

        self.generate_both_secrets = config.generate_both_secrets

    def _gen_forward_pass(self, mels):
        z = torch.randn(mels.shape[0], self.gen.noise_dim).to(mels.device)
        fake_secret = torch.randint(0, 1, mels.shape[0:1]).to(mels.device)  # (bsz,)
        fake_mels = self.gen(mels, z, fake_secret)  # (bsz, 1, n_mels, frames)

        fake_mels_score = self.fake_disc(fake_mels, frozen=True)  # (bsz, n_secret)
        secret_score = self.secret_disc(fake_mels, frozen=True)  # (bsz, n_secret)

        if self.generate_both_secrets:
            alt_fake_mels = self.gen(mels, z, 1 - fake_secret, frozen=True)  # (bsz, 1, n_mels, frames)
            alt_fake_mels_score = self.fake_disc(alt_fake_mels, frozen=True)  # (bsz, n_secrets + 1)
            alt_secret_score = self.secret_disc(alt_fake_mels, frozen=True)  # (bsz, n_secrets + 1)
        else:
            alt_fake_mels, alt_fake_mels_score, alt_secret_score = None, None, None

        return {'fake_mel': fake_mels, 'fake_secret': fake_secret, 'fake_mel_score': fake_mels_score,
                'secret_score': secret_score, 'alt_fake_mel': alt_fake_mels, 'alt_fake_mel_score': alt_fake_mels_score,
                'alt_secret_score': alt_secret_score}

    def _fake_disc_forward_pass(self, mels, fake_mels, alt_fake_mels):
        mel_score = self.fake_disc(mels.detach())  # (bsz, n_secret)
        fake_mel_score = self.fake_disc(fake_mels.detach())  # (bsz, n_secret)

        if self.generate_both_secrets:
            alt_fake_mel_score = self.fake_disc(alt_fake_mels.detach())  # (bsz, n_secret)
        else:
            alt_fake_mel_score = None

        return {'mel_score': mel_score, 'fake_mel_score': fake_mel_score, 'alt_fake_mel_score': alt_fake_mel_score}

    def _secret_disc_forward_pass(self, mels, fake_mel, alt_fake_mel):
        real_secret_score = self.secret_disc(mels)  # (bsz, n_secrets + 1)
        fake_secret_score = self.secret_disc(fake_mel.detach().clone())  # (bsz, n_secrets + 1)

        if self.generate_both_secrets:
            alt_fake_secret_score = self.secret_disc(alt_fake_mel.detach().clone())
        else:
            alt_fake_secret_score = None

        return {'fake_secret_score': fake_secret_score, 'real_secret_score': real_secret_score,
                'alt_fake_secret_score': alt_fake_secret_score}

    def forward_pass(self, mels, secrets):
        assert mels.dim() == 4

        gen_output = self._gen_forward_pass(mels)
        fake_disc_output = self._fake_disc_forward_pass(mels, gen_output['fake_mel'], gen_output['alt_fake_mel'])
        secret_disc_output = self._secret_disc_forward_pass(mels, gen_output['fake_mel'], gen_output['alt_fake_mel'])

        losses = loss_compiling.compute_losses(self.loss_funcs, mels, secrets, gen_output, fake_disc_output,
                                               secret_disc_output, self.generate_both_secrets)
        batch_metrics = compute_metrics(secrets, gen_output, fake_disc_output, secret_disc_output, losses)

        return batch_metrics, losses


    def set_mode(self, mode):
        if mode == Mode.TRAIN:
            self.gen.train()
            self.fake_disc.train()
            self.secret_disc.train()
        elif mode == Mode.EVAL:
            self.gen.eval()
            self.fake_disc.eval()
            self.secret_disc.eval()

    def save_test_samples(self, example_dir, data_loader, audio_mel_converter, models, loss_func, epoch, sampling_rate,
                          device,
                          n_samples_generated):
        self.set_mode(Mode.EVAL)
        noise_dim = models['filter_gen'].noise_dim
        models['filter_gen'].to(device)

        with torch.no_grad():
            for i, (data, secret, label, id, _) in tqdm.tqdm(enumerate(data_loader), 'Generating Samples',
                                                             total=len(data_loader)):

                if i >= n_samples_generated:
                    break
                # data: (1 x seq_len), secret: (1,), label: (1,), id: (1,)
                data, secret, label, id = data[:1], secret[:1], label[:1], id[:1]

                label, secret = label.to(device), secret.to(device)
                original_mel = audio_mel_converter.audio2mel(data).detach()
                mel, means, stds = preprocess_spectrograms(original_mel)
                mel = mel.unsqueeze(dim=1).to(device)


                # filter_gen
                filter_z = torch.randn(mel.shape[0], noise_dim).to(device)
                filtered = models['filter_gen'](mel, filter_z, secret.long()).detach()

                # predict label
                pred_label_male = torch.argmax(models['label_classifier'](fake_mel_male).data, 1)
                pred_label_female = torch.argmax(models['label_classifier'](fake_mel_female).data, 1)

                # predict secret
                pred_secret_male = torch.argmax(models['secret_classifier'](fake_mel_male).data, 1)
                pred_secret_female = torch.argmax(models['secret_classifier'](fake_mel_female).data, 1)

                # distortions
                filtered_distortion = loss_func['secret_gen_distortion'](mel, filtered).item()
                male_distortion = loss_func['secret_gen_distortion'](mel, fake_mel_male).item()
                female_distortion = loss_func['secret_gen_distortion'](mel, fake_mel_female).item()
                sample_distortion = loss_func['secret_gen_distortion'](fake_mel_male, fake_mel_female).item()

                unnormalized_filtered_mel = torch.squeeze(filtered, 1).to(device) * 3 * stds.to(device) + means.to(
                    device)
                unnormalized_fake_mel_male = torch.squeeze(fake_mel_male, 1).to(device) * 3 * stds.to(
                    device) + means.to(device)
                unnormalized_fake_mel_female = torch.squeeze(fake_mel_female, 1).to(device) * 3 * stds.to(
                    device) + means.to(
                    device)
                unnormalized_spectrograms = torch.squeeze(mel.to(device) * 3 * stds.to(device) + means.to(device))

                # TODO: These could be on gpu if we use MelGAnGenerator
                filtered_audio = audio_mel_converter.mel2audio(unnormalized_filtered_mel.squeeze().detach().cpu())
                audio_male = audio_mel_converter.mel2audio(unnormalized_fake_mel_male.squeeze().detach().cpu())
                audio_female = audio_mel_converter.mel2audio(unnormalized_fake_mel_female.squeeze().detach().cpu())

                utils.save_sample(utils.create_subdir(example_dir, 'audio'), id, label, epoch, pred_label_male,
                                  pred_label_female, filtered_audio, audio_male, audio_female, data.squeeze(),
                                  sampling_rate)

                comparison_plot_pcgan(original_mel, unnormalized_filtered_mel, unnormalized_fake_mel_male,
                                      unnormalized_fake_mel_female, secret, label, pred_secret_male, pred_secret_female,
                                      pred_label_male, pred_label_female, male_distortion, female_distortion,
                                      sample_distortion,
                                      utils.create_subdir(example_dir, 'spectrograms'), epoch, id)
        print("Success!")