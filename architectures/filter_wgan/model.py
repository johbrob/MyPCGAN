from architectures.utils import preprocess_spectrograms, create_model_from_config
from architectures.filter_wgan.loss_computations import compute_losses, LossConfig
from architectures.filter_wgan.metrics_computations import compute_metrics
from neural_networks.whisper_encoder import WhisperEncoder
from utils import Mode
import torch
import glob
import os
import librosa


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class FilterWGANConfig:
    def __init__(self, filter_gen_config, filter_disc_config, whisper_config, audio2mel_config, mel2audio_config,
                 lr=None, betas=None, generate_both_secrets=True, filter_gen_label_smoothing=0,
                 filter_disc_label_smoothing=0, loss_config=LossConfig()):

        self.architecture = FilterWGAN

        self.filter_gen = filter_gen_config
        self.filter_disc = filter_disc_config

        self.whisper = whisper_config

        if lr is None:
            self.lr = {'filter_gen': 0.0001, 'filter_disc': 0.0004}
        else:
            self.lr = lr

        if betas is None:
            self.betas = {'filter_gen': (0.5, 0.999), 'filter_disc': (0.5, 0.999)}
        else:
            self.betas = betas

        self.generate_both_secrets = generate_both_secrets
        self.filter_gen_label_smoothing = filter_gen_label_smoothing
        self.filter_disc_label_smoothing = filter_disc_label_smoothing

        self.loss = loss_config

        self.audio2mel = audio2mel_config
        self.mel2audio = mel2audio_config


class FilterWGAN:
    def __init__(self, config, device):

        self.audio2mel = config.audio2mel.model(config.audio2mel.args)
        self.mel2audio = config.mel2audio.model(config.mel2audio.args)
        if config.mel2audio.pretrained_path:
            self.mel2audio.load_state_dict(
                torch.load(config.mel2audio.pretrained_path, map_location=torch.device('cpu')))

        self.audio_encoder = config.whisper.model(config.whisper.args, device)

        image_width, image_height = self.audio2mel.output_shape(config.data_info['sample_data'][0])
        self.n_secrets = config.data_info['n_secrets']
        self.n_labels = config.data_info['n_labels']

        config.filter_gen.args.image_width = image_width
        config.filter_gen.args.image_height = image_height
        config.filter_gen.args.n_classes = self.n_secrets
        config.filter_disc.args.n_classes = self.n_secrets

        self.loss_funcs = {'filter_gen_distortion': torch.nn.L1Loss(), 'filter_gen_entropy': HLoss(),
                           'filter_gen_adversarial': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.filter_gen_label_smoothing),
                           'filter_disc': torch.nn.CrossEntropyLoss(
                               label_smoothing=config.filter_disc_label_smoothing)}

        self.filter_gen = create_model_from_config(config.filter_gen).to(device)
        self.filter_disc = create_model_from_config(config.filter_disc).to(device)

        self.lr = config.lr
        betas = config.betas
        self.optimizers = {
            'filter_gen': torch.optim.Adam(self.filter_gen.parameters(), self.lr['filter_gen'], betas['filter_gen']),
            'filter_disc': torch.optim.Adam(self.filter_disc.parameters(), self.lr['filter_disc'],
                                            betas['filter_disc']),
        }

        self.loss_config = config.loss
        self.sampling_rate = config.audio2mel.args.sampling_rate
        self.device = device

        self.means = None
        self.stds = None

    def preprocess(self, audio):
        mels = self.audio2mel(audio).detach()  # mels: (bsz, n_mels, frames)
        # mels, self.means, self.stds = preprocess_spectrograms(mels)
        mels = mels.to(self.device)  # mels: (bsz, 1, n_mels, frames)
        return mels

    def postprocess(self, audio, mel, filtered_mel):

        # unnormalize
        # filtered_mel = (torch.squeeze(filtered_mel.cpu(), 1) * 3 * std + mean)
        # fake_mel_male = (torch.squeeze(fake_mel_male.cpu(), 1) * 3 * std + mean)
        # fake_mel_female = (torch.squeeze(fake_mel_female.cpu(), 1) * 3 * std + mean)
        # mel = torch.squeeze(mel.cpu() * 3 * std + mean)

        print(mel.shape, filtered_mel.shape)

        if self.stds and self.means:
            mel = mel.cpu() * 3 * self.stds + self.means
            filtered_mel = (filtered_mel.cpu() * 3 * self.stds) + self.means

        audio = audio.squeeze().cpu()
        a2m2_audio = self.mel2audio(mel.squeeze().cpu())
        filtered_audio = self.mel2audio(filtered_mel.squeeze().cpu())

        a2m2_audio = a2m2_audio.squeeze()
        filtered_audio = filtered_audio.squeeze()

        print(audio.shape, a2m2_audio.shape, filtered_audio.shape, )

        return audio, a2m2_audio, filtered_audio

    def _filter_gen_forward_pass(self, mels, secrets):
        mel_encodings = self.audio_encoder(mels)
        filter_z = torch.randn(mels.shape[0], self.filter_gen.noise_dim).to(mels.device)
        filtered_mels = self.filter_gen(mels.unsqueeze(dim=1), filter_z, secrets.long())  # (bsz, 1, n_mels, frames)
        filtered_secret_preds_gen = self.filter_disc(filtered_mels, frozen=True)  # (bsz, n_secret)
        filtered_mel_encodings = self.audio_encoder(filtered_mels.squeeze(dim=1))
        return {'filtered_mel': filtered_mels, 'filtered_secret_score': filtered_secret_preds_gen,
                'mel_encodings': mel_encodings, 'filtered_mel_encodings': filtered_mel_encodings}

    def _filter_disc_forward_pass(self, mels, filtered_mels):
        filtered_secret_preds_disc = self.filter_disc(filtered_mels.detach())  # (bsz, n_secret)
        unfiltered_secret_preds_disc = self.filter_disc(mels.unsqueeze(dim=1).detach(), frozen=True)  # (bsz, n_secret)
        return {'filtered_secret_score': filtered_secret_preds_disc,
                'unfiltered_secret_score': unfiltered_secret_preds_disc}

    def forward_pass(self, audio, secrets, labels):
        mels = self.preprocess(audio)

        filter_gen_output = self._filter_gen_forward_pass(mels, secrets)
        filter_disc_output = self._filter_disc_forward_pass(mels, filter_gen_output['filtered_mel'])

        losses = compute_losses(self.loss_funcs, mels, secrets, filter_gen_output, filter_disc_output, self.loss_config)
        batch_metrics = compute_metrics(mels, secrets, labels, filter_gen_output, filter_disc_output, losses,
                                        self.loss_funcs)

        return batch_metrics, losses

    def generate_sample(self, audio, secret, label, id, epoch):

        mel = self.preprocess(audio)

        # filter_gen
        filter_z = torch.randn(mel.shape[0], self.filter_gen.noise_dim).to(self.device)
        filtered_mel = self.filter_gen(mel.unsqueeze(dim=1), filter_z, secret.long())

        # secret disc
        pred_secret = torch.argmax(self.filter_disc(filtered_mel.detach().clone()).data, 1)

        audio, a2m2_audio, filtered_audio = self.postprocess(audio, mel, filtered_mel)
        print(audio.shape, a2m2_audio.shape, filtered_audio.shape)

        output = self._build_output_dict(id, label, secret, epoch, audio, pred_secret, filtered_audio, a2m2_audio)

        [print(v.shape) for k, v in output.items()]
        return output

    def set_mode(self, mode):
        if mode == Mode.TRAIN:
            self.filter_gen.train()
            self.filter_disc.train()
        elif mode == Mode.EVAL:
            self.filter_gen.eval()
            self.filter_disc.eval()

    def save(self, checkpoint_dir, epoch):
        models = {'filter_gen': self.filter_gen, 'filter_disc': self.filter_disc}

        old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
        if old_checkpoints:
            for i, _ in enumerate(old_checkpoints):
                os.remove(old_checkpoints[i])
        for k, v in models.items():
            if k == 'label_classifier' or k == 'secret_classifier':
                continue
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'{k}_epoch_{epoch}.pt'))
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'{k}_latest_epoch_{epoch}.pt'))
        for k, v in self.optimizers.items():
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{k}_epoch_{epoch}.pt'))
            torch.save(v.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{k}_latest_epoch_{epoch}.pt'))

    def _build_output_dict(self, id, label, secret, epoch, audio, pred_secret, filtered_audio, a2m2_audio):
        speaker_digit_str = f'speaker_{id.item()}_label_{label.item()}'
        speaker_digit_epoch_str = speaker_digit_str + f'_epoch_{epoch}'

        filtered_audio_file = speaker_digit_epoch_str + f'gender_{secret}_filtered_predicted_gender_{pred_secret.item()}.wav'
        a2m2a_file = speaker_digit_epoch_str + '_a2m2a.wav'
        original_audio_file = speaker_digit_str + '.wav'

        return {original_audio_file: audio,
                filtered_audio_file: filtered_audio,
                a2m2a_file: a2m2_audio}


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(outputs=model_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
