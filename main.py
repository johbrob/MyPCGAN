from DataManaging.AudioDatasets import AudioDataset
from torch.utils.data import DataLoader
import local_vars
from nn.modules import Audio2Mel, MelGanGenerator
from nn.models import load_modified_AlexNet, load_modified_ResNet, UNetFilter, AudioNet
import torch


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
    #spectrograms = spectrograms[:,:,:-1]
    # Normalize to zero mean unit variance, clip above 3 std and rescale to [-1,1]
    means = torch.mean(spectrograms, dim = (1,2), keepdim = True)
    stds = torch.std(spectrograms, dim = (1,2), keepdim = True)
    normalized_spectrograms = (spectrograms - means)/(3*stds + 1e-6)
    clipped_spectrograms = torch.clamp(normalized_spectrograms, -1, 1)

    return clipped_spectrograms, means, stds



def main():
    num_genders = 2
    num_digits = 10
    batch_size = 128
    sampling_rate = 8000
    segment_length = 8192
    device = 'cpu'

    n_mel_channels = 80
    ngf = 32
    n_residual_layers = 3

    # U-Net hparams
    kernel_size = 3

    train_data, test_data = AudioDataset.load()

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    fft = Audio2Mel()
    mel2audio = MelGanGenerator(n_mel_channels, ngf, n_residual_layers).to(device)
    fix_digit_spec_classifier = load_modified_ResNet(num_genders).to(device)
    fix_genders_spec_classifier = load_modified_ResNet(num_genders).to(device)
    fix_digit_spec_classifier.eval()
    fix_genders_spec_classifier.eval()

    image_width, image_height = fft.output_shape(train_data[0][0])

    # loss functions
    # dist_loss = 'l1'
    # distortion_loss = torch.nn.L1Loss()
    # entropy_loss = HLoss()
    # adversarial_loss = torch.nn.CrossEntropyLoss()
    # adversarial_loss_rf = torch.nn.CrossEntropyLoss()
    losses = {'dist_loss': torch.nn.L1Loss(), 'entropy_loss': HLoss(), 'adv_loss': torch.nn.CrossEntropyLoss(), 'adv_loss_rf': torch.nn.CrossEntropyLoss()}

    netF = UNetFilter(1, 1, chs=[8, 16, 32 , 64, 128], kernel_size=kernel_size,
                      image_width=image_width, image_height=image_height, noise_dim=10,
                      nb_classes=2, embedding_dim=16, use_cond=False).to(device)
    netFD = load_modified_AlexNet(num_genders).to(device)
    netG = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size=kernel_size,
                      image_width=image_width, image_height=image_height, noise_dim=10,
                      nb_classes=2, embedding_dim=16, use_cond=False).to(device)
    netGD = load_modified_AlexNet(num_genders + 1).to(device)

    F_lr = 0.00001

    # Optimizers
    optF = torch.optim.Adam(netF.parameters(), F_lr, betas=(0.5, 0.9))
    optFD = torch.optim.Adam(netFD.parameters(), F_lr, betas=(0.5, 0.9))
    optG = torch.optim.Adam(netG.parameters(), F_lr, betas=(0.5, 0.9))
    optGD = torch.optim.Adam(netGD.parameters(), F_lr, betas=(0.5, 0.9))

    for epoch in range(0, 10):
        correct_FD = 0
        correct_fake_GD = 0
        correct_real_GD = 0
        correct_gender_fake_GD = 0
        correct_digit = 0
        fixed_correct_gender = 0

        # Add variables to add batch losses to
        F_distortion_loss_accum = 0
        F_adversary_loss_accum = 0
        FD_adversary_loss_accum = 0
        G_distortion_loss_accum = 0
        G_adversary_loss_accum = 0
        GD_real_loss_accum = 0
        GD_fake_loss_accum = 0

        netF.train()
        netFD.train()
        netG.train()
        netGD.train()

        for i, (x, gender, digit, _) in enumerate(train_loader):
            digit = digit.to(device)
            gender = gender.to(device)
            x = torch.unsqueeze(x, 1)
            spectrograms = fft(x).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms, 1).to(device)


            # Train filter
            optF.zero_grad()

            z = torch.randn(spectrograms.shape[0], 10).to(device)
            filter_mel = netF(spectrograms, z, gender.long())
            pred_secret = netFD(filter_mel)

            ones = torch.autograd.Variable(FloatTensor(gender.shape).fill_(1.0), requires_grad=True).to(device)
            target = ones - gender.float()
            target = target.view(target.size(0))
            filter_distortion_loss = losses['dist_loss'](filter_mel, spectrograms)



if __name__ == '__main__':
    main()