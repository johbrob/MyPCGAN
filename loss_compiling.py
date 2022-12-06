from torch import LongTensor, FloatTensor
from torch.autograd import Variable
import torch

class LossCompilingConfig():
    def __init__(self, lamb, eps, use_entropy_loss):
        lamb = lamb
        eps = eps
        use_entropy_loss = use_entropy_loss


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class LossCompiling():
    def __init__(self, loss_compiling_config):
        self.lamb = loss_compiling_config.lamb
        self.eps = loss_compiling_config.eps
        self.use_entropy_loss = loss_compiling_config.use_entropy_loss

        self.distortion_loss = torch.nn.L1Loss()
        self.entropy_loss = HLoss()
        self.adversary_loss = torch.nn.CrossEntropyLoss()
        self.adversary_rf_loss = torch.nn.CrossEntropyLoss()

    def for_filter_gen(self, spectrograms, filtered_mel, gender, pred_filtered_secret, use_entropy_loss, lamb, eps, device):
        ones = Variable(FloatTensor(gender.shape).fill_(1.0), requires_grad=True).to(device)
        target = ones - gender.float()
        target = target.view(target.size(0))
        distortion_loss = self.distortion_loss(filtered_mel, spectrograms)

        if use_entropy_loss:
            adversary_loss = self.entropy_loss(pred_filtered_secret)
        else:
            adversary_loss = self.adversary_loss(pred_filtered_secret, target.long())

        return adversary_loss + lamb * torch.pow(torch.relu(distortion_loss - eps), 2)


    def for_secret_gen(self, spectrograms, fake_secret, faked_mel, pred_faked_secret, lamb, eps):
        distortion_loss = self.distortion_loss(faked_mel, spectrograms)
        adversary_loss = self.adversary_loss(pred_faked_secret, fake_secret)
        return adversary_loss + lamb * torch.pow(torch.relu(distortion_loss - eps), 2)


    def for_filter_disc(self, pred_filtered_secret, gender):
        return self.adversary_loss(pred_filtered_secret, gender.long())


    def for_secret_disc(self, fake_pred_secret, real_pred_secret, gender, device):
        fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(2.0), requires_grad=False).to(device)

        real_loss = self.adversary_rf_loss(real_pred_secret, gender.long().to(device)).to(device)
        fake_loss = self.adversary_rf_loss(fake_pred_secret, fake_secret).to(device)
        return (real_loss + fake_loss) / 2
