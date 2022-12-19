import torch.nn.functional as F
import torchvision.models
import torch.utils.data
import torch.nn as nn
import utils
import torch
import enum

available_activations = {'relu': torch.nn.ReLU, 'leaky_relu': torch.nn.LeakyReLU}


class AvailableModels(enum.Enum):
    UNET = 0
    ALEXNET = 1
    RESNET18 = 2


def get_activation(activation):
    if activation.lower() in available_activations:
        return available_activations[activation.lower()]
    else:
        raise NotImplementedError(
            f"activation '{activation} not found. Available activations are {available_activations.keys()}'")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def double_conv(channels_in, channels_out, kernel_size, activation='relu'):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        get_activation(activation)(),
        nn.utils.weight_norm(nn.Conv2d(channels_out, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        get_activation(activation)()
    )


class UNetConfig:
    def __init__(self, channels_in=1, channels_out=1, hidden_channels=None, kernel_size=3, embedding_dim=16, noise_dim=10, activation='relu',
                 use_cond=True, n_classes=None):
        self.channels_in = channels_in
        self.channels_out = channels_out
        if hidden_channels:
            self.hidden_channels = hidden_channels
        else:
            self.hidden_channels = [8, 16, 32, 64, 128]
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.use_cond = use_cond
        self.activation = activation

        n_classes = n_classes


class UNet(nn.Module):
    def __init__(self, channels_in=1, channels_out=1, hidden_channels=None, kernel_size=3, image_width=64, image_height=64,
                 noise_dim=10, activation='relu', n_classes=2, embedding_dim=16, use_cond=True):
        super().__init__()
        # chs = [2, 4, 8, 16, 32]
        # chs=[32, 64, 128, 256, 512]
        self.channels_in = channels_in
        self.channels_out = channels_out
        if hidden_channels:
            self.hidden_channels = hidden_channels
        else:
            self.hidden_channels = [8, 16, 32, 64, 128]
        self.kernel_size = kernel_size
        self.image_width = image_width
        self.image_height = image_height
        self.noise_dim = noise_dim
        self.activation = activation
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.use_cond = use_cond

        self.embed_condition = nn.Embedding(n_classes, embedding_dim)
        self.d = 16

        # noise projection layer
        self.project_noise = nn.Linear(noise_dim,
                                       (image_width // self.d) * (image_height // self.d) * hidden_channels[-1])

        # condition projection layer
        self.project_cond = nn.Linear(embedding_dim, (image_width // self.d) * (image_height // self.d))

        self.dconv_down1 = double_conv(channels_in, hidden_channels[0], kernel_size)
        self.pool_down1 = nn.MaxPool2d(2, stride=2)

        self.dconv_down2 = double_conv(hidden_channels[0], hidden_channels[1], kernel_size)
        self.pool_down2 = nn.MaxPool2d(2, stride=2)

        self.dconv_down3 = double_conv(hidden_channels[1], hidden_channels[2], kernel_size)
        self.pool_down3 = nn.MaxPool2d(2, stride=2)

        self.dconv_down4 = double_conv(hidden_channels[2], hidden_channels[3], kernel_size)
        self.pool_down4 = nn.MaxPool2d(2, stride=2)

        self.dconv_down5 = double_conv(hidden_channels[3], hidden_channels[4], kernel_size)

        if self.use_cond:
            self.dconv_up5 = double_conv(hidden_channels[4] + hidden_channels[4] + 1 + hidden_channels[3],
                                         hidden_channels[3], kernel_size)
        else:
            self.dconv_up5 = double_conv(hidden_channels[4] + hidden_channels[4] + hidden_channels[3],
                                         hidden_channels[3], kernel_size)
        self.dconv_up4 = double_conv(hidden_channels[3] + hidden_channels[2], hidden_channels[2], kernel_size)
        self.dconv_up3 = double_conv(hidden_channels[2] + hidden_channels[1], hidden_channels[1], kernel_size)
        self.dconv_up2 = double_conv(hidden_channels[1] + hidden_channels[0], hidden_channels[0], kernel_size)
        self.dconv_up1 = nn.Conv2d(hidden_channels[0], channels_out, kernel_size=1)

    def forward(self, x, z, cond, frozen=False):
        if frozen:
            utils.freeze(self)

        bsz, chs, wth, hgt = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # B x C x H x W -> B x 8C x H x W -> B x 8C x H/2 x W/2
        conv1_down = self.dconv_down1(x)
        pool1 = self.pool_down1(conv1_down)

        # B x C x H x W -> B x 2C x H x W -> B x 2C x H/2 x W/2
        conv2_down = self.dconv_down2(pool1)
        pool2 = self.pool_down2(conv2_down)

        # B x C x H x W -> B x 2C x H x W -> B x 2C x H/2 x W/2
        conv3_down = self.dconv_down3(pool2)
        pool3 = self.pool_down3(conv3_down)

        # B x C x H x W -> B x 2C x H x W -> B x 2C x H/2 x W/2
        conv4_down = self.dconv_down4(pool3)
        pool4 = self.pool_down4(conv4_down)

        # B x C x H x W -> B x 2C x H x W
        conv5_down = self.dconv_down5(pool4)

        noise = self.project_noise(z)
        # noise = noise.reshape(bsz, 128 * chs, hgh // 16, wth // 16)
        noise = noise.reshape(bsz, self.hidden_channels[-1] * chs, wth // self.d, hgt // self.d)

        if self.use_cond:
            cond_emb = self.embed_condition(cond)
            cond_emb = self.project_cond(cond_emb)
            cond_emb = cond_emb.reshape(bsz, 1, wth // 16, hgt // 16)
            conv5_down = torch.cat((conv5_down, noise, cond_emb), dim=1)
        else:
            # conv5_down: B x 128C x H/16 x W/30
            conv5_down = torch.cat((conv5_down, noise), dim=1)

        conv5_up = F.interpolate(conv5_down, size=conv4_down.size()[-2:], mode='nearest')
        # conv5_up = F.interpolate(conv5_down, scale_factor=2, mode='nearest')

        conv5_up = torch.cat((conv4_down, conv5_up), dim=1)
        conv5_up = self.dconv_up5(conv5_up)

        conv4_up = F.interpolate(conv5_up, size=conv3_down.size()[-2:], mode='nearest')
        # conv4_up = F.interpolate(conv5_up, scale_factor=2, mode='nearest')
        conv4_up = torch.cat((conv3_down, conv4_up), dim=1)
        conv4_up = self.dconv_up4(conv4_up)

        conv3_up = F.interpolate(conv4_up, size=conv2_down.size()[-2:], mode='nearest')
        # conv3_up = F.interpolate(conv4_up, scale_factor=2, mode='nearest')
        conv3_up = torch.cat((conv2_down, conv3_up), dim=1)
        conv3_up = self.dconv_up3(conv3_up)

        conv2_up = F.interpolate(conv3_up, size=conv1_down.size()[-2:], mode='nearest')
        # conv2_up = F.interpolate(conv3_up, scale_factor=2, mode='nearest')
        conv2_up = torch.cat((conv1_down, conv2_up), dim=1)
        conv2_up = self.dconv_up2(conv2_up)

        conv1_up = self.dconv_up1(conv2_up)

        out = torch.tanh(conv1_up)

        if frozen:
            utils.unfreeze(self)

        return out


class AudioNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # conv3-100, conv3-100, maxpool2, conv3-64, maxpool2, conv3-128, maxpool2, conv3-128, maxpool2, conv3-128, maxpool2, conv3-128, maxpool2, FC-1024, FC-512,

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 100, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(100, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(nn.Linear(8192, 1024), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.Dropout(0.5))
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        conv_out = self.conv6(out)
        out = conv_out.view(-1, 8192)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, conv_out


class AlexNetConfig:
    def __init__(self, activation='relu', n_classes=None):
        self.activation = activation
        self.n_classes = n_classes


class AlexNet(nn.Module):
    def __init__(self, n_classes, activation='relu'):
        super().__init__()
        self.model = torchvision.models.AlexNet(num_classes=n_classes)

        # Make single input channel
        self.model.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Change number of output classes to num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            get_activation(activation)(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes))

    def forward(self, x, frozen=False):
        if frozen:
            utils.freeze(self)

        out = self.model(x)

        if frozen:
            utils.unfreeze(self)
        return out


class FID_AlexNet(AlexNet):
    def __init__(self, n_classes):
        super(FID_AlexNet, self).__init__(n_classes)
        # Change to single input channel
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Change to num_classes output classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class ResNetConfig:
    def __init__(self, activation='relu', pretrained=False, n_classes=None):
        self.activation = activation
        self.pretrained = pretrained
        self.n_classes = n_classes


class ResNet18(nn.Module):
    def __init__(self, n_classes, activation='relu', pretrained=False):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for name, module in self.model.named_modules():
            if hasattr(module, 'relu'):
                module.relu = get_activation(activation)()

    def forward(self, x, frozen=False):
        if frozen:
            utils.freeze(self)

        out = self.model(x)

        if frozen:
            utils.unfreeze(self)

        return out


MODEL_MAP = {AvailableModels.UNET: UNet,
             AvailableModels.ALEXNET: AlexNet,
             AvailableModels.RESNET18: ResNet18}


if __name__ == '__main__':
    print(ResNet18(2))
    print('--------------------------------')
    print(AlexNet(2))
    print('--------------------------------')
    print(UNet(1, 1, [8, 16, 32, 64, 128]))
