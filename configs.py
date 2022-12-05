

class TrainingConfig:
    def __init__(self, train_batch_size, test_batch_size, train_num_workers, test_num_workers,
                 n_mels, ngf, n_resnet_layers, kernel_size, lr, embedding_dim, noise_dim,
                 n_fft, hop_length, win_length, sampling_rate):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers
        self.n_mels = n_mels
        self.ngf = ngf
        self.n_resnet_layers = n_resnet_layers
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.lr = lr

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate


def get_training_config_mini():
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

    lamb = 100
    eps = 1e-3
    use_entropy_loss = False

    save_interval = 1
    checkpoint_interval = 1

    lr = {'filter_gen': 0.00001, 'filter_disc': 0.00001, 'secret_gen': 0.00001, 'secret_disc': 0.00001}
    return TrainingConfig(train_batch_size=8, test_batch_size=1, train_num_workers=1, test_num_workers=1, n_mels=80,
                          ngf=32, n_resnet_layers=3, kernel_size=3, lr=lr, embedding_dim=16, noise_dim=10,
                          n_fft=1024, hop_length=256, win_length=1024, sampling_rate=8000)
