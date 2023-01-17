from audio_mel_conversion import MelGanMel2Audio, MelGanAudio2Mel, AudioMelConfig, WhisperAudio2Mel, \
    CustomWhisperAudio2Mel
from finetune_melgan.discriminator import Discriminator
from torch.utils.data import DataLoader
from datasets import AvailableDatasets
from datasets import get_dataset
import torch.nn.functional as F
import torch
import utils
import time
import tqdm
import log
import numpy as np

class DiscriminatorConfig:
    def __init__(self, numD=3, ndf=16, n_layers=4, downsampling_factor=4, lambda_feat=10, cond_disc=True):
        self.num_D = numD
        self.ndf = ndf
        self.n_layers = n_layers
        self.downsampling_factor = downsampling_factor
        self.lambda_feat = lambda_feat
        self.cond_disc = cond_disc


class FineTuneMelGanConfig:
    def __init__(self, audio_mel, discriminator, dataset=AvailableDatasets.CremaD, n_train_samples=None,
                 n_test_samples=None,
                 train_batch_size=128, test_batch_size=128, netG_lr=1e-4, netD_lr=1e-4,
                 netG_betas=(0.5, 0.9), netD_betas=(0.5, 0.9), continue_training_from=False, epochs=1000, do_log=True,
                 train_num_workers=1, test_num_workers=1, do_train_shuffle=True, do_test_shuffle=True,
                 updates_per_train_log_commit=10, save_interval=10, n_samples=5):
        self.dataset = dataset
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers
        self.do_train_shuffle = do_train_shuffle
        self.do_test_shuffle = do_test_shuffle

        # audio_mel_arguments
        self.audio_mel = audio_mel
        self.discriminator = discriminator

        self.netG_lr = netG_lr
        self.netD_lr = netD_lr
        self.netG_betas = netG_betas
        self.netD_betas = netD_betas

        self.continue_training_from = continue_training_from

        self.epochs = epochs

        if self.continue_training_from:
            self.run_id = self.continue_training_from
        else:
            self.run_id = self.random_id()

        self.do_log = do_log
        self.updates_per_train_log_commit = updates_per_train_log_commit
        self.save_interval = save_interval
        self.n_samples = n_samples

    def random_id(self):
        return str(np.random.randint(0, 9, 7))[1:-1].replace(' ', '')

    def __str__(self):
        return self.run_id.rsplit('_')[0]


def initialize(settings):
    train_data, test_data = get_dataset(settings.dataset, n_train_samples=settings.n_train_samples,
                                        n_test_samples=settings.n_test_samples)

    # print split ratios
    train_female_speakar_ratio = sum(1 - train_data.gender_idx) / len(train_data.gender_idx)
    test_female_speakar_ratio = sum(1 - test_data.gender_idx) / len(test_data.gender_idx)
    print(f'Training set contains {train_data.n_speakers} speakers with {int(100 * train_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(train_data.gender_idx)}')
    print(f'Test set contains {test_data.n_speakers} speakers with {int(100 * test_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(test_data.gender_idx)}')

    # init dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=settings.train_batch_size,
                              num_workers=settings.train_num_workers, shuffle=settings.do_train_shuffle)
    test_loader = DataLoader(dataset=test_data, batch_size=settings.test_batch_size,
                             num_workers=settings.test_num_workers, shuffle=settings.do_test_shuffle)

    netG = MelGanMel2Audio(settings.audio_mel)
    netD = Discriminator(settings.discriminator.num_D, settings.discriminator.ndf, settings.discriminator.n_layers,
                         settings.discriminator.downsampling_factor)
    # fft = MelGanAudio2Mel(settings.audio_mel)
    fft = CustomWhisperAudio2Mel(settings.audio_mel)

    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    return train_loader, test_loader, netG, netD, fft, optG, optD


def main(settings, device):
    train_loader, test_loader, netG, netD, fft, optG, optD = initialize(settings)
    print(netG)
    print(netD)
    netG.to(device)
    netD.to(device)

    if settings.continue_training_from and settings.continue_training_from.exists():
        netG.load_state_dict(torch.load(settings.continue_training_from / "netG.pt"))
        optG.load_state_dict(torch.load(settings.continue_training_from / "optG.pt"))
        netD.load_state_dict(torch.load(settings.continue_training_from / "netD.pt"))
        optD.load_state_dict(torch.load(settings.continue_training_from / "optD.pt"))

    save_path = utils.create_run_subdir('finetune_melgan', settings.run_id, '')

    # init wandb
    if settings.do_log:
        log.init(vars(settings), project='melgan_finetuning', run_name='run_1')

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, (audio, _, _, _, _) in tqdm.tqdm(enumerate(test_loader), 'Save test samples', total=len(train_loader)):
        x_t = audio.to(device)
        s_t = fft(x_t).detach()

        test_voc.append(s_t.to(device))
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        utils.save_audio_file(save_path + ("original_%d.wav" % i), fft.sampling_rate, audio)
        # writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=22050)

        if i == settings.n_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True
    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, settings.epochs + 1):
        for iterno, (audio, _, _, _, _) in tqdm.tqdm(enumerate(train_loader), 'Training', total=len(train_loader)):
            x_t = audio.to(device)
            s_t = fft(x_t).detach()

            x_pred_t = netG(s_t.to(device))
            x_pred_t = x_pred_t[..., :x_t.shape[-1]]

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.to(device).detach())
            D_real = netD(x_t.unsqueeze(dim=1).to(device))

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.to(device))

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (settings.discriminator.n_layers + 1)
            D_weights = 1.0 / settings.discriminator.num_D
            wt = D_weights * feat_weights
            for i in range(settings.discriminator.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + settings.discriminator.lambda_feat * loss_feat).backward()
            optG.step()

            ######################
            # Update logs #
            ######################
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            do_log_train = steps % settings.updates_per_train_log_commit == 0
            if settings.do_log and do_log_train:
                if do_log_train:
                    costs = np.asarray(costs).mean(0)
                    metrics = {"loss/discriminator": costs[0],
                               "loss/generator": costs[1],
                               "loss/feature_matching": costs[2],
                               "loss/mel_reconstruction": costs[3]}
                    log.metrics(metrics, steps, suffix='train', commit=False)
                    costs = []
                log.metrics({'Epoch': epoch + (i / len(train_loader))}, steps, commit=True)

            # writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            # writer.add_scalar("loss/generator", costs[-1][1], steps)
            # writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            # writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % settings.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        utils.save_audio_file(save_path + ("generated_%d.wav" % i), fft.sampling_rate, pred_audio)
                        # writer.add_audio(
                        #     "generated/sample_%d.wav" % i,
                        #     pred_audio,
                        #     epoch,
                        #     sample_rate=22050,
                        # )

                torch.save(netG.state_dict(), save_path + "netG.pt")
                torch.save(optG.state_dict(), save_path + "optG.pt")

                torch.save(netD.state_dict(), save_path + "netD.pt")
                torch.save(optD.state_dict(), save_path + "optD.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), save_path + "best_netD.pt")
                    torch.save(netG.state_dict(), save_path + "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            # if steps % args.log_interval == 0:
            #     print(
            #         "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
            #             epoch,
            #             iterno,
            #             len(train_loader),
            #             1000 * (time.time() - start) / args.log_interval,
            #             np.asarray(costs).mean(0),
            #         )
            #     )
            #     costs = []
            #     start = time.time()
