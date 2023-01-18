from audio_mel_conversion import MelGanAudio2Mel, AudioMelConfig
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import CremaD, get_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from typing import Any, Dict, List, Union, Tuple
import torch
from dataclasses import dataclass
import numpy as np
import tqdm
import enum
import log


class Aggregation(enum.Enum):
    FIRST = 0
    LAST = 1
    AVERAGE = 2


class BasicModel(torch.nn.Module):

    def __init__(self, input_dims, aggregation: Aggregation):
        super().__init__()
        self.input_dims = input_dims
        self.linear1 = torch.nn.Linear(input_dims, input_dims)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout()
        self.linear2 = torch.nn.Linear(input_dims, 2)
        self.aggregation = aggregation

    def _aggregate(self, input_features):
        if self.aggregation is Aggregation.FIRST:
            return input_features[:, 0, :]
        elif self.aggregation is Aggregation.LAST:
            return input_features[:, -1, :]
        elif self.aggregation is Aggregation.AVERAGE:
            return torch.mean(input_features, dim=1)
        else:
            raise ValueError('Aggregation not recognized...')

    def __call__(self, input_features):
        x = self._aggregate(input_features)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


def compute_accuracy(preds, labels):
    if len(preds) == 1:
        preds = preds[0]
        labels = labels[0]
    else:
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
    preds = torch.argmax(preds, 1).cpu().numpy()
    return np.array(accuracy_score(labels.cpu().numpy(), preds))


def _pad(data):
    lth = data.shape[-1]
    p = 3000 - lth
    output = torch.nn.functional.pad(data, (0, p), "constant", 0)
    return output


def main(settings=None, device=None):
    if settings is None:
        settings = {'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
                    'lr': 10e-5, 'aggreagtion': Aggregation.AVERAGE, 'updates_per_evaluation': 20,
                    'updates_per_train_log_commit': 10, 'do_log': False}

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    audio2mel = MelGanAudio2Mel(AudioMelConfig())
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").get_encoder().to(device)
    whisper_encoder._freeze_parameters()

    embedding_dim = whisper_encoder.embed_positions.embedding_dim
    model = BasicModel(embedding_dim, settings['aggregation']).to(device)

    print('load data...')
    train_data, test_data = get_dataset(settings['dataset'])

    train_female_speakar_ratio = sum(1 - train_data.gender_idx) / len(train_data.gender_idx)
    test_female_speakar_ratio = sum(1 - test_data.gender_idx) / len(test_data.gender_idx)

    print(f'\nTraining set contains {train_data.n_speakers} speakers with {int(100 * train_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(train_data.gender_idx)}')
    print(f'Test set contains {test_data.n_speakers} speakers with {int(100 * test_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(test_data.gender_idx)}\n')

    train_loader = DataLoader(dataset=train_data, batch_size=settings['batch_size'],
                              num_workers=settings['num_workers'], shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=settings['batch_size'], num_workers=settings['num_workers'],
                             shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), settings['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    if settings['do_log']:
        log.init(settings, project='whisper_gender_classification',
                 run_name=f"{settings['aggregation'].name}_bsz_{settings['batch_size']}_epochs_{settings['epochs']}_lr_{settings['lr']}")

    # def get_whisper_embeddings(data):
    #     input_features = [audio.numpy() for audio in data]
    #     print(len(input_features), input_features[0].shape)
    #     input_features = processor(input_features, return_tensors="pt",
    #                                sampling_rate=settings['sampling_rate']).input_features
    #     print(input_features.shape)
    #     return whisper_encoder(input_features.to(device)).last_hidden_state

    def get_whisper_embeddings(data):
        input_features = audio2mel(data)
        # input_features = [audio.numpy() for audio in data]
        # print(len(input_features), input_features[0].shape)
        # input_features = processor(input_features, return_tensors="pt",
        #                            sampling_rate=settings['sampling_rate']).input_features
        input_features = _pad(input_features)
        log_spec = torch.maximum(input_features, input_features.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        print(log_spec.shape)
        return whisper_encoder(log_spec.to(device)).last_hidden_state

    model.train()
    optimizer.zero_grad()

    total_steps = 1

    for epoch in range(0, settings['epochs']):
        all_train_output = []
        all_train_labels = []
        all_train_loss = []
        for i, (data, secrets, _, _, _) in tqdm.tqdm(enumerate(train_loader), 'Epoch {}: Training'.format(epoch),
                                                     total=len(train_loader)):

            embeddings = get_whisper_embeddings(data)
            output = model(embeddings)

            loss = criterion(output.cpu(), secrets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_train_output.append(output)
            all_train_labels.append(secrets)
            all_train_loss.append(loss.detach().numpy())

            do_log_eval = total_steps % settings['updates_per_evaluation'] == 0
            do_log_train = total_steps % settings['updates_per_train_log_commit'] == 0
            total_steps += 1

            if settings['do_log']:
                metrics = {}
                if do_log_train:
                    metrics['train_loss'] = np.array(all_train_loss).mean().item()
                    metrics['train_acc'] = compute_accuracy(all_train_output, all_train_labels)

                    all_train_output = []
                    all_train_labels = []
                    all_train_loss = []

                if do_log_eval:
                    all_val_outputs = []
                    all_val_labels = []
                    all_val_losses = []
                    with torch.no_grad():
                        for i, (val_data, val_secrets, _, _, _) in tqdm.tqdm(enumerate(test_loader),
                                                                                      'Validation',
                                                                                      total=len(test_loader)):
                            val_embeddings = get_whisper_embeddings(val_data)
                            val_output = model(val_embeddings)
                            val_loss = criterion(val_output.cpu(), val_secrets)

                            all_val_outputs.append(val_output)
                            all_val_labels.append(val_secrets)
                            all_val_losses.append(val_loss.detach().numpy())

                    metrics['val_loss'] = np.array(all_val_losses).mean().item()
                    metrics['val_acc'] = compute_accuracy(all_val_outputs, all_val_labels)
                log._log_values(metrics, step=total_steps, commit=True)
