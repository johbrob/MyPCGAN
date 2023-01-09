from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import CremaD
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union, Tuple
import torch
from dataclasses import dataclass
import numpy as np
import tqdm
import enum
import log


@dataclass
class DataCollatorSpeechClassification:
    processor: Any

    def __init__(self, encoder, sampling_rate):
        self.sampling_rate = sampling_rate
        self.encoder = encoder

    def __call__(self, data: List[Union[torch.Tensor, Tuple[str]]]) -> Dict[str, torch.Tensor]:
        input_features = [audio.numpy() for audio in data[0]]
        input_features = self.processor(input_features, return_tensors="pt",
                                        sampling_rate=self.sampling_rate).input_features
        embeddings = self.encoder(input_features).last_hidden_state

        return {'input_features': embeddings, 'labels': data[1]}
        # print(embeddings)
        # batch = {}
        # batch['labels'] = data[1]
        # return batch


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


def main():
    batch_size = 16
    num_workers = 2
    sampling_rate = 16000
    epochs = 10
    lr = 10e-4
    do_log = True
    updates_per_evaluation = 20
    updates_per_train_log_commit = 0
    aggreagtion = Aggregation.AVERAGE

    settings = {'batch_size': batch_size, 'num_workers': num_workers, 'sampling_rate': sampling_rate, 'epochs': epochs,
                'lr': lr, 'aggreagtion': aggreagtion.name, 'updates_per_evaluation': updates_per_evaluation,
                updates_per_train_log_commit: updates_per_train_log_commit}

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print('load whisper...')
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").get_encoder().to(device)
    whisper_encoder._freeze_parameters()

    embedding_dim = whisper_encoder.embed_positions.embedding_dim
    model = BasicModel(embedding_dim, aggreagtion).to(device)

    # print(model.get_encoder())

    print('load data...')
    train_data, test_data = CremaD.load()

    train_female_speakar_ratio = sum(1 - train_data.gender_idx) / len(train_data.gender_idx)
    test_female_speakar_ratio = sum(1 - test_data.gender_idx) / len(test_data.gender_idx)
    print()
    print(f'Training set contains {train_data.n_speakers} speakers with {int(100 * train_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(train_data.gender_idx)}')
    print(f'Test set contains {test_data.n_speakers} speakers with {int(100 * test_female_speakar_ratio)}% '
          f'female speakers. Total size is {len(test_data.gender_idx)}')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    if do_log:
        log.init(settings, project='whisper_gender_classification',
                 run_name=f'{aggreagtion.name}_bsz_{batch_size}_epochs_{epochs}_lr_{lr}')

    def get_whisper_embeddings(data):
        input_features = [audio.numpy() for audio in data]
        input_features = processor(input_features, return_tensors="pt",
                                   sampling_rate=sampling_rate).input_features
        return whisper_encoder(input_features.to(device)).last_hidden_state

    model.train()
    optimizer.zero_grad()

    total_steps = 0

    for epoch in range(0, epochs):
        all_train_output = []
        all_train_labels = []
        all_train_loss = []
        for i, (data, secrets, _, _, _) in tqdm.tqdm(enumerate(train_loader), 'Epoch {}: Training'.format(epochs),
                                                          total=len(train_loader)):
            embeddings = get_whisper_embeddings(data)
            output = model(embeddings)

            loss = criterion(output.cpu(), secrets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_train_output.append(output)
            all_train_labels.append(secrets)
            all_train_loss.append(loss)

            do_log_eval = total_steps % updates_per_evaluation == 0
            do_log_train = total_steps % updates_per_train_log_commit == 0

            if do_log:
                if do_log_train:
                    log._log_values({'train_loss': all_train_loss}, step=total_steps, commit=True)
                    all_train_loss = []

                if do_log_eval:
                    all_val_ouputs = []
                    all_val_labels = []
                    all_eval_losses = []
                    for i, (val_data, val_secrets, val_labels, _, _) in tqdm.tqdm(enumerate(test_loader), 'Validation',
                                                                      total=len(test_loader)):
                        val_embeddings = get_whisper_embeddings(val_data)
                        val_output = model(val_embeddings)
                        val_loss = criterion(val_output.cpu(), val_secrets)

                        all_val_ouputs.append(val_output)
                        all_val_labels.append(val_labels)
                        all_eval_losses.append(val_loss)

                    log._log_values({'val_loss': loss}, step=total_steps, commit=True)



            # log.metrics({'loss': loss}, total_steps, suffix='val', aggregation=np.mean, commit=False)
