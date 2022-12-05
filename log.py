import wandb
import time
import numpy as np


def init(config_data, project='MyPCGAN', entity='johbrob', runName='tmp-{}', disabled=False,
             add_timestamp_to_name=False):
    timestamp = time.time()
    config_data['model_ID'] = timestamp

    runName = runName.format(timestamp) if add_timestamp_to_name else runName.format('')
    mode = 'disabled' if disabled else None

    wandb.init(name=runName, config=config_data, project=project, entity=entity, mode=mode)


def data(data):
    pass


def _log_values(data, commit=True):
    wandb.log(data, commit=commit)


def metrics(data, prefix, aggregation=np.mean, selectedKeys=None, commit=False):
    targetKeys = selectedKeys if selectedKeys is not None else data.keys()
    _log_values({"{}-{}".format(prefix, k): aggregation(data[k]) for k in targetKeys}, commit=commit)
