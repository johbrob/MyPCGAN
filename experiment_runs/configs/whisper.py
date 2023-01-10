from whisper_gender_classification import Aggregation
from datasets import AvailableDatasets

Q = [{'dataset': AvailableDatasets.CremaD, 'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
      'lr': 10e-5, 'aggregation': Aggregation.AVERAGE, 'updates_per_evaluation': 20,
      'updates_per_train_log_commit': 10, 'do_log': False},

     {'dataset': AvailableDatasets.CremaD, 'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
      'lr': 10e-5, 'aggregation': Aggregation.FIRST, 'updates_per_evaluation': 20,
      'updates_per_train_log_commit': 10, 'do_log': True},

     {'dataset': AvailableDatasets.CremaD, 'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
      'lr': 10e-5, 'aggregation': Aggregation.LAST, 'updates_per_evaluation': 20,
      'updates_per_train_log_commit': 10, 'do_log': True},
     ]
