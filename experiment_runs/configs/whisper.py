from whisper_gender_classification import Aggregation

class ModelConfig:
    def __init__(self, model, model_config, pretrained_path=None):
        self.model = model
        self.args = model_config
        self.pretrained_path = pretrained_path


Q = [{'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
                'lr': 10e-5, 'aggregation': Aggregation.AVERAGE, 'updates_per_evaluation': 20,
                'updates_per_train_log_commit': 10, 'do_log': True},
     {'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
                'lr': 10e-5, 'aggregation': Aggregation.FIRST, 'updates_per_evaluation': 20,
                'updates_per_train_log_commit': 10, 'do_log': True},
    {'batch_size': 16, 'num_workers': 2, 'sampling_rate': 16000, 'epochs': 10,
                'lr': 10e-5, 'aggregation': Aggregation.LAST, 'updates_per_evaluation': 20,
                'updates_per_train_log_commit': 10, 'do_log': True},
]
