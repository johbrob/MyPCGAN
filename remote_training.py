import argparse
from DataManaging import AvailableDatasets
import configs
from training import init_training

available_dataset = {
    'audiomnist': AvailableDatasets.AudioMNIST
}

available_settings = {
    'debug': configs.create_debug_config(),
    'github_default': configs.create_github_default_config(),
    'github_lower_lr': configs.create_github_lower_lr_config(),
}


def verify_args(args):
    assert args['dataset'] in available_dataset
    assert args['settings'] in available_settings
    assert isinstance(args['gpu'], int) or args['gpu'].isdigit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="name of dataset")
    parser.add_argument("-s", "--settings", help="which pre-defined setting to use", default='debug')
    parser.add_argument("-g", "--gpu", help="specify which gpu to use. Will use cpu if no gpu is available",
                        default='0')
    args = parser.parse_args()
    args = vars(args)

    args = {'dataset': 'audiomnist', 'settings': 'debug', 'gpu': 0}
    verify_args(args)

    init_training(dataset_name=available_dataset[args['dataset']],
                  experiment_settings=available_settings[args['settings']], device=args['gpu'])