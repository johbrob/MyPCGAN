from datasets import create_audiomnist, create_crema_d
import argparse
import local_vars

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="name of dataset", default='audiomnist')
    args = parser.parse_args()
    args = vars(args)

    args = {'dataset': 'cremad'}

    if args['dataset'] == 'audiomnist':
        create_audiomnist(local_vars.AUDIO_MNIST_PATH, local_vars.PREPROCESSED_DATA_PATH, 0.20, 8192, 8000, True)
    elif args['dataset'] == 'cremad':
        create_crema_d(local_vars.CREMA_D_PATH, local_vars.PREPROCESSED_DATA_PATH, 0.20, 40000, 16000, True)
