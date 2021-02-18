import torch
from models.first_model import SimpleNet
from models import model_utils
from data.data_proc_v2 import *
import matplotlib.pyplot as plt


def main(params):
    file_loc = 'data/daten-comma.txt'
    data_ped = DatasetPED(file_loc, params)
    train_loader, validation_loader = split_dataset(data_ped, params['batch_size'])

    model = SimpleNet(params)
    optimizer = model_utils.map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'])
    checkpoint = torch.load(params['load_model'][1])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()

    # TODO: Make a plotting function

    predictions_T = []
    truth_T = []
    for i, batch in enumerate(validation_loader):
        with torch.no_grad():
            data, targets = batch['input'], batch['target']
            output = model(data)

            predictions_T.append(output[0][0]) # Take only temperature
            truth_T.append(targets[0][0]) # ebenso

    plt.scatter(truth_T, predictions_T, c='orange')
    plt.plot(truth_T, truth_T)
    plt.show()


def generate_default_params():
    '''
    Generate default parameters for network.
    '''
    params = {
        'hidden_size_1': 32,
        'hidden_size_2': 32,
        'hidden_size_3': 32,
        'act_func': 'ReLU',
        'learning_rate': 0.025,
        'optimizer': 'Adam',
        'loss': 'SmoothL1Loss',
        'batch_size': 1,
        'save_model': (False, 'simplenet170121'),
        'load_model': (True, 'simplenet170121_v1_500')

    }
    return params  # from experiment jX3RYvtW that LR and Batch Size should be 50 and 0.005


if __name__ == '__main__':
    params = generate_default_params()
    main(params)
