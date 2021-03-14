import yaml
from data.dataloading import *
from models.SimpleNet import SimpleNet
from models import model_utils
import torch
import matplotlib.pyplot as plt


def fitted_scale(shot_df):
    I_p = shot_df[0] ** (1.38)
    B_t = shot_df[1] ** (-0.42)
    P_NBI = shot_df[2] ** (-0.25)
    delta = shot_df[3] ** (0.71)
    gamma = shot_df[4] ** (0.11)
    return 11.4 * I_p * B_t * P_NBI * delta * gamma


def denormalize(values, parameter_list, value_dict):
    j = 0
    denormed = []
    for val in values:
        max_val, min_val = value_dict[parameter_list[j]]
        # print(max_val, min_val)
        # print(values)
        denorm = val * (max_val - min_val) + min_val
        denormed.append(denorm)
        j += 1
    return denormed


def main(params, config):
    dataset = PedestalDataset(config)

    train_loader, validation_loader = split_dataset(dataset, params['batch_size'])

    if config['experiment']['load_model'] != None:
        PATH = config['experiment']['load_model']
        checkpoint = torch.load(PATH)
        # Load Model
        net = SimpleNet(params, config)
        net.load_state_dict(checkpoint['model_state_dict'])

        # Load Optimizer
        optimizer = model_utils.map_optimizer(params['optimizer'], net.parameters(), 0.0)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Assign Loss Function
        loss_func = model_utils.map_loss_func(params['loss'])

        # Set EPOCH and LOSS for retraining
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    else:
        net = SimpleNet(params, config)
        optimizer = model_utils.map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
        loss_func = model_utils.map_loss_func(params['loss'])

    metrics = {}
    target_norms = dataset.targets_norms
    input_norms = dataset.inputs_norms

    target_list = dataset.target_params
    input_list = dataset.input_params

    net.eval()

    outputs = []
    actual_array = []
    scaled_list = []

    save_path = config['experiment']['name']

    for i, batch in enumerate(validation_loader):
        inputs, targets = batch['input'], batch['target']
        for val in inputs:
            output = net(val).detach().numpy()
            output = denormalize(output, target_list, target_norms)
            outputs.append(output[0])

            normed_vals = denormalize(val.numpy(), input_list, input_norms)

        for tar in targets:
            denorm_targ = denormalize(tar.numpy(), target_list, target_norms)
            actual_array.append(denorm_targ[0])

    if config['experiment']['target'] == 'density':
        for i, batch in enumerate(validation_loader):
            inputs, targets = batch['input'], batch['target']
            for val in inputs:
                normed_vals = denormalize(val.numpy(), input_list, input_norms)
                scaled_vals = fitted_scale(normed_vals)
                scaled_list.append(scaled_vals)
        plt.scatter(actual_array, scaled_list, label='Scale Law')

    plt.scatter(actual_array, actual_array, label='Actual')
    plt.scatter(actual_array, outputs, label='NN')
    plt.legend()
    plt.ylabel('Predicted')
    plt.xlabel('Actual Density Height')
    plt.ylim(0, 12)

    plt.title('Neural Network vs Scaling Law')
    plt.savefig('./results/' + save_path)
    plt.show()


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/example_config.yaml'))
    torch.manual_seed(42)  # Always set this as we want to reproduce values
    try:
        params = model_utils.generate_default_params()
        main(params, config)
    except Exception as exc:
        print(exc)
        raise
