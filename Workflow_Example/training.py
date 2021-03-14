import yaml
from data.dataloading import *
from models.SimpleNet import SimpleNet
from models import model_utils
import torch
import math


def main(params, config):
    data_pedestal = PedestalDataset(config)

    train_loader, validation_loader = split_dataset(data_pedestal, params['batch_size'])

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

    epochs = config['epochs']

    last_results = []
    metrics = {}
    for epoch in range(epochs):

        # TRAINING
        net.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch['input'], batch['target']
            output = net(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 5 == 4:
            net.eval()
            max_error = 0.0

            for i, batch in enumerate(validation_loader):
                inputs, targets = batch['input'], batch['target']
                output = net(inputs)
                MSE = torch.sum((output - targets) ** 2) / (len(output) * params['batch_size'])
                max_error = max(MSE, max_error)
                score = -math.log10(max_error)
                # print(epoch, score)

        if epoch > epochs - 5:
            last_results.append(score)

    final_score = min(last_results)
    metrics['default'] = final_score

    if config['experiment']['save_model'] is not None:
        PATH = config['experiment']['save_model']
        # save mode
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, PATH)


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/example_config.yaml'))
    torch.manual_seed(42)  # Always set this as we want to reproduce values
    try:
        params = model_utils.generate_default_params()
        main(params, config)
    except Exception as exc:
        raise
