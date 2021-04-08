import time
import math
import numpy as np
import torch

import logging

root_logger = logging.getLogger('pedsm')
logger = root_logger
logger.setLevel(logging.INFO)
fh = logging.FileHandler('pedsm.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

try:
    from PedSM.PENN.training.data_loading import prepare_dataset_torch
    from PedSM.PENN.models.FFNN import SimpleNet, SimpleNet2, Complex_Cross1, Simple_Cross2
    from PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, map_config_params, \
        EarlyStopping
    from PedSM.PENN.training.diagnostics import plot_results, plot_loss, plot_early_stopping
except Exception as exc:
    from ....PedSM.PENN.training.data_loading import prepare_dataset_torch
    from ....PedSM.PENN.models.FFNN import SimpleNet, SimpleNet2, Complex_Cross1
    from ....PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, \
        map_config_params, EarlyStopping
    from ....PedSM.PENN.training.diagnostics import plot_results, plot_loss


def train(config, params=None, warm_start_NN=None, restore_old_checkpoint=False, workers=1, verbosity=0):
    if verbosity == 0:
        logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    start = time.time()

    logger.info('Preparing Datasets')

    train_dataset, validation_dataset = prepare_dataset_torch(config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)

    logger.info('Initializing Torch Network')

    if config.get('nn_type') == 'SimpleNet':
        net = SimpleNet2(config, params)
    elif config.get('nn_type') == 'ComplexCross':
        net = Complex_Cross1(config, params)
    elif config.get('nn_type') == 'SimpleCross':
        net = Simple_Cross2(config, params)
    else:
        raise KeyError('NN Type does not yet exist... please choose from SimpleNet, ComplexCross, SimpleCross')

    logger.info('Optimizer Initialize')
    optimizer = map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
    loss_func = map_loss_func(params['loss'])

    logger.info('Start Training!')
    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['scheduler_milestones'],
                                                         gamma=0.1)

    epochs = config['epochs']
    criterion = torch.nn.MSELoss()

    last_results = []
    metrics = {}
    losses = []

    # Track the losses to determine early stopping
    train_losses = []
    validation_losses = []

    avg_train_loss = []
    avg_valid_loss = []

    # initalize the early_stopping object
    early_stopping = EarlyStopping(verbose=True, trace_func=logger.info)

    for epoch in range(epochs):

        # TRAINING
        net.train()
        max_error = 0.0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch['input'], batch['target']
            output = net(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            max_error = max(max_error, loss.detach().numpy())

            train_losses.append(loss.item())

        if config['scheduler']:
            scheduler.step()
        losses.append(max_error)

        # Validation

        net.eval()
        max_error = 0.0

        for i, batch in enumerate(test_loader):
            inputs, targets = batch['input'], batch['target']
            output = net(inputs)

            MSE = criterion(output, targets)
            MSE = torch.sqrt(MSE)
            max_error = max(MSE, max_error)
            score = -math.log10(max_error)

            loss = loss_func(output, targets)
            validation_losses.append(loss.item())

        train_loss = np.average(train_losses)
        validation_loss = np.average(validation_losses)

        avg_train_loss.append(train_loss)
        avg_valid_loss.append(validation_loss)

        train_losses = validation_losses = []

        early_stopping(validation_loss, net)

        logger.info('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                                    validation_loss,
                                                                                                    max_error))
        # print('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Validation RMSE: {:.5}'.format(epoch, train_loss, validation_loss, max_error))
        if early_stopping.early_stop:
            logger.info('Early Stopping')
            break
        last_results.append(score)

    final_score = min(last_results)
    metrics['default'] = final_score

    net.load_state_dict(torch.load('checkpoint.pt'))

    end = time.time()
    logger.info('Training Completed: Time elapsed: {:.2} Seconds'.format(end - start))
    save_path = '/home/adam/Uni_Sache/Bachelors/Thesis/next_phase/NNI_meditations/density_predictions/final_cuts/SimpleNet/trial_1'
    plot_results(net, validation_dataset, criterion,
                 save_path=save_path)
    plot_early_stopping(avg_train_loss, avg_valid_loss, save_path=save_path+'_loss')


def main(config, restore_old_checkpoint=False, **kwargs):
    train(config, warm_start_NN=None, restore_old_checkpoint=restore_old_checkpoint, **kwargs)


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Launch PENN training')
    parser.add_argument('--config', default='PedSM/PENN/training/default_config.yaml', help='Configuration File Loc')
    parser.add_argument('--load-checkpoint', default=False, help='Start from a saved checkpoint')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    # main(config=config, restore_old_checkpoint=args.load_checkpoint, verbosity=args.verbose)
    try:
        torch.manual_seed(42)  # Always set this as we want to reproduce values
        params = map_config_params(config['hyperparameters'])
        main(config=config, restore_old_checkpoint=args.load_checkpoint, verbosity=args.verbose, params=params)
    except Exception as exc:
        raise
