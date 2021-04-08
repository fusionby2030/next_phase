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
    from PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, map_config_params, \
        EarlyStopping, map_model
    from PedSM.PENN.training.diagnostics import plot_results, plot_early_stopping
except Exception as exc:
    from ....PedSM.PENN.training.data_loading import prepare_dataset_torch
    from ....PedSM.PENN.models.FFNN_torch import SimpleNet, SimpleNet2, PedDeepCross
    from ....PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, \
        map_config_params, EarlyStopping
    from ....PedSM.PENN.training.diagnostics import plot_results


def train_epoch(net, optimizer, loss_func, train_loader, test_loader, scheduler=None, criterion=None):
    """
    -------------------------------------------------
    Ye old training loop
    Loop through the train loader, back propogate
    then loop through the test loader, get the losses for validation

    Net, Optimizer, and Loss_func, Scheduler are all instances of torch.nn
    the loaders must be also torch.utils.data.DataLoader instances
    early stopping is a self made class found in the torch_utils file

    __________________________________________________
    :param net: Torch Net to be trained
    :param optimizer: instance of
    :param loss_func: Loss function
    :param train_loader:
    :param test_loader:
    :param scheduler:
    :param criterion:
    :return:
    """
    # The losses over the epoch, we will average them later
    train_losses = []
    validation_losses = []

    # Set model into training mode
    net.train()
    max_error = 0.0 # placeholder value

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = batch['input'], batch['target']
        output = net(inputs)
        # the reason for loss_t is so it is not confused with validation loss, i.e., loss_training
        loss_t = loss_func(output, targets)
        loss_t.backward()
        optimizer.step()
        max_error = max(max_error, loss_t.detach().numpy())
        # keep track of losses
        train_losses.append(loss_t.item())

    # Learning rate scheduler milestones can be set in config
    if scheduler is not None:
        scheduler.step()

    # Test on the unseen data!
    # Set model into eval mode so no optimization or backprop occurs, i.e., it won't learn
    net.eval()
    max_error = 0.0 # reset the placeholder value

    for i, batch in enumerate(test_loader):
        inputs, targets = batch['input'], batch['target']
        output = net(inputs)

        # the criterion is for us the secondary objective, i.e., we want to minimize the RMSE
        # So here want our score to be the RMSE
        MSE = criterion(output, targets)
        MSE = torch.sqrt(MSE)
        max_error = max(MSE, max_error)

        loss = loss_func(output, targets)
        validation_losses.append(loss.item())

    # Get average losses over the epoch
    train_loss = np.average(train_losses)
    validation_loss = np.average(validation_losses)

    return train_loss, validation_loss, max_error


def train(config, params=None, warm_start_NN=None, restore_old_checkpoint=False, workers=1, verbosity=0,
           diagnostics=False):
    """
    ---------------------------------
    Implements the train per epoch to train a pytorch model based on the configuration file.
    It will also save the model to the location specified in the config.

    ---------------------------------
    :param config: see example config in repository
    :param params: this is mapped from the config, if not generated using default parameters that generally cover the bases for all models here
    :param warm_start_NN: this needs to be a loaded pytorch module, so you can define your own model, and use that
    :param restore_old_checkpoint: TBD
    :param workers: TBD
    :param verbosity: logging
    :param diagnostics: output results figure
    :return: None, but maybe in future it returns the net
    """
    if verbosity == 0:
        logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    logger.info('Preparing Datasets')

    train_dataset, validation_dataset = prepare_dataset_torch(config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)

    logger.info('Initializing Torch Network')

    if warm_start_NN is not None:
        net = warm_start_NN
    else:
        net = map_model(config, params)

    logger.info('Optimizer Initialize')
    optimizer = map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
    loss_func = map_loss_func(params['loss'])

    logger.info('Start Training!')
    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['scheduler_milestones'],
                                                         gamma=0.1)

    epochs = config['epochs']
    criterion = torch.nn.MSELoss()

    # Track the losses to determine early stopping
    avg_train_loss = []
    avg_valid_loss = []

    # initalize the early_stopping object
    early_stopping = EarlyStopping(verbose=True, trace_func=logger.info, path=config['save_model_path'])

    for epoch in range(epochs):
        train_loss, validation_loss, RMSE = train_epoch(net, optimizer, loss_func, train_loader=train_loader,
                                                  test_loader=test_loader, scheduler=scheduler, criterion=criterion)
        if early_stopping is not None:
            early_stopping(validation_loss, net, RMSE)

        avg_train_loss.append(train_loss)
        avg_valid_loss.append(validation_loss)

        RMSE = early_stopping.RMSE
        logger.info('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Best Validation RMSE: {:.5}'.format(epoch, train_loss, validation_loss, RMSE))
        print('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Validation RMSE: {:.5}'.format(epoch, train_loss, validation_loss, RMSE))
        if early_stopping.early_stop:
            logger.info('Early Stopping')
            break

    if diagnostics:
        net.load_state_dict(torch.load(config['save_model_path']))
        try:
            save_path = config['diagnostics_path']
        except KeyError as exc:
            Warning('No Path to Save Diagnostics, saving to root dir')
            save_path = 'trial_run'

        plot_results(net, validation_dataset, criterion,
                     save_path=save_path, config=config)
        plot_early_stopping(avg_train_loss, avg_valid_loss, save_path=save_path + '_loss')


def train_full(config, params=None, warm_start_NN=None, restore_old_checkpoint=False, workers=1, verbosity=0,
          cross_fold_loaders=None):
    """
    OLD AND OUTDATED, IS replaced by train.


    :param config:
    :param params:
    :param warm_start_NN:
    :param restore_old_checkpoint:
    :param workers:
    :param verbosity:
    :param cross_fold_loaders:
    :return:
    """
    if verbosity == 0:
        logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    start = time.time()

    logger.info('Preparing Datasets')
    if cross_fold_loaders is not None:
        pass
    else:
        train_dataset, validation_dataset = prepare_dataset_torch(config)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)

    logger.info('Initializing Torch Network')

    net = map_model(config, params)

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
    early_stopping = EarlyStopping(verbose=True, trace_func=logger.info, path=config['save_model_path'])

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

    net.load_state_dict(torch.load(config['save_model_path']))

    end = time.time()
    logger.info('Training Completed: Time elapsed: {:.2} Seconds'.format(end - start))
    save_path = config['diagnostics_path']
    plot_results(net, validation_dataset, criterion,
                 save_path=save_path)
    plot_early_stopping(avg_train_loss, avg_valid_loss, save_path=save_path + '_loss')


def main(config, restore_old_checkpoint=False, **kwargs):
    start = time.time()
    train(config, warm_start_NN=None, restore_old_checkpoint=restore_old_checkpoint, **kwargs)

    end = time.time()
    logger.info('Training Completed: Time elapsed: {:.2} Seconds'.format(end - start))


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Launch PENN training')
    parser.add_argument('--config', default='PedSM/PENN/training/default_config.yaml', help='Configuration File Loc')
    parser.add_argument('--load-checkpoint', default=False, help='Start from a saved checkpoint')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--diagnostics', '-d', default=True, help='Enable plotting after training')

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    try:
        torch.manual_seed(42)  # Always set this as we want to reproduce values
        params = map_config_params(config['hyperparameters'])
        main(config=config, restore_old_checkpoint=args.load_checkpoint, verbosity=args.verbose, params=params, diagnostics=args.diagnostics)
    except Exception as exc:
        raise
