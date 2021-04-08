import time
import math
import numpy as np
import torch
from sklearn.model_selection import KFold
import logging

root_logger = logging.getLogger('pedsm')
logger = root_logger
logger.setLevel(logging.INFO)
fh = logging.FileHandler('pedsm.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

try:
    from PedSM.PENN.training.data_loading import prepare_dataset_torch
    from PedSM.PENN.models.FFNN_torch import SimpleNet, SimpleNet2, PedDeepCross, Simple_Cross2
    from PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, map_config_params, \
        EarlyStopping, map_model
    from PedSM.PENN.training.diagnostics import plot_results, plot_early_stopping
    from PedSM.PENN.training.training_torch import train_epoch
except Exception as exc:
    from ....PedSM.PENN.training.data_loading import prepare_dataset_torch
    from ....PedSM.PENN.models.FFNN_torch import SimpleNet, SimpleNet2, PedDeepCross
    from ....PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, \
        map_config_params, EarlyStopping
    from ....PedSM.PENN.training.diagnostics import plot_results


def main2(config, params=None, verbosity=0):
    if verbosity == 0:
        logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    start = time.time()
    params = map_config_params(config['hyperparameters'])
    logger.info('Preparing Datasets')

    # Split dataset into K-Fold
    # Train on each fold
    # make predictions on test set
    # calculate RMSE and store in list
    # calc std. dev in the RMSE for predictions

    kfold = KFold(n_splits=5, shuffle=True)

    train_dataset, validation_dataset = prepare_dataset_torch(config)
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):

        print(f'Fold {fold}')
        print('-----------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64, sampler=test_subsampler)

        logger.info('Initializing Torch Network')

        net = map_model(config, params)

        logger.info('Optimizer Initialize')
        optimizer = map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
        loss_func = map_loss_func(params['loss'])

        logger.info('Start Training!')
        if config['scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['scheduler_milestones'],
                                                             gamma=0.1)

        early_stopping = EarlyStopping(verbose=True, trace_func=logger.info, patience=35)

        epochs = config['epochs']
        criterion = torch.nn.MSELoss()

        # Track the losses to determine early stopping

        avg_train_loss = []
        avg_valid_loss = []

        for epoch in range(epochs):

            train_loss, validation_loss, RMSE = train_epoch(net, optimizer, loss_func, train_loader=train_loader,
                                                            test_loader=test_loader, scheduler=scheduler,
                                                            criterion=criterion)

            early_stopping(validation_loss, net, RMSE)

            logger.info(
                'Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Max Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                                validation_loss,
                                                                                                RMSE))

            print('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Max Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                                validation_loss,
                                                                                                RMSE))

            avg_train_loss.append(train_loss)
            avg_valid_loss.append(validation_loss)

            if early_stopping.early_stop:
                logger.info('Early Stopping')
                break

        save_path = '/home/adam/Uni_Sache/Bachelors/Thesis/next_phase/NNI_meditations/density_predictions/final_cuts/ComplexCross/' + str(fold) + '_crossvalid'
        plot_results(net, dataset=validation_dataset, criterion=criterion, save_path=save_path, config=config)
        plot_early_stopping(avg_train_loss, avg_valid_loss, save_path=save_path + str(fold)  + '_loss')
        results[fold] = early_stopping.RMSE

    print(f'K_FOLD CROSS VAL Ssores for 5 Folds')
    print('-------------------------')
    sum = 0.0
    values = []
    for key, value in results.items():
        print(f'Fold {key}: {value} ')
        sum += value
        values.append(value.detach().numpy())
    print(f'Average: {sum / len(results.items())} ')
    print(f'Std Dev: {np.std(values)}')


def main(config, params=None, verbosity=0):
    if verbosity == 0:
        logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    start = time.time()
    params = map_config_params(config['hyperparameters'])
    logger.info('Preparing Datasets')

    # Split dataset into K-Fold
    # Train on each fold
    # make predictions on test set
    # calculate RMSE and store in list
    # calc std. dev in the RMSE for predictions

    kfold = KFold(n_splits=5, shuffle=True)

    train_dataset, validation_dataset = prepare_dataset_torch(config)
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):

        print(f'Fold {fold}')
        print('-----------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64, sampler=test_subsampler)

        logger.info('Initializing Torch Network')

        if config.get('nn_type') == 'SimpleNet':
            net = SimpleNet2(config, params)
        elif config.get('nn_type') == 'ComplexCross':
            net = PedDeepCross(config, params)
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
        early_stopping = EarlyStopping(verbose=True, trace_func=logger.info)

        epochs = config['epochs']
        criterion = torch.nn.MSELoss()

        last_results = []

        # Track the losses to determine early stopping
        train_losses = []
        validation_losses = []

        avg_train_loss = []
        avg_valid_loss = []

        for epoch in range(epochs):

            # Training

            net.train()

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, targets = batch['input'], batch['target']
                output = net(inputs)
                loss = loss_func(output, targets)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            if config['scheduler']:
                scheduler.step()

            # Validation
            net.eval()
            max_error = 0.0

            for i, batch in enumerate(test_loader):
                inputs, targets = batch['input'], batch['target']
                output = net(inputs)

                MSE = criterion(output, targets)
                MSE = torch.sqrt(MSE)
                max_error = max(MSE, max_error)

                loss = loss_func(output, targets)
                validation_losses.append(loss.item())

            train_loss = np.average(train_losses)
            validation_loss = np.average(validation_losses)

            avg_train_loss.append(train_loss)
            avg_valid_loss.append(validation_loss)

            train_losses = validation_losses = []

            early_stopping(validation_loss, net, max_error)

            logger.info(
                'Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Max Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                                validation_loss,
                                                                                                max_error))

            print('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Max Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                                validation_loss,
                                                                                                max_error))
            if early_stopping.early_stop:
                logger.info('Early Stopping')
                break

            last_results.append(max_error)

        save_path = '/home/adam/Uni_Sache/Bachelors/Thesis/next_phase/NNI_meditations/density_predictions/final_cuts/SimpleNet/' + str(fold) + '_crossvalid'
        plot_results(net=net, dataset=validation_dataset, criterion=criterion, save_path=save_path, config=config)
        results[fold] = early_stopping.RMSE

    print(f'K_FOLD CROSS VAL Ssores for 5 Folds')
    print('-------------------------')
    sum = 0.0
    values = []
    for key, value in results.items():
        print(f'Fold {key}: {value} ')
        sum += value
        values.append(value.detach().numpy())
    print(f'Average: {sum / len(results.items())} ')
    print(f'Std Dev: {np.std(values)}')


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
        main2(config=config, params=params)
    except Exception as exc:
        raise
