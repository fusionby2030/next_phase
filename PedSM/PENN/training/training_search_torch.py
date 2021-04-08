import time
import math

try:
    from PedSM.PENN.training.data_loading import prepare_dataset_torch
    from PedSM.PENN.models.FFNN_torch import SimpleNet, SimpleNet2, PedDeepCross, Simple_Cross2
    from PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params, map_model, \
        EarlyStopping
    from PedSM.PENN.training.diagnostics import plot_results, plot_against_scaling
    from PedSM.PENN.training.training_torch import train_epoch, train
except Exception as exc:
    from ....PedSM.PENN.training.data_loading import prepare_dataset_torch
    from ....PedSM.PENN.models.FFNN_torch import SimpleNet, SimpleNet2, PedDeepCross
    from ....PedSM.PENN.models.torch_utils import map_optimizer, map_loss_func, generate_default_params
    from ....PedSM.PENN.training.diagnostics import plot_results
import torch

import nni
import logging

root_logger = logging.getLogger('pedsm_search')
logger = root_logger
logger.setLevel(logging.INFO)


def train_search(config, params=None, warm_start_NN=None, restore_old_checkpoint=False, workers=1, verbosity=0):
    """
    train_search is practically the same as the train function from training_torch, just made for NNI experiments

    :param config:
    :param params:
    :param warm_start_NN:
    :param restore_old_checkpoint:
    :param workers:
    :param verbosity:
    :return:
    """
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

    net = map_model(config, params)

    logger.info('Optimizer Initialize')
    optimizer = map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
    loss_func = map_loss_func(params['loss'])
    criterion = torch.nn.MSELoss()

    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['scheduler_milestones'],
                                                         gamma=0.1)
    else:
        scheduler = None

    epochs = config['epochs']

    # Track the losses to determine early stopping
    avg_train_loss = []
    avg_valid_loss = []

    # initalize the early_stopping object
    early_stopping = EarlyStopping(verbose=True, trace_func=logger.info)

    logger.info('Start Training!')
    for epoch in range(epochs):

        train_loss, validation_loss, RMSE = train_epoch(net, optimizer, loss_func, train_loader=train_loader,
                                                        test_loader=test_loader, scheduler=scheduler,
                                                        criterion=criterion)

        nni.report_intermediate_result(-math.log10(RMSE))
        if early_stopping is not None:
            early_stopping(validation_loss, net, RMSE)
            RMSE = early_stopping.RMSE

        avg_train_loss.append(train_loss)
        avg_valid_loss.append(validation_loss)

        logger.info(
            'Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Best Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                                 validation_loss,
                                                                                                 RMSE))
        print('Epoch {}; Train Loss: {:.5}; Valid Loss: {:.5}; Validation RMSE: {:.5}'.format(epoch, train_loss,
                                                                                              validation_loss, RMSE))
        if early_stopping.early_stop:
            logger.info('Early Stopping')
            RMSE = early_stopping.RMSE
            break

    nni.report_final_result(-math.log10(RMSE))
    end = time.time()
    logger.info('Training Completed: Time elapsed: {:.2} Seconds'.format(end - start))
    plot_against_scaling(net, validation_dataset, criterion, trial_id=str(nni.get_trial_id()),
                 exp_id=str(nni.get_experiment_id()))


def main(config, restore_old_checkpoint=False, **kwargs):
    train_search(config, warm_start_NN=None, restore_old_checkpoint=restore_old_checkpoint, **kwargs)


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Launch PENN training')
    parser.add_argument('--config', default='PedSM/PENN/training/default_config.yaml', help='Configuration File Location')
    # TODO: these parsers below should do something but at the moment they don't
    parser.add_argument('--load-checkpoint', default=False, help='Start from a saved checkpoint')

    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    try:
        torch.manual_seed(42)  # Always set this as we want to reproduce values
        updated_params = nni.get_next_parameter()
        params = generate_default_params()
        params.update(updated_params)
        main(config=config, restore_old_checkpoint=args.load_checkpoint, verbosity=args.verbose, params=params)
    except Exception as exc:
        logging.debug('Failed to Complete Search')
        logging.debug(exc)
        raise
