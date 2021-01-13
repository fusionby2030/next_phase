import torch
import nni
import math
import numpy as np
import argparse
import logging

# Importing of UTILS
from models.torch_model_1 import FuncApproxNet
from data.data_set import FuncApproxDataset
from models import model_utils



logger = logging.getLogger('func_approx_NNI') # LOGGING IS FUN


def main(params):
    func_name = 'oscillator'
    if not validate_params(params):  # for invalid param combinations, report the worst possible result
        nni.report_final_result(0.0)

    train_ds = FuncApproxDataset(func_name, is_train=True)
    eval_ds = FuncApproxDataset(func_name, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True,
                                                   num_workers=2)
    net = FuncApproxNet(params)
    optimizer = model_utils.map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
    loss_func = model_utils.map_loss_func(params['loss'])

    epochs = 250 if func_name == 'quadratic' else 1000
    last_results = []
    for epoch in range(epochs):
        # log evaluation results every 5 epochs
        if epoch % 5 == 4:
            rms_error = 0
            max_error = 0
            net.eval()
            with torch.no_grad():
                for i in range(len(eval_ds)):
                    sample = eval_ds[i]
                    data, label = sample['x'], sample['y_exp']
                    output = net(data)
                    max_error = max(max_error, abs(output - label))
                    rms_error += (output - label) * (output - label)
            rms_error = math.sqrt(rms_error / eval_ds.size)
            eval_metric = -math.log10(rms_error)
            nni.report_intermediate_result(eval_metric)
            print("epoch ", str(epoch), " | eval metric : ", str(eval_metric), " | max error: ", str(max_error))

        # do training
        net.train()
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()  # zero the gradient buffers
            data, label = batch['x'], batch['y_exp']
            output = net(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

        if epoch >= epochs - 25:
            last_results.append(eval_metric)
    nni.report_final_result(min(last_results))  # use min of last results since results fluctuates a lot sometimes


def generate_default_params():
    '''
    Generate default parameters for mnist network.
    '''
    params = {
        'hidden_size_1': 16,
        'hidden_size_2': 16,
        'hidden_size_3': 16,
        'act_func': 'LeakyReLU',
        'learning_rate': 0.005,
        'optimizer': 'Adam',
        'loss': 'SmoothL1Loss',
        'batch_size': 50
    }
    return params # from experiment jX3RYvtW that LR and Batch Size should be 50 and 0.005


def validate_params(params):
    if params['hidden_size_2'] == 0 and params['hidden_size_3'] != 0:
        return False
    return True


if __name__ == '__main__':
    torch.manual_seed(42)
    try:
        # get parameters form tuner
        updated_params = nni.get_next_parameter()
        logger.debug(updated_params)
        # run a NNI session
        params = generate_default_params()
        params.update(updated_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
