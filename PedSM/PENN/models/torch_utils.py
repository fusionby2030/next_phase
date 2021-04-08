import sys
import torch
import numpy as np


def generate_default_params():
    """
    Generate default parameters for network.
    """
    params = {
        'hidden_size_1': 40,
        'hidden_size_2': 40,
        'hidden_size_3': 40,
        'hidden_size_4': 40,
        'act_func': 'ELU',
        'learning_rate': 0.015,
        'optimizer': 'Adam',
        'loss': 'MSELoss',
        'batch_size': 360,
        'batch_norm': 1,
        'hidden_layer_sizes': [40, 40, 40, 40, 40, 40],
        'num_synth': 150,

    }
    return params
    # lr = 0.00252


def map_config_params(hyperparameters):
    params = generate_default_params()
    for key in params:
        try:
            params[key] = hyperparameters[key]
        except KeyError as exc:
            print('Param {} not specialized, using default parameter value {}'.format(key, params[key]))
            continue
    return params


def map_model(config, params):
    from PedSM.PENN.models.FFNN_torch import SimpleNet2, Simple_Cross2, PedFFNN, PedDeepCross, PedFFNN_Cross

    if config.get('nn_type') == 'SimpleNet':
        net = SimpleNet2(config, params)
    elif config.get('nn_type') == 'PedDeepCross':
        net = PedDeepCross(config, params)
    elif config.get('nn_type') == 'SimpleCross':
        net = Simple_Cross2(config, params)
    elif config.get('nn_type') == 'PedFFNN':
        net = PedFFNN(config, params)
    elif config.get('nn_type') == 'PedFFNN_Cross':
        net = PedFFNN_Cross
    else:
        raise KeyError('NN Type does not yet exist... please choose from SimpleNet, ComplexCross, SimpleCross, or FFNN')

    return net


def map_act_func(af_name):
    if af_name == "ReLU":
        act_func = torch.nn.ReLU()
    elif af_name == "LeakyReLU":
        act_func = torch.nn.LeakyReLU()
    elif af_name == "ELU":
        act_func = torch.nn.ELU()
    elif af_name == "Sigmoid":
        act_func = torch.nn.Sigmoid()
    elif af_name == "Tanh":
        act_func = torch.nn.Tanh()
    elif af_name == "Softplus":
        act_func = torch.nn.Softplus()
    else:
        sys.exit("Invalid activation function")
    return act_func


def map_optimizer(opt_name, net_params, lr):
    if opt_name == "SGD":
        opt = torch.optim.SGD(net_params, lr=lr)
    elif opt_name == "Adam":
        opt = torch.optim.Adam(net_params, lr=lr)
    elif opt_name == "RMSprop":
        opt = torch.optim.RMSprop(net_params, lr=lr)
    else:
        sys.exit("Invalid optimizer")
    return opt


def map_loss_func(loss_name):
    if loss_name == "MSELoss":
        loss_func = torch.nn.MSELoss()
    elif loss_name == "SmoothL1Loss":
        loss_func = torch.nn.SmoothL1Loss()
    elif loss_name == 'L1Loss':
        loss_func = torch.nn.L1Loss()
    elif loss_name == 'KLDiv':
        loss_func = torch.nn.KLDivLoss()
    else:
        sys.exit("Invalid loss function")
    """
    """
    return loss_func


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=25, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.RMSE = 0.0

    def __call__(self, val_loss, model, max_error):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, max_error)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, max_error)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, RMSE):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.RMSE = RMSE
