import torch
try:
    # Make sure you know what you are doing with importing! Here, since we are running training.py in the project dir,
    # we need to make sure that model_utils is being pulled as it were in that project dir.
    from model_utils import *
except ImportError:
    from .model_utils import *


class FuncApproxNet (torch.nn.Module):
    def __init__(self, params):
        super(FuncApproxNet, self).__init__()
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        act_func = map_act_func(params['act_func'])
        self.fc1 = self._fc_block(1, self.hidden_size_1, act_func)
        last_layer_size = self.hidden_size_1
        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
            last_layer_size = self.hidden_size_3
        self.out = self._fc_block(last_layer_size, 1, act_func)

    def forward(self, x):
        x = self.fc1(x)
        if self.hidden_size_2:
            x = self.fc2(x)
        if self.hidden_size_3:
            x = self.fc3(x)
        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block