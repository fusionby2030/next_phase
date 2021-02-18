import torch

try:
    from model_utils import *
except ImportError:
    from .model_utils import *


class SimpleNet(torch.nn.Module):

    def __init__(self, params):
        super(SimpleNet, self).__init__()

        # input size: batch size
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']

        act_func = map_act_func(params['act_func'])

        # TODO: Add parameters for size of data currently 9 since 9 features
        self.fc1 = self._fc_block(9, self.hidden_size_1, act_func)

        last_layer_size = self.hidden_size_1

        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
            last_layer_size = self.hidden_size_3
        self.out = self._fc_block(last_layer_size, 2, act_func)

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
