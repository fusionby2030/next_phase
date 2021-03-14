import torch

try:
    from model_utils import *
except ImportError:
    from .model_utils import *


class SimpleNet(torch.nn.Module):
    def __init__(self, params, config):
        super().__init__()
        input_name = config['experiment']['input']
        target_name = config['experiment']['target']
        input_size = len(config['input_params'][input_name])
        target_size = len(config['target_params'][target_name])

        # input size: batch size
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']

        act_func = map_act_func(params['act_func'])

        self.fc1 = self._fc_block(input_size, self.hidden_size_1, act_func)

        last_layer_size = self.hidden_size_1

        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
            last_layer_size = self.hidden_size_3
        self.out = self._fc_block(last_layer_size, target_size, act_func)

    def forward(self, inputs):
        forward_pass = self.fc1(inputs)
        if self.hidden_size_2 is not 0:
            forward_pass = self.fc2(forward_pass)
        if self.hidden_size_3 is not 0 and self.hidden_size_2 is not 0:
            forward_pass = self.fc3(forward_pass)

        result = self.out(forward_pass)
        return result

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func)
        return block
