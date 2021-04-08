import torch
from PedSM.PENN.models.torch_utils import map_act_func, generate_default_params


class SimpleNet(torch.nn.Module):

    def __init__(self, config, params=None):
        super(SimpleNet, self).__init__()
        if params is None:
            params = generate_default_params()

        target_size = config['target_size']
        input_size = config['input_size']

        # input size: batch size
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        self.hidden_size_4 = params['hidden_size_4']

        act_func = map_act_func(params['act_func'])

        self.fc1 = self._fc_block(input_size, self.hidden_size_1, act_func)
        self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
        self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
        self.fc4 = self._fc_block(self.hidden_size_3, self.hidden_size_4, act_func)
        last_layer_size = self.hidden_size_4

        self.out = self._fc_block(last_layer_size, target_size, torch.nn.ELU(alpha=params['alpha']))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block


class PedFFNN(torch.nn.Module):
    def __init__(self, config, params):
        super(PedFFNN, self).__init__()
        target_size = config['target_size']
        input_size = config['input_size']
        act_func = map_act_func(params['act_func'])

        self.hidden_layers = []
        last_size = input_size

        for size in params['hidden_layer_sizes']:
            self.hidden_layers.append(self._fc_block(last_size, size, act_func))
            last_size = size
        if params['batch_norm'] == 1:
            self.bn1 = torch.nn.BatchNorm1d(input_size)
        else:
            self.bn1 = None
        self.out = self.out = torch.nn.Linear(last_size, target_size)

    def forward(self, x):
        if self.bn1 is not None:
            x = self.bn1(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block


class PedFFNN_Cross(torch.nn.Module):
    def __init__(self, config, params):
        super(PedFFNN_Cross, self).__init__()
        if params is None:
            params = generate_default_params()

        target_size = config['target_size']
        input_size = config['input_size']

        act_func = map_act_func(params['act_func'])
        self.layers = []

        if params['batch_norm'] == 1:
            self.layers.extend(torch.nn.BatchNorm1d(input_size))
        try:
            for _ in range(params['cross_layers']):
                self.layers.extend(Cross(input_size))
        except KeyError as exc:
            Warning('Number of cross layers not specified, using 4')
            for _ in range(4):
                self.layers.extend(Cross(input_size))

        self.out = torch.nn.Linear(input_size, target_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)

        return x


class SimpleNet2(torch.nn.Module):

    def __init__(self, config, params):
        super(SimpleNet2, self).__init__()
        if params is None:
            params = generate_default_params()

        target_size = config['target_size']
        input_size = config['input_size']

        # input size: batch size
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        self.hidden_size_4 = params['hidden_size_4']

        act_func = map_act_func(params['act_func'])

        self.fc1 = self._fc_block(input_size, self.hidden_size_1, act_func)
        if params['batch_norm'] == 1:
            self.bn1 = torch.nn.BatchNorm1d(input_size)
        else:
            self.bn1 = None
        self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
        self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
        self.fc4 = self._fc_block(self.hidden_size_3, self.hidden_size_4, act_func)

        last_layer_size = self.hidden_size_4

        self.out = torch.nn.Linear(last_layer_size, target_size)



    def forward(self, x):
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block


class Cross(torch.nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features

        self.weights = torch.nn.Parameter(torch.Tensor(input_features))
        # Kaiming/He initialization with a=0
        # nn.init.normal_(self.weights, mean=0, std=math.sqrt(2/input_features))
        torch.nn.init.constant_(self.weights, 1.)

        self.bias = torch.nn.Parameter(torch.Tensor(input_features))
        torch.nn.init.constant_(self.bias, 0.)

    def forward(self, x0, x):
        x0xl = torch.bmm(x0.unsqueeze(-1), x.unsqueeze(-2))
        return torch.tensordot(x0xl, self.weights, [[-1], [0]]) + self.bias + x

    # Define some output to give when layer
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.input_features
        )


class Simple_Cross2(torch.nn.Module):
    def __init__(self, config, params):
        super(Simple_Cross2, self).__init__()
        target_size = config['target_size']
        input_size = config['input_size']
        act_func = map_act_func(params['act_func'])

        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        self.hidden_size_4 = params['hidden_size_4']
        last_layer_size = self.hidden_size_4

        self.fc1 = self._fc_block(input_size, self.hidden_size_1, act_func)
        if params['batch_norm'] == 1:
            self.bn1 = torch.nn.BatchNorm1d(input_size)
        else:
            self.bn1 = None
        self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
        self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
        self.fc4 = self._fc_block(self.hidden_size_3, self.hidden_size_4, act_func)

        # self.out = self._fc_block(last_layer_size, target_size, torch.nn.ELU(alpha=params['alpha']))
        self.out = torch.nn.Linear(last_layer_size, target_size)
        self.c1 = Cross(input_size)

    def forward(self, x):
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.c1(x, x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block


class PedDeepCross(torch.nn.Module):
    def __init__(self, config, params):
        super(PedDeepCross, self).__init__()

        target_size = config['target_size']
        input_size = config['input_size']
        act_func = map_act_func(params['act_func'])

        self.cross_layers = []

        try:
            for _ in range(params['cross_layers']):
                self.cross_layers.append(Cross(input_size))
        except KeyError as exc:
            Warning('Number of cross layers not specified, using 4')
            for _ in range(4):
                self.cross_layers.append(Cross(input_size))

        self.out_cross = torch.nn.Linear(input_size, target_size, act_func)

        self.hidden_layers = []
        last_size = input_size

        for size in params['hidden_layer_sizes']:
            self.hidden_layers.append(self._fc_block(last_size, size, act_func))
            last_size = size

        self.out_dense = torch.nn.Linear(last_size, target_size)

        if params['batch_norm'] == 1:
            self.bn1 = torch.nn.BatchNorm1d(input_size)
        else:
            self.bn1 = None

        self.final = torch.nn.Linear(target_size*2, target_size)

    def forward(self, x):
        if self.bn1 is not None:
            x = self.bn1(x)
        x2 = x
        for cross_layer in self.cross_layers:
            x = cross_layer(x, x)

        x = self.out_cross(x)

        for layer in self.hidden_layers:
            x2 = layer(x2)
        x2 = self.out_dense(x2)

        out = torch.cat((x, x2), 1)
        out = self.final(out)

        return out

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block
