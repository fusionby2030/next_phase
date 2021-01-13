import torch
import numpy as np


class FuncApproxDataset(torch.utils.data.Dataset):
    def __init__(self, func_name, is_train):
        self.size = 1000 if is_train else 100
        self.is_train = is_train
        if func_name == 'quadratic':
            self.func = self._quadratic
        elif func_name == 'oscillator':
            self.func = self._oscillator
        self._prep_oscillator()

    def __getitem__(self, index):
        if self.is_train:
            data = torch.rand(1, 1)
        else:
            data = torch.empty(1, 1, dtype=torch.float)
            val = (index % self.size)/float(self.size)
            data.fill_(val)  # make uniform spacing for consistent evaluation
        label = self.func(data)
        return {'x': data, 'y_exp': label}

    def __len__(self):
        return self.size

    def _prep_oscillator(self):
        self.osc_x = np.arange(11) * 0.1
        self.osc_y = np.zeros_like(self.osc_x)
        y = self.osc_x[0]
        for i in range(len(self.osc_x)):
            y = -(y * y - 0.5)
            self.osc_y[i] = y

    def _quadratic(self, x):
        return x*x

    def _oscillator(self, x):
        y = torch.from_numpy(np.interp(x, self.osc_x, self.osc_y))
        y = y.float()
        return y
