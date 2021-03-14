import torch
import pandas as pd
import numpy as np


def get_DC_distrubtion(loader):
    """
    If you want to check how distributed the train and validation sets are, then use this.
    :param loader: train or validation loader
    :return: a dictionary with the counts for the corresponding divertor configurations
    """
    count_dict = {str(k): 0 for k in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}

    for i, batch in enumerate(loader):
        for shot in batch['input']:
            dc = shot[10] * 5 + 1
            count_dict[str(dc.numpy())] += 1
    return count_dict


def plot_DC_distribution(count_dict):
    """
    used in coordination with the above counting method,  and will return a plot of the distribution
    :param count_dict:
    :return: None, but feel free to save the fig
    """
    import matplotlib.pyplot as plt
    keys = count_dict.keys()
    values = count_dict.values()
    plt.bar(keys, values)
    # plt.savefig('')
    plt.show()


def split_dataset(torch_dataset, batch_size, split=0.2):
    """
    Split the dataset into two dataloader
    :param torch_dataset:
    :param split: (float) percentage of dataset to split into training and validation
    :param batch_size:
    :return: train_loader, validation_loader
    """
    indices = list(range(len(torch_dataset)))  # make a list of all the indices
    data_split = int(np.floor(split * len(torch_dataset)))
    train_indices, validation_indices = indices[data_split:], indices[:data_split]

    length_dataset = len(torch_dataset)
    train_size = int(length_dataset * 0.7)
    validation_size = length_dataset - train_size

    train_ds, valid_ds = torch.utils.data.random_split(torch_dataset, (train_size, validation_size))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader


def normalize_data(dataframe):
    """
    :param dataframe: (pandas dataframe)
    :return: normalized dataframe (pandas), and dictionary with each column that was normalized
    """
    result = dataframe.copy()
    cached_norm_values = {}
    for feature_name in dataframe.columns:
        max_value = dataframe[feature_name].max()
        min_value = dataframe[feature_name].min()
        result[feature_name] = (dataframe[feature_name] - min_value) / (max_value - min_value)
        cached_norm_values[feature_name] = (max_value, min_value)
    return result, cached_norm_values


class PedestalDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        # First Read the Data
        df = pd.read_csv(config['file_loc'])

        # Find the columns you need
        input_name = config['experiment']['input']
        target_name = config['experiment']['target']

        # List of columns

        self.input_params = config['input_params'][input_name]
        self.target_params = config['target_params'][target_name]

        # Select the data that is what we need
        data_filtered = df
        self.data = data_filtered

        inputs = data_filtered[config['input_params'][input_name]]
        targets = data_filtered[config['target_params'][target_name]]

        self.inputs, self.inputs_norms = normalize_data(inputs)
        self.targets, self.targets_norms = normalize_data(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        target = self.targets.iloc[item]
        control = self.inputs.iloc[item]

        target = target.to_numpy('float32')
        control = control.to_numpy('float32')
        return {'input': control, 'target': target}
