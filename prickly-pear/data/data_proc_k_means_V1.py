import torch
import logging
from utils import *

logger = logging.getLogger('Dataloading_V1')  # Logging is fun!

"""
We want to have one main dataset, then sample with k means the dataset 
into different dataloaders 


"""

# TODO: FILTER ONLY H-MODE PLASMA


def split_data_kmeans():
    pass


class DatasetPED1(torch.utils.data.Dataset):
    def __init__(self, data_loc=None, params=None):
        if data_loc is None:
            logger.error('Dataset not created, as no file was supplied. Try to add file')

        df = convert_to_dataframe(data_loc)
        df = clean_data(df)
        self.df = df

        if params['target_variables'] and params['input_variables']:
            target_labels = params['target_variables']
            input_labels = params['input_variables']
        else:
            target_labels = ['Tepedheight(keV)', 'nepedheight1019(m3)']
            input_labels = ['BetaN(MHD)', 'Ip(MA)', 'R(m)', 'B(T)', 'a(m)', 'averagetriangularity', 'plasmavolume(m3)',
                            'q95', 'P_TOTPNBIPohmPICRHPshi(MW)']

        self.targets = df[target_labels]
        self.inputs = df[input_labels]

        # TODO: Add more robust clean data to remove shit parameters
        # that way the len will actually be real

    def __getitem__(self, item, precision='float32'):
        """

        :param item: location of item in dataframe
        :param precision: this is temperorary but maybe we want to look at multiprecision learning
        :return: dictionary of target and inputs
        """
        target = self.targets.iloc[item]
        inputs = self.inputs.iloc[item]

        target = target.to_numpy(precision)
        inputs = inputs.to_numpy(precision)
        return {'input': inputs, 'target': target}

    def __len__(self):
        return len(self.df)
