import torch
import logging
import re
import pandas as pd
import numpy as np


logger = logging.getLogger('Dataloading_V1')  # Logging is fun!


def convert_to_dataframe(data_loc):
    """

    :param data_loc: string
    :return: (pd dataframe)  returns the data as a dataframe.
    """
    shots = []
    with open(data_loc, 'r') as infile:
        headers = infile.readline()
        for line in infile:
            shot_n = line.strip()
            shot_n = shot_n.split(',')
            shot_n = [re.sub(r'[^a-zA-Z0-9_().]', '', word) for word in shot_n]
            shots.append(shot_n)
    headers = headers.split(',')
    headers_fin = [re.sub(r'[^a-zA-Z0-9_()]', '', word) for word in headers]
    df = pd.DataFrame(shots, columns=headers_fin, dtype='float32')
    logger.info('Converted data from {} to a pandas dataframe, moving on to cleaning'.format(data_loc))
    return df


def clean_data(dataframe):
    """
    The goal of this function is to drop all NA values
    Maybe we want to also get rid of some stuff but for the moment just NA values
    :param dataframe: (pandas dataframe) to be cleaned
    :return: cleaned dataframe
    """
    dataframe_cleaned = dataframe.dropna()
    logger.info('Cleaned data, i.e., removed all na rows, moving on to normailizing')
    return dataframe_cleaned


def normalize_dataframe(df):
    """
    Goal is to normailize the data, maybe we should go ahead now and make a un normalize?
    Will have to cache or use a built in function like sklearn or something??
    maybe replace with sklearn function?

    :param df:
    :return:
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result


def split_dataset(torch_dataset, batch_size):
    """
    Splits data into training and validation sets

    :param torch_dataset:
    :param batch_size: (int) you know how it be
    :return: dataloaders! Maybe not best idea but hey lets go with it for the moment
    """
    validation_split = 0.2
    dataset_size = len(torch_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    logger.info("Split dataset into training and validation")
    return train_loader, validation_loader


"""
What needs to be done:
 
 
4.) Figure out where to put logger statements (where in file like function vs calling of function)  

This needs to be done in some combi function from a package utils 

"""


class DatasetPED(torch.utils.data.Dataset):
    def __init__(self, data_loc=None, params=None):
        """
        :param data_loc: (string) Path to the data file
        """
        # TODO:  ADD TRY Catch

        if data_loc is None:
            logging.error('Dataset not created, as no file was supplied. Try to add file')
        df = convert_to_dataframe(data_loc)
        df = clean_data(df)
        self.data = df


        target_labels = ['Tepedheight(keV)', 'nepedheight1019(m3)']
        input_labels = ['BetaN(MHD)', 'Ip(MA)', 'R(m)', 'B(T)', 'a(m)', 'averagetriangularity', 'plasmavolume(m3)',
                        'q95', 'P_TOTPNBIPohmPICRHPshi(MW)']

        self.targets = self.data[target_labels]
        self.inputs = self.data[input_labels]

        # TODO: Un normalize data!

        self.targets = normalize_dataframe(self.targets)
        logger.info('Normailized TArgets.')
        self.inputs = normalize_dataframe(self.inputs)
        logger.info('Normailized Inputs.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        target = self.targets.iloc[item]
        input = self.inputs.iloc[item]

        target = target.to_numpy('float32')
        input = input.to_numpy('float32')
        return {'input':input, 'target':target}


if __name__ == '__main__':
    torch.manual_seed(42)
    # TODO: Add config file and decide what goes into config file
    try:
        file_loc = 'daten-comma.txt'
        data_ped = DatasetPED(file_loc)
        train_loader, validation_loader = split_dataset(data_ped)

    except Exception as exception:
        logger.exception(exception)
        raise
