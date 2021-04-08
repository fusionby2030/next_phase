import pandas as pd
import numpy as np

import logging
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

root_logger = logging.getLogger('pedsm')
logger = root_logger
logger.setLevel(logging.INFO)


class PEDset(torch.utils.data.Dataset):
    def __init__(self, control_df, target_df, scalers, synth_data=None):
        """
        datasets?
        """

        self.controls = control_df
        self.targets = target_df
        self.control_scaler, self.target_scaler = scalers


    def __len__(self):
        return len(self.controls)

    def __getitem__(self, item):
        # TODO:This can not be the best way
        target = self.targets.iloc[item].to_numpy('float32')
        control = self.controls.iloc[item].to_numpy('float32')

        return {'input': control, 'target': target}

    def plot_histograms(self):
        import matplotlib.pyplot as plt

        self.targets.hist()
        plt.title('Scaled Target Ranges')
        plt.show()
        self.controls.hist()
        plt.title('Scaled Control Ranges')
        plt.show()


def prepare_dataset_torch(config):
    """
    Prepares two pytorch datasets, since we are using pytorch here

    :param config:
    :return:
    """
    (train_datasets, validation_datasets), (train_scalers, validation_scalers) = prepare_dataset_pandas(config)

    control_train, target_train = train_datasets
    control_test, target_test = validation_datasets
    # print(control_train.shape)
    config['input_size'] = control_train.shape[1]
    config['target_size'] = target_train.shape[1]
    # NOTE: HERE WE USE THE TRAIN SCALERS TO SCALE THE VALIDATION SET
    # https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data
    train_torch_dataset = PEDset(control_train, target_train, train_scalers)
    validation_torch_datset = PEDset(control_test, target_test, train_scalers)

    return train_torch_dataset, validation_torch_datset


def synthesize_data(all_df, exp_df, num_synth=150, control_col=None, synth_col='nepedheight1019(m3)', synth_val=10.5):
    # TODO: Make agnostic

    synth_df = pd.DataFrame()
    all_cols = control_col
    all_cols.append(synth_col)
    uncert_cols = []
    for col in all_cols:
        if 'error_' + col in all_df.columns:
            uncert_cols.append('error_' + col)
        else:
            pass
    synth_pool = all_df[all_cols+uncert_cols]

    if 'divertorconfiguration' in synth_pool.columns:
        synth_pool = pd.get_dummies(synth_pool, columns=['divertorconfiguration'])
    if 'shot' in synth_pool.columns:
        synth_pool = pd.get_dummies(synth_pool, columns=['shot'])

    # Find indices of high neped
    synth_pool = synth_pool[synth_pool[synth_col] >= synth_val]

    new_shots = []
    # Sample from them
    for _ in range(num_synth):
        sample = synth_pool.sample(1)
        new_shot = {}
        for col in exp_df.columns:
            if 'error_' + col not in uncert_cols:
                value = sample[col].values[0]
                new_shot[col] = value
            else:
                value = sample[col]
                uncert = sample['error_' + col]
                new_value = np.random.normal(loc=value, scale=uncert, size=1)[0]
                new_shot[col] = new_value
        new_shots.append(new_shot)
    synth_df = pd.DataFrame(new_shots)
    return synth_df



def split_dataset(df, control_columns, target_columns, test_size, synth_data=False):
    control_df, target_df = df[control_columns], df[target_columns]
    logger.info('To Categorical')
    if 'divertorconfiguration' in control_df.columns:
        control_df = pd.get_dummies(control_df, columns=['divertorconfiguration'])
    if 'shot' in control_df.columns:
        control_df = pd.get_dummies(control_df, columns=['shot'])
    # TODO: Logging  categorical

    control_train, control_test, target_train, target_test = train_test_split(control_df, target_df,
                                                                              test_size=test_size, random_state=2)

    return control_train, control_test, target_train, target_test


def normalize_validation(dataframes, train_scalers):
    control_scaler, target_scaler = train_scalers
    control_df, target_df = dataframes
    scaled_control, scaled_target = control_df.copy(), target_df.copy()

    control_cols = []
    for col in control_df.columns:
        if col[:2] == 'di' or col[:2] == 'sh':
            continue
        else:
            control_cols.append(col)
    target_cols = []
    for col in target_df.columns:
        if col[:2] == 'di' or col[:2] == 'sh':
            continue
        else:
            target_cols.append(col)

    scaled_c = control_scaler.transform(control_df[control_cols])
    scaled_control[control_cols] = scaled_c

    scaled_t = target_scaler.transform(target_df[target_cols])
    scaled_target[target_cols] = scaled_t

    return (scaled_control, scaled_target), (control_scaler, target_scaler)


def normalize_dataframe(dataframes):
    control_df, target_df = dataframes
    scaled_control, scaled_target = control_df.copy(), target_df.copy()

    control_cols = []
    # This is as to not normalize the divertor column, as well as the shot column, as they are categorical
    # TODO: make a list of columns that are to be categorical, global call, and use that instead of this debauchery
    for col in control_df.columns:
        if col[:2] == 'di' or col[:2] == 'sh':
            continue
        else:
            control_cols.append(col)
    target_cols = []
    for col in target_df.columns:
        if col[:2] == 'di' or col[:2] == 'sh':
            continue
        else:
            target_cols.append(col)

    control_scale = StandardScaler()
    control_scale.fit(control_df[control_cols])
    scaled_c = control_scale.transform(control_df[control_cols])
    scaled_control[control_cols] = scaled_c

    target_scale = StandardScaler()
    target_scale.fit(target_df[target_cols])
    scaled_t = target_scale.transform(target_df[target_cols])
    scaled_target[target_cols] = scaled_t

    return (scaled_control, scaled_target), (control_scale, target_scale)


def prepare_dataset_pandas(config):
    """
    Given config file, return two pandas dataframes, training and validation split,
    that are synthed (if config['synth_data'd]), normed and categoricallized as well as their scaling from sklearn.
    :param config:
    :return: dataframes (train_dataframe, test_dataframe), scalers(train_scaler, validation_scaler)
    """
    try:
        data_path = config['data_loc']
        control_columns = config['control_params']
        target_columns = config['target_params']
        test_size = config['test_set_size'] if config['test_set_size'] is not None else 0.3
    except KeyError as exc:
        raise KeyError('Config dict does not specify data_loc, control_params and or target_params')
    full_df = pd.read_csv(data_path)
    logger.info('Spliting data into train and test')

    control_train, control_test, target_train, target_test = split_dataset(full_df, control_columns, target_columns,
                                                                           test_size)

    control_cols = control_train.columns
    target_cols = target_train.columns

    # print(control_train)
    try:
        # TODO: Logging
        num_synth = config['num_synth']

        # TODO: IT DOES WORKS JUST FOR NOW
        synthed_df = synthesize_data(all_df=full_df, exp_df=control_train.join(target_train), num_synth=num_synth, control_col=control_columns)
        control_train = control_train.append(synthed_df[control_cols], ignore_index=True)
        target_train = target_train.append(synthed_df[target_cols], ignore_index=True)
    except KeyError as exc:
        Warning('No synth data specified, moving forward without synthesis')
        pass
    train_dataframes = (control_train, target_train)
    validation_dataframes = (control_test, target_test)

    logger.info('Normalizing Data')

    train_dataframes, train_scalers = normalize_dataframe(train_dataframes)
    # TODO: We really do not need to create new scalers for the validation set!
    validation_dataframes, validation_scalers = normalize_validation(validation_dataframes, train_scalers)

    return (train_dataframes, validation_dataframes), (train_scalers, validation_scalers)
