from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def plot_early_stopping(train_loss, valid_loss, save_path):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', lw=4)
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss', lw=2.5)

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path)


def denormalize(values, parameter_list, value_dict):
    j = 0
    denormed = []
    for val in values:
        max_val, min_val = value_dict[parameter_list[j]]
        # print(max_val, min_val)
        # print(values)
        denorm = val * (max_val - min_val) + min_val
        denormed.append(denorm)
        j += 1
    return denormed


def fitted_scale2(shot_df):
    # Urano Scaling
    I_p = shot_df[0] ** (1.38)
    B_t = shot_df[1] ** (-0.42)
    P_NBI = shot_df[2] ** (-0.25)
    delta = shot_df[3] ** (0.71)
    gamma = shot_df[4] ** (0.11)
    if (11.4 * I_p * B_t * P_NBI * delta * gamma) > 14.0:
        return 0.0
    return 11.4 * I_p * B_t * delta * gamma * P_NBI


def fitted_scale(shot_df):
    # Lorenzo Scaling
    I_p = shot_df[0] ** 1.24
    P_tot = shot_df[1] ** (-0.34)
    delta = shot_df[2] ** (0.62)
    gamma = shot_df[3] ** 0.08
    meff = shot_df[4] ** 0.2
    if (9.9 * I_p * P_tot * delta * gamma * meff) > 14.0 or (9.9 * I_p * P_tot * delta * gamma * meff) <= 0.0:
        return None
    return 9.9 * I_p * P_tot * delta * gamma * meff


def fitted_scale_pandas(shot_df):
    I_p = shot_df['Ip(MA)'] ** (1.38)
    B_t = shot_df['B(T)'] ** (-0.42)
    P_NBI = shot_df['P_NBI(MW)'] ** (-0.25)
    delta = shot_df['averagetriangularity'] ** (0.71)
    gamma = shot_df['gasflowrateofmainspecies1022(m3)'] ** (0.11)
    if (11.4 * I_p * B_t * P_NBI * delta * gamma) > 14.0 or 11.4 * I_p * B_t * P_NBI * delta * gamma == 0.0:
        return None

    return 11.4 * I_p * B_t * P_NBI * delta * gamma


def denorm_controls(dataset):
    control_df = dataset.controls
    numerical_cols = []
    control_cols = []
    for col in control_df.columns:
        if col[:2] == 'di' or col[:2] == 'sh':
            continue
        else:
            control_cols.append(col)

    denormed_df = dataset.control_scaler.inverse_transform(control_df[control_cols])
    return denormed_df, control_cols


def plot_against_scaling(net, dataset, criterion, save_path=None, trial_id=None, exp_id=None, return_rmse=False, dataloader=None):
    # TODO: Move dataloader to outside the shit
    # TODO: Convert plotting to agnostic dataset, compare control parameters to outputs, angostic to scaling!
    import torch
    if dataloader is not None:
        loader = dataset
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    for i, batch in enumerate(loader):
        inputs, targets = batch['input'], batch['target']
        output = net(inputs)
        MSE = criterion(output, targets)
        output = output.detach().numpy()
        MSE = torch.sqrt(MSE)

    scaled_inputs = denorm_controls(dataset)
    scaled_targets = dataset.target_scaler.inverse_transform(targets)
    scaled_output = dataset.target_scaler.inverse_transform(output)

    MSE = mean_squared_error(y_pred=scaled_output, y_true=scaled_targets, squared=False)

    scaling_law = []
    for shot in scaled_inputs:
        scaling_law.append(fitted_scale(shot))

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(scaled_targets, scaled_targets, label='Truth')
    plt.scatter(scaled_targets, scaling_law, label='Scaling Law')
    plt.scatter(scaled_targets, scaled_output, label='Predicted')
    plt.legend()

    if exp_id and trial_id:
        plt.title('ANN vs Scaling Law: ID: {} ; RMSE {:.2}'.format(trial_id, MSE))
        fig.savefig(
            '/home/adam/Uni_Sache/Bachelors/Thesis/next_phase/NNI_meditations/density_predictions/density_results/' + exp_id + '/' + trial_id)
    elif save_path:
        # TODO: Not just simple net
        plt.title('{} vs Scaling Law: RMSE {:.3}'.format('SimpleNet', MSE))
        fig.savefig(save_path)
    else:
        plt.show()

    if return_rmse:
        return MSE


def plot_parameters(plot_controls, scaled_output, scaled_targets, scaled_inputs, target_params):
    fig, axs = plt.subplots(nrows=len(plot_controls), ncols=len(target_params), figsize=(15, 15))
    axs = axs.ravel()
    # scaled inputs is a numpy array
    i = 0
    scaled_inputs = scaled_inputs.transpose()
    if len(target_params) == 1:
        for control in scaled_inputs:
            axs[i].scatter(control, scaled_output)
            axs[i].scatter(control, scaled_targets)
            axs[i].set(xlabel=plot_controls[i])
            i += 1
    else:
        j = 0
        p = 0
        scaled_output = scaled_output.transpose()
        scaled_target = scaled_targets.transpose()
        for output in scaled_output:
            for control in scaled_inputs:
                axs[i].scatter(control, output)
                axs[i].scatter(control, scaled_target[j])
                axs[i].set(xlabel=plot_controls[p], ylabel=target_params[j])
                i += 1
            j += 1
            p = 0
    plt.suptitle('Controls VS Targets')
    plt.tight_layout()
    plt.show()


def plot_predictions(scaled_targets, scaled_output, target_params, exp_id=None, trial_id=None, config=None, MSE=0.0, save_path=None):
    # TODO: Add the additional parts
    if len(target_params) == 1:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(scaled_targets, scaled_targets, label='Truth')
        plt.scatter(scaled_targets, scaled_output, label='Predicted')
        plt.legend()
    else:
        scaled_targets = scaled_targets.transpose()
        scaled_ouptut = scaled_output.transpose()
        fig, axs = plt.subplots(nrows=len(target_params), ncols=1, figsize=(15, 15))
        axs = axs.ravel()
        i = 0
        for target in scaled_targets:
            axs[i].scatter(target, target, label='Truth')
            axs[i].scatter(target, scaled_ouptut[i], label='Prediction')
            axs[i].set(title=target_params[i] + ' RMSE: {:.3}'.format(MSE))
            i += 1
    if exp_id and trial_id:
        plt.suptitle('ANN : ID: {} ; RMSE {:.2}'.format(trial_id, MSE))
        fig.savefig(
            '/home/adam/Uni_Sache/Bachelors/Thesis/next_phase/NNI_meditations/density_predictions/density_results/' + exp_id + '/' + trial_id)
    elif save_path:
        # TODO: Not just simple net
        plt.suptitle('{} vs Scaling Law: RMSE {:.3}'.format(config['nn_type'], MSE))
        fig.savefig(save_path)
    else:
        plt.suptitle('Predictions vs Actual with {}, {:.3}'.format(config['nn_type'], MSE))
        plt.show()


def plot_results(net, dataset, criterion, save_path=None, trial_id=None, exp_id=None, return_rmse=False, dataloader=None, config=None):
    # TODO: Move dataloader to outside the shit
    # TODO: Maybe just drop everything in the config, like too many shits there
    import torch
    if dataloader is not None:
        loader = dataset
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    for i, batch in enumerate(loader):
        inputs, targets = batch['input'], batch['target']
        output = net(inputs)
        MSE = criterion(output, targets)
        output = output.detach().numpy()
        MSE = torch.sqrt(MSE)

    # These are all numpy arrays now of shape (num_samples, _)
    # Where _ represents either number of control parameters for inputs, or number of target parameters for targets
    scaled_inputs, plot_controls = denorm_controls(dataset)
    scaled_targets = dataset.target_scaler.inverse_transform(targets)
    scaled_output = dataset.target_scaler.inverse_transform(output)
    target_params = config['target_params']

    MSE = mean_squared_error(y_pred=scaled_output, y_true=scaled_targets, squared=False)
    # TODO: CALCULATE MSE FOR BOTH OUTPUTS

    plot_predictions(scaled_targets, scaled_output, target_params, config=config, MSE=MSE)

    plot_parameters(plot_controls, scaled_output, scaled_targets, scaled_inputs, target_params)

    if return_rmse:
        return MSE