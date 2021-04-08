# Pedestal Surrogate Modeling (PedSM) 

The beginnings of a surrogate modeling tool kit for the pedestal database. More documentation is on the way, but keep it simple no. 

The current dependencies are required: 
- NNI (for search optimization)
- pytorch 
- sklearn
- pandas
- numpy 
- matplotlib (for plotting)

## Overview 

- PENN
    - ANN's
    
See `\PENN` for more details.

For quick development, try out the following after cloning the repository: 

Outside of the PedSM directory, create a python script that has the following: 

`import PedSM`

a dictionary, this will be your config, you can take the default config found in `training/default_config.yaml`, but can create your own: 

```
config = {
    'synth_data': 0,
    'num_synth': 150,
    'epochs': 200,
    'hyperparameters': {
      'hidden_size_1': 84,
      'hidden_size_2': 66,
      'hidden_size_3': 54,
      'hidden_size_4': 41,
      'act_func': 'ELU',
      'learning_rate': 0.015,
      'optimizer': 'Adam',
      'loss': 'MSELoss',
      'batch_size': 401,
      'batch_norm': 1,
      'hidden_layer_sizes': [84, 66, 54, 41]
    },
    'scheduler': 1,
    'scheduler_milestones': [50, 75, 100],
    'nn_type': 'PedFFNN',
    'save_model_path': 'name_you_wil_remember',
    'diagnostics_path': 'graph_outputs',
    'data_loc': '/home/adam/Uni_Sache/Bachelors/Thesis/next_phase/Workflow_Example/data/filtered_daten_comma.csv',
    'target_params': ['nepedheight1019(m3)'],
    'control_params': ['shot', 'Ip(MA)', 'P_TOTPNBIPohmPICRHPshi(MW)', 'averagetriangularity',
                     'gasflowrateofmainspecies1022(es)', 'Meff', 'B(T)', 'divertorconfiguration'],
    'test_set_size': 0.3
}
```

As you can see, there are many parameters you can specify, like `nn_type`, which decides which NN you want, (see PENN README). 
Also important is the `target_params` and `control_params`, specifying which columns from the dataset you want to use. 

Then you can map the config to another dictionary called `params`, which is used by the setting up of the NN. In the future I will disolve this param and just use the config, but it is used for the optimization of the ANN using NNI. 
`params = PedSM.Penn.torch_utils.map_config_params(config)`
Do this if you are using pytorch or built-in training functions

then you could split the dataset to use in your own training function using: 
`train_df, test_df = PedSM.PENN.training.dataloading.normalize_dataframe.prepare_dataset_pandas(config)`

If you just want to jump straight to training something, then use 
`train(config=config, params=params, diagnostics=True)`

Which will run a training loop based on your configurations. When diagnostics is true, then graphs will be shown of the training. 

You could instead just change the default config to your liking, or write your own config file in yaml,  then run the following:
`python3 -m PedSM.PENN.training.training_torch --config /path/to/your/config --diagnostics True`
If no config is specified, then the default config is used. 

#### To train a 