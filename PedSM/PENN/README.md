# Pedestal Neural Networks 

As of 7.04.2021, the PENN section is built to run for predicting density height, and compares it to the scaling law developed by Lorenzo. 

Currently, there is support only for pytorch, but further improvements would include adding other neural network kits (e.g., keras). 
## Structure of Repository 

```
dataset/
models/
training/
```

### dataset/
Will contain all the data processing needed to train a pytorch built NN. 

Currently, it is empty, as `training/dataloading.py` contains all the necessary tools to port the filtered dataset. For example, if you just wanted to split the dataset into training and test sets for your own training algorithms, you can simply call `PedSM.PENN.training.dataloading.prepare_dataset_pandas` and define three dict paramereters as input, two lists titled `control_params` and `target_params` and a string with the path to the dataset. 
The function will split the dataset into two pandas dataframes into training and validation sets (with a 0.3 split if not specified using additional config `test_size`).
This function is agnostic to training platform, i.e., you can use it for splitting the data for any other training platform. 

Additionally there is a synthisize of the dataset feature, but it only works at the moment with nepedheight.



### models/
The following feed forward network (FFNN) pytorch models available are listed below, and can be found in `models/FFNN_torch`. 
All the networks have hyperparameters that can be configured in the default config file, under the dict `hyperparameters`. 
- PedFFNN 
  - Simple feed forward net with variable hidden layer depth and width. All hidden layers are dense layers, and their widths be specified in the default config via the `hidden_layer_size: [size1, size2, ..., sizen]` 
- PedFFNN_Cross
  - Cross network, LINK? 
- PedDeepCross
  - Two highways, one cross, one PedFFNN, choose how many cross layers with `cross_layers`, and hidden layer sizes is similar to PedFFNN. 

### training/

All necessary tools can be found for training. TBD is implementing it agnostic to training platform. 