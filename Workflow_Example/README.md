# A workflow example 

This directory is aimed at showing the workflow of training/using a ANN. 

The process begins _after_ the data has been filtered and cleaned, so we assume a clean and ready to use dataset. 
For more information on the filtering process, check out the **README** in `data/`. 

As of 14.03.21, the repository is generally made for training a NN for pedestal pressure density predictions, given engineering parameters.
Hopefully it will expand outwards. 


## General Structure 

A typical NN project has the following structure: 

``` 
	data/
	models/
	configs/
	training.py
	testing.py*
```

It is perhaps useful to go through this structure in the following order:
- `training.py`: where the main training occurs 
- `data/`: where the data processing occurs
- `configs/`: my personal touch, to keep things as generalized as possible
- `models/`: where different NN models are stored/created
- `testing.py*`: This is a work in progress, and is not quite completed, but will generally graph how well the NN predicts compared to ground truth

## training.py 

The general structure of the training file resembles this: 

``` 
	epochs = 400
	for epoch in epochs: 
		# Training 
		for batch in training_loader:
			inputs, targets = batch['input'], batch['target']
			prediction = net(inputs)
			loss = loss_function(prediction, targets)
			back_prop(loss)

		# Validation
		for batch in validation_loader:
			inputs, targets = batch['input'], batch['target']
			prediction = net(inputs)
			loss = loss_function(prediction, targets)
			score = scoring_function(loss)
			report_score(score)
```

So as we see, there are generally two sections (loops), training and validation. 
Can you spot the differences?

They both require different subsets of the full dataset. Validation is thus the measure on how well the NN predicts on data it 'has not seen before'.
You may wonder, but it has seen the data after one epoch?! Yes true, but it has not _learned_ from it like it has in the training loop. This is the other difference, the line `back_prop(loss)`. 
While going through the `training_loader`, the NN is adjusting its weights and losses in order to more accurately predict the data it is recieveing. On the other hand, while looping through the `validation_loader`, there is no back-prop, and the NN simply makes predictions, without adjustments after.

There is no fundamental difference between the training and validation data, only that they consist of different entries. But where is the data loader coming from?

## Data 
 
In order to 'feed' the NN, we need the data in a specific format, which is normally matrices, or depending on which NN framework you use, a tensor. For simplicity, I will explain things in terms of using the framework *pytorch*, since that is what I use. 

So a pytorch ANN model will take tensor as well as numpy like objects as an input. So we know that our variables `inputs, targets` will have to be either numpy arrays or tensor like objects. So the big picture is going to be transforming the data in our main data file to a tensor. A naive approach would be to have a training loop that loops through all of the data in a given dataset, then selects the specifc columns from the data for targets and inputs, then feeds those to the NN: 
``` 
	input_variables = ['I_p(MA)', 'B(T)', 'P_NBI(MW)', 'averagetriangularity', 'gasflowrate(es)']
	target_variables = ['nepedheight(m3)']
	data_set = read_data_from_file('/path/to/data.csv')
	for entry in dataset:
		inputs = entry[input_variables]
		targets = entry[target_variables]
		inputs = torch.to_tensor(inputs)
		targets = torch.to_tensor(targets)
		prediction = net(inputs)
		# then loss and back prop... etc. 
```

But this is going to be a bottleneck, and is not very efficient, since during the training loop, you are doing many computations to grab the right data then convert it into a tensor. 
So how do we overcome this? Well we make our own dataloader, one that is initialized outside of the training loop to prepare the data so that it can be called for in the training loop, just like our first example in the training.py section. 
I will save the details of DIY your own dataloader using numpy, and suggest to simply use the pytorch built in tools `Dataset` and `DataLoader`. 
The idea is that you simply pass your data at the beginning to the *Dataset*, then pass that Dataset to a dataloader. 
That way, you are removing the bottleneck. Look at the code as well as the **README** in `data/` for details. 

## Configs


This is essentially a way to meta your experiments. 
We can set up the dataloaders such that they spit out input and target data corresponding to a list that we have in our config file. 
For example, we could store different lists of inputs and lists of targets, and have a final dictionary that says which to choose as input and as target, then give that information to the dataloader. 
Additionally, if we wanted to load//save a model before/after training, then we can include this in the config.
The load model attribute in the config would have the location and name of the model. If the model is saved in the `models/saved_models` directory, then the string would look like: 
`./models/saved_models/name_of_model`. 

For information on saving or loading a model, I used [this](https://pytorch.org/tutorials/beginner/saving_loading_models.html) as a reference guide. 
An example file is found in `config.yaml`. 

## Models 

Here you have to create a general structure to your NN. The example given in the directory is for a three layer feed forward network.  

# How to use this right now

After cloning the repository and installing the dependencies below: 
- torch
- numpy
- matplotlib
- pandas
- pyyaml

you should be able to train a NN using the following command:
`python3 training.py`, and to change the hidden layer size, go to `models/model_utils.py` and find the function `generate_default_parameters`, and change there accordingly. 
You can save this model by changing the `example_config`, under `experiment{save_model: '/path/to/MODELNAME'}`. 
To test how the NN predicts against the ground truths, run `python3 testing.py`, while configuring the `example_config` to point to the model you want to load. 
Be careful, you may run into errors in which the loaded model size does not fit the NN you want, and in this case, make sure the `generate_default_params` has the same params as the model you saved and want to load.

Testing is TBD expanding, and will include plotting for any type of experiment, but at the moment it is just for the density scaling experiment. 
