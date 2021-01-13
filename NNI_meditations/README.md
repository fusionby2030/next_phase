# Automation of NN Development, or how I stopped worrying about the META

Adam if you are too lazy to google `NNI readthedocs`, then here is the [link](https://nni.readthedocs.io/en/stable/). 
Also there is already a quick start in the read the docs - [here](https://nni.readthedocs.io/en/stable/Tutorial/QuickStart.html)

For a more indepth summary, check out the README in the function-fit dir. 

This README is just the quick start, for those just wanting to implement NNI. 

### Outlining Project Structure

Assuming the following project structure ,

```
ml_project/
 models/
 data/
 training.py
```
one must introduce a new directory for the config and search space files used by NNI. They do no thave to be in their own dir but hey, readability is cool. 
The new structure will look like this: 
```
ml_project/
 models/
   torch_model_1.py 
 data/
   data.dat
   utils.py
 configs/
   search_space_good_file_name.json
   config.yml
 training.py
```

### Updating script to include NNI 

An NNI Expirement consists of the training of many neural networks with parameters that are searched across a user defined space. 
Thus, your training loop will be called many times throughout an expirement. 
In the `training.py` file, you should have a `main` function that looks something like this: 

```
def main(params):
    # single training loop given a set of hyperparameters
    validate_params(params) # basically in the case of arch. search, you can not have the second layer being 0 while the third layer is nonzero. see funciton in example_project 
    training_ds, validation_ds = load_data() # can also include params here if your param file incldues dataset dir location 
    model = Model_Imported_From_models(params)
    optimizer = params['optimizer']
    loss_func = params['loss_func'] 
    # the above will vary depending on which ml package you use, in this case torch 
    # also you will need some kind of mapping function, see models/model_utils.py 
    epochs = params['EPOHCS'] 
    for _ in range(epochs):
        # log evaluation every 5 epochs 
        if epoch % 5 == 4: 
            # make some predictions on validation set
            predictions = model(validation_ds)
            loss = calculate_error(validation_ds, predictions)
            your_custom_score = calculate_score(loss) 
            # Make a custom score, i.e., if you want to just minimize loss, then take the negativelog of the error (NNI looks to maximize scores) 
            # you can for example make a custom score that punishes networks that are large, or that train slowly 
            nni.report_intermediate_result(your_custom_score) 
            # ++++++++++++
            
        model.train() # Standard Training Loop! 
        if _ >= epochs - 25: 
            last_results.append(your_custom_score) 
            
    nni.report_final_result(min(last_results)) # Use the minimum of the last results, since they fluctuate a lot     
```

Okay so that looks like a pretty standard training loop, but we added two function from NNI. 
- `report_intermediate_result(your_custom_score)`: basically send NNI your modified loss 
- `report_final_result`: send NNI your last recorded modified loss 

Thats it for the training loop, now we move on to the actual file main: 

``` 
if __name__ == '__main__':
    try: 
        updated_params = nni.get_next_parameter() 
        params = generate_default_params() 
        params.update(updated_params) 
        main(params)
    except Exception as exception:
        log(exception) 
        raise # yeah come on do some logging you know you'll like it in the end 
```

So here we first get the next params via `get_next_parameter`. 
Then we pass the params to a generated default dictionary using the built-in dictionary method `update`, 
Finally we run the `main(params)` training loop found above. 
That is it. Now lets make some search spaces. 

### Config Files 

There are 2 main configuration files, each consisting of different formats. 
The first is the main config of nni, well named by me as `config.yml` It is a yml file type, and looks roughly like this: 

``` 
authorName: your_name
experimentName: cool_exp_name_420_69
trialConcurrency: 2
maxExecDuration: 5h
maxTrialNum: 5000
trainingServicePlatform: local
searchSpacePath: search_space_arch.json
useAnnotation: false
tuner:
        builtinTunerName: Random
trial:
        command: python3 training.py
        codeDir: ..
        gpuNum: 0

```

Lots going on, but important stuff: 

- searchSpacePath: 
    - since we have a config dir, the path to the search space is in the same folder! 
    - more on search spaces below 
- tuner: aka how to search the space
    - check out all the different tuners available [here](https://nni.readthedocs.io/en/stable/builtin_tuner.html)
    - you can write your own but that sounds laborious
- trial: 
    - command: python3 training_file.py
    - codeDir: since we our configs are contained inside a directory within the main project, we must bounce out using `..`
    - gpuNum: you can guess what this means 

if you want to add early stopping, just include the following into the `config.yml` file: [overview](https://nni.readthedocs.io/en/stable/Assessor/BuiltinAssessor.html)

```
# config.yml
assessor:
    builtinAssessorName: Medianstop
    classArgs:
      optimize_mode: maximize
      start_step: 5
```

Now onto the search space, which is in the config folder. You can split up your experiments as you want. 
I typically try first to search the best batch size and learning rate, then make a new search space for other hyperparameters, and finally architecture. 
You can then split up the various search spaces in various .json files, and just change the `config.yml` file appropriately. 
The search space file looks like this: 

``` 
{
    "act_func": {"_type": "choice", "_value": ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softplus"]},
    "optimizer": {"_type": "choice", "_value": ["SGD", "Adam", "RMSprop"]},
    "loss": {"_type":  "choice", "_value":  ["SmoothL1Loss", "MSELoss"]},
    "learning_rate": {"_type": "quniform", "_value": [0.005,  0.025, 0.005]},
	"batch_size":  {"_type": "choice", "_value":[25, 40, 50, 60, 75, 200]}
}
```

A basic template: choose hypereparamter, then define what type of choosing is being done. Above we use 
- choice: chooses one of the values supplied in the list next to value 
- quniform: chooses from a quniform distribution defined by the list next to value

More on types [here](https://nni.readthedocs.io/en/stable/Tutorial/SearchSpaceSpec.html?highlight=search%20space). 

### Running experiment 


To run the given experiment, one simply calls: 

`nnictl create --config route/to/config.yml`


You will get some nice visuals which I shall update later. 


### Troubleshooting

If you run into problems about a port being not available, try `lsof -i :PORT#` to find out which (on linux) and kill it. 
On my laptop, port 5000 is normally free so: 
`nnictl create --config example_project/configs/config.yml --port 5000`


If you find that the shit keeps failing (the experiments fail but nni starts), then here is why you should use logging! 
After starting a new experiments, go to details, then click on one of the failed exp. Two options appear, choose ERROR LOG. 

There should be a json output of the error that occurred in which file. Normally it is a silly import error but if you are able to run everything normally without NNI, then it should work with NNI. 

Best of luck! 