from data.data_proc_v2 import *
from models import model_utils
from models.first_model import SimpleNet
import math # maybe figure out how to not need it
import matplotlib.pyplot as plt
# TODO: LEARN HOW IMPORTS WORKS

# TODO: LEARN HOW LOGGING WORKS ACROSS MUTILPLE FILES
logging.basicConfig(filename='data_proc.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('Dataloading_V1')  # Logging is fun!


def main(params):
    file_loc = 'data/daten-comma.txt'
    data_ped = DatasetPED(file_loc, params)
    train_loader, validation_loader = split_dataset(data_ped, params['batch_size'])

    if params['load_model'][0]:
        net = SimpleNet(params)
        optimizer = model_utils.map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
        checkpoint = torch.load(params['load_model'][1])
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    net = SimpleNet(params)  # define neaural network with params
    optimizer = model_utils.map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
    loss_func = model_utils.map_loss_func(params['loss'])

    epochs = 5

    last_results = []
    batch_size = int(params['batch_size'])
    losses = []
    for epoch in range(epochs):

        # TODO: ADD LOGGING
        # TODO: ADD TESTING
        if epoch % 5 == 4:
            rms_error = 0
            max_error = 0
            net.eval()
            with torch.no_grad():
                for i, batch in enumerate(validation_loader):
                    data, targets = batch['input'], batch['target']
                    output = net(data)
                    # print(abs(output-targets))
                    # TODO: Either figoure out a new loss or loop through individually the batch
                    # or you could always sum the shit up
                    # i.e., targets is a 2d tensor, sum up row by row pick max?
                    max_error = max(max_error, torch.max(abs(output - targets)))
                    rms_error += torch.sum((output - targets) * (output - targets), 0)
                rms_error_1 = math.sqrt(rms_error[0] / (batch_size*len(validation_loader))) # this is wrong with batch size
                rms_error_2 = math.sqrt(rms_error[1] / (batch_size*len(validation_loader)))
                print(rms_error_2, rms_error_1)
                eval_metric_1 = -math.log10(rms_error_1)
                eval_metric_2 = -math.log10(rms_error_2)
                # nni.report_intermediate_result(eval_metric)
                print("epoch ", str(epoch), " | eval metric : ", str(eval_metric_1), " | max error: ", str(max_error))
                print('\n# Targets')
                print(targets)
                print("\n# Outputs")
                print(output)
        net.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data, targets = batch['input'], batch['target']
            output = net(data)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
        losses.append(loss)


    if params['save_model'][0]:
        logger.info('Model Saving........')
        torch.save({
            'epoch': epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss},  params['save_model'][1])
        logger.info("Model Saved! ")
        """
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor])

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        """
    plt.plot(range(epochs), losses)
    plt.show()

def generate_default_params():
    '''
    Generate default parameters for network.
    '''
    params = {
        'hidden_size_1': 32,
        'hidden_size_2': 32,
        'hidden_size_3': 32,
        'act_func': 'ReLU',
        'learning_rate': 0.0025,
        'optimizer': 'Adam',
        'loss': 'SmoothL1Loss',
        'batch_size': 10,
        'save_model': (True, 'models/checkpoints/simplenet170121'),
        # Maybe change the path to a non static variable
        'load_model': (False, None)

    }
    return params  # from experiment jX3RYvtW that LR and Batch Size should be 50 and 0.005


if __name__ == '__main__':
    params = generate_default_params()
    torch.manual_seed(42)
    main(params)
