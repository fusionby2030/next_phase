import numpy as np
import torch

# tensors look like this

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(4., requires_grad=True)

# Doing arithmetic

y = w * x + b
# Output
# tensor(17., grad_fn=<AddBackward0>)

# We can then compute the gradients w.r.t to the tensors in y
y.backward()

# Gradients

w_grad = w.grad
# 3
b_grad = b.grad
# 1

"""
General Problem: 
Predict target variables using input features 

Linear Regression, each target variable is estimated to be a weighted
 sum of the input variables, offset by some constant (bias)

The objective is to find a suitable set of weights and biases using the training data to make accurate predictions

Sample Problem
Using features of temp, rainfall, and humidity, predict the crop yield of apples and oranges
"""

# Training Data
# Inputs (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Convert to Tensors

inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)

"""
Simple Linear Regression Model Implementation 
"""

# Weights and Biases are tensors initialized with random values.
# This part I am concerned with philosophically, i.e., different random initialized states change output entirely
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)


# first row of w and first element of b are used to predict first target variable (apples)
# second row of w and second element of b for second target variable (oranges)

# Matrix multiplication of input x with w transposed plus the bias

def model(x):
    return x @ w.t() + b


# predictions made by passing the input data through model
preds = model(inputs)

# now we could compare the preds with the targets, but we know it will be quite different

"""
Loss Function 

Used to compare predictions with actual targets
Use Mean Squared Error for comparison 

Loss provides an indication on how bad the model is predicting the target variables. 
Generally a lower loss the better the model ( although many factors play into this )

From calculus we know that the gradient of the loss represents the rate of change of the loss. 
If the gradient of the loss w.r.t element (weight or bias) is:
    positive: 
        increasing the element will increase the loss 
        decreasing the element will decrease the loss
    negative:
        increasing the element will decrease the loss 
        decreasing the element will increase the loss 
        
"""


# Loss Function
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff*diff) / diff.numel()


# Compute loss
loss = mse(preds, targets)
# print(loss)

# Now compute the gradients of loss
loss.backward()

# gradients stored in the .grad property

"""
Reducing Loss using all famous gradient descent 

Steps:
1 Generate predictions
2 Calculate the loss
3 Compute gradients w.r.t the weights and biases
4 Adjust the weights by subtracting a small quantity proportional to the gradient
5 Reset the gradients to zero

"""

# 1 Generate Predictions
preds = model(inputs)

# 2 Calculate Loss
loss = mse(preds, targets)
print(loss)

# 3 compute gradients
loss.backward()

# 4 Adjust Weights and 5 Reset Gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= -b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

# New weights and biases to the model should have lower loss.

# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)

"""
Now we can establish a training loop
Iterations are dubbed epochs
"""
# train for 100 epochs
for _ in range(100):
    # 1 Generate Predictions
    preds = model(inputs)
    # 2 Calculate Loss
    loss = mse(preds, targets)
    # 3 compute gradients
    loss.backward()
    # 4 Adjust Weights and 5 Reset Gradients
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= -b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

"""
Check if predictions are slowly converging to targets
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
print('\n# Predictions')
print(preds)
print('\n# Targets')
print(targets)
"""


"""
Linear Regression using pytorch built-ints
"""

# Use same Targets and Inputs

# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
# This will be able to split the data into batches while training, and utilize shuffling and sampling

train_ds = TensorDataset(inputs, targets)

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
# Call next batch using next(iter(train_dl))

# Use nn.Linear to initialize the weights and biases

# Define Model
model = torch.nn.Linear(3, 2)
# model.weight is equiv. to w
# model.bias is equiv. to b

# Instead of using gradients manual use Stochastic Gradient Descent from optim.SGD

# Define Optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Loss Function is pre build
import torch.nn.functional as F

# Define Loss Function

loss_fn = F.mse_loss

loss = loss_fn(model(inputs), targets)

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))


# Train the model for 100 epochs
fit(100, model, loss_fn, opt)



