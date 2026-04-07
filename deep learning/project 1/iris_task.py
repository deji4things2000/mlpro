from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE %%%

# Specify the load_data arguments
data_path = 'iris_dataset.pt'
mean_subtraction = False
normalization = False

iris_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# Iris has 4 input features
in_features = 4
# Output has 3 classes
out_size = 3
# Two hidden layers: try 10 neurons in first, 8 in second
hidden_units = [10, 8]
# Use tanH non-linearity after each hidden layer
non_linearity = ['tanH', 'tanH']

# create a network based on the architecture
net = create_net(in_features, hidden_units, non_linearity, out_size)

# specify the training opts
train_opts = {
    'num_epochs': 80,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'batch_size': 24,
    'step_size': 25,
    'gamma': 1.0
}

# Train and save the trained model
train(net, iris_dataset, train_opts)
save(net, "iris_solution.pt")
