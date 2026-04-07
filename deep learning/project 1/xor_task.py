from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE


# Specify the load_data arguments
data_path = 'xor_dataset.pt'
mean_subtraction = False
normalization = False

xor_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# For XOR, input has 2 features
in_features = 2
# Output has 2 classes (0 and 1)
out_size = 2
# Hidden units: try one hidden layer with 4 neurons (less complex model)
hidden_units = [4]
# Non-linearity: use 'tanH' for better gradient flow
non_linearity = ['tanH']

# create a network based on the architecture
net = create_net(in_features, hidden_units, non_linearity, out_size)

# specify the training opts
train_opts = {
    'num_epochs': 60,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 0,
    'batch_size': 4,
    'step_size': 25,
    'gamma': 1.0
}

# train  and save the model
train(net, xor_dataset, train_opts)
save(net, 'xor_solution.pt')
