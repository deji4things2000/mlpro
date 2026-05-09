from torch import save, random
from train import train
from create_sequence import create_sequence
from sample import sample
from rnn import RNN

random.manual_seed(0)
# ***** DO NOT EDIT ABOVE ****

# specify the dataset path and sequence length
# input_file =
# seq_len =

train_ds, val_ds, char_to_index, index_to_char = create_sequence(input_file, seq_len=seq_len)

vocab_size = len(char_to_index)

# specify the hidden size, number of layers and the RNN type
# hidden_size
# num_layers
# rnn_type

model = RNN(vocab_size, hidden_size=hidden_size, num_layers=num_layers, rnn_type=rnn_type)

# specify the directory to save the model checkpoints to
exp_dir = None

# specify train opts. train-opts should contain 'num_epochs', 'lr', 'batch_size' and 'weight_decay'
train_opts = {
    # "num_epochs":
    # "lr": ,
    # "batch_size":
    # "weight_decay":
}

train(model, train_ds, val_ds, train_opts=train_opts, exp_dir=exp_dir)
save(model.state_dict(), "basic-rnn.pt")


# time to compose essays
seed = "ACT "
sample_len = 2000
sampled_text = sample(model, seed=seed, sample_len=sample_len, char_to_index=char_to_index, index_to_char=index_to_char)
with open("basic_rnn_output.txt", 'w') as f:
    f.write(sampled_text)

print(sampled_text)

