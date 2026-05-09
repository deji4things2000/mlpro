import torch
from torch import nn, sigmoid, tanh, Tensor
from math import sqrt


class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables

        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, (4 * m))
        # b, the vector of biases has size (4 * m)
        k = sqrt(1 / hidden_size)
        self.W = nn.Parameter(torch.empty(vocab_size, 4 * hidden_size).uniform_(-k, k))
        self.V = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size).uniform_(-k, k))
        self.b = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """
        a = self.b + torch.mm(x, self.W) + torch.mm(h, self.V)
        m = self.hidden_size
        a_i = a[:, :m] # input gate
        a_f = a[:, m:2*m] # forget gate
        a_o = a[:, 2*m:3*m] # output gate
        a_g = a[:, 3*m:] # candidate cell state

        #Apply activations
        i_t = sigmoid(a_i)
        f_t = sigmoid(a_f)
        o_t = sigmoid(a_o)
        g_t = tanh(a_g)

        # Update hidden and cell states
        c_out = f_t * c + i_t * g_t
        h_out = o_t * tanh(c_out)

        return h_out, c_out


