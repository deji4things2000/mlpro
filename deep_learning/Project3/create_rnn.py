import torch
from lstm_cell import LSTMCell
from basic_rnn_cell import BasicRNNCell
from torch import nn, zeros, empty_like


class CustomRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        """
        Creates an recurrent neural network of type {basic_rnn, lstm_rnn}

        basic_rnn is an rnn whose layers implement a tanH activation function
        lstm_rnn is ann rnn whose layers implement an LSTM cell

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in each layer of the RNN.
        num_layers: (int), the number of RNN layers at each time step
        rnn_type: (string), the desired rnn type. rnn_type is a member of {'basic_rnn', 'lstm_rnn'}
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # create a ModuleList self.rnn to hold the layers of the RNN
        # and append the appropriate RNN layers to it
        self.rnn = nn.ModuleList()
        
        for i in range(num_layers):
            # First layer takes vocab_size as input, subsequent layers take hidden_size
            input_size = vocab_size if i == 0 else hidden_size
            if rnn_type == 'basic_rnn':
                self.rnn.append(BasicRNNCell(input_size, hidden_size))
            else:  # lstm_rnn
                self.rnn.append(LSTMCell(input_size, hidden_size))

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x the mini-batch of input sequence
        h: (Tensor) of size (l x B x m) where l is the number of layers and m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (l x B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the final hidden state of each time step in order
        h: (Tensor) of size (l x B x m), the hidden state of the last time step
        c: (Tensor) of size (l x B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """

        # compute the hidden states and cell states (for an lstm_rnn) for each mini-batch in the sequence
        B, T, _ = x.shape
        m = self.hidden_size
        l = self.num_layers
        
        # Output tensor: final hidden state of each time step
        outs = torch.zeros(B, T, m, device=x.device)
        
        # Initialize hidden states for each layer (will be updated per time step)
        # We'll build a list of hidden states per layer
        h_list = [h[layer_idx] for layer_idx in range(l)]
        if self.rnn_type == 'lstm_rnn':
            c_list = [c[layer_idx] for layer_idx in range(l)]
        
        # Iterate over time steps
        for t in range(T):
            # Input at current time step
            x_t = x[:, t, :]  # Shape: (B, n)
            
            # Iterate over layers
            for layer_idx, layer in enumerate(self.rnn):
                # Get current hidden state for this layer
                h_prev = h_list[layer_idx]  # Shape: (B, m)
                
                if self.rnn_type == 'basic_rnn':
                    # Forward through basic RNN cell
                    h_next = layer(x_t, h_prev)
                    h_list[layer_idx] = h_next
                    x_t = h_next  # Pass to next layer
                else:  # lstm_rnn
                    c_prev = c_list[layer_idx]
                    h_next, c_next = layer(x_t, h_prev, c_prev)
                    h_list[layer_idx] = h_next
                    c_list[layer_idx] = c_next
                    x_t = h_next
            
            # Store output of the last layer at this time step
            outs[:, t, :] = x_t
        
        # Stack the hidden states back into a tensor
        h_out = torch.stack(h_list, dim=0)
        
        if self.rnn_type == 'lstm_rnn':
            c_out = torch.stack(c_list, dim=0)
            return outs, h_out, c_out
        else:
            return outs, h_out, c