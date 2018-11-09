import torch.nn as nn


class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity="relu",
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        pred = self.linear(r_out)
        return pred, h_state
