'''
Recurrent Neural Networks (Concise Implementation)
'''

import torch
from torch import nn
from torch.nn import functional as F
from d2l_importer import d2l, d2l_save

#@save
class RNNModel(nn.Module):
    """Recurrent neural network model"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (introduced later), num_directions should be 2; otherwise it should be 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer first reshapes Y to (time steps * batch size, number of hidden units)
        # Its output shape is (time steps * batch size, vocabulary size).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU uses a tensor as its hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM uses a tuple as its hidden state
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l_save.load_data_time_machine(batch_size, num_steps)

    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    state = torch.zeros((1, batch_size, num_hiddens)) # (num_hidden_layers, batch_size, num_hiddens)
    print(state.shape)

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)

    device = d2l_save.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    # num_epochs, lr = 500, 1
    # d2l_save.train_ch8(net, train_iter, vocab, lr, num_epochs, device, net_name='RNNModel (concise)')

    # d2l.plt.ioff()
    # d2l.plt.show()

if __name__ == '__main__':
    main()
