'''
Recurrent Neural Networks
'''

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l_importer import d2l_save

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Parameters for the hidden layer
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Parameters for the output layer
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    # Initialize the hidden state of the RNN to zero
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    # Shape of inputs: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of X: (batch_size, vocab_size)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # outputs: (num_steps * batch_size, vocab_size), state: (batch_size, num_hiddens)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:  #@save
    """Recurrent neural network model implemented from scratch"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):  # allow the object to be called like a function
        # X: (batch_size, num_steps), state: (batch_size, num_hiddens)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # X: (num_steps, batch_size, vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the given prefix"""
    state = net.begin_state(batch_size=1, device=device)  # (1, num_hiddens)
    outputs = [vocab[prefix[0]]]  # [index] of the prefix and generated characters
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # [last index] -> (1, 1): (num_steps, batch_size)
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)  # warm-up state only
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict for num_preds steps
        y, state = net(get_input(), state)  # (1 * 1, vocab_size), (1, num_hiddens)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):  #@save
    """Clip gradients (global norm clipping)"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter, timer):
    """
    Train the network for one epoch (see Chapter 8 for the definition).
    
    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
        device: the device to use
        use_random_iter: whether to use random sampling
        timer: the timer instance (Timer)
    Returns:
        A tuple of perplexity and the number of tokens: (ppl, num_tokens)
        - ppl: the perplexity
        - num_tokens: the number of tokens (num_steps * batch_size)
    """
    state = None
    metric = d2l_save.Accumulator(2)  # l.sum(), num_tokens (num_steps * batch_size)
    for X, Y in train_iter:
        timer.start()
        if state is None or use_random_iter:
            # Initialize state during the first iteration or when using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # detach the state from the computation graph, avoid gradient explosion
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # For nn.GRU, state is a tensor
                state.detach_()
            else:
                # For nn.LSTM or for our scratch implementation, state is a tuple of tensors
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)  # Flatten over num_steps (num_steps * batch_size)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)  # (num_steps * batch_size, vocab_size), (batch_size, num_hiddens)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater()
        metric.add(l * y.numel(), y.numel())
        timer.stop()
    return math.exp(metric[0] / metric[1]), metric[1]  # perplexity, num_tokens

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False, net_name=None):
    """
    Train the model (see Chapter 8 for the definition).
    
    Args:
        net: the network
        train_iter: the training data iterator
        vocab: a vocabulary object
        lr: the learning rate
        num_epochs: the number of epochs
        device: the device to use
        use_random_iter: whether to use random sampling (Default: False)
        net_name: the name of the network (Default: None)
    Returns:
        None: prints the perplexity, speed, and total time, and the generated characters
    """
    loss = nn.CrossEntropyLoss()
    animator = d2l_save.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialization
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda: d2l_save.sgd(net.params, lr)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    if net_name is not None:
        print(f'\n{net_name} is training on {str(device)} ...')
    else:
        print(f'\nTraining on {str(device)} ...')
    
    timer, total_tokens = d2l_save.Timer(), 0.0
    # Training and prediction
    for epoch in range(num_epochs):
        ppl, num_tokens = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter, timer)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
        total_tokens += num_tokens
    print(f'Perplexity {ppl:.1f}, {total_tokens / timer.sum():.1f} tokens/sec, total time: {timer.format_time()}')
    print(predict('time traveller'))
    print(predict('traveller'))

def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l_save.load_data_time_machine(batch_size, num_steps)

    # print(F.one_hot(torch.tensor([0, 2]), len(vocab)))  # one-hot encoding, (2, 28)
    # X = torch.arange(10).reshape((2, 5))
    # print(F.one_hot(X.T, 28).shape)  # (5, 2, 28)

    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l_save.try_gpu(), 
                          get_params, init_rnn_state, rnn)
    # state = net.begin_state(X.shape[0], d2l_save.try_gpu())
    # Y, new_state = net(X.to(d2l_save.try_gpu()), state)
    # print(Y.shape, len(new_state), new_state[0].shape)
    # (num_steps * batch_size, vocab_size), 1, (batch_size, num_hiddens)

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l_save.try_gpu(), 
              net_name='RNN Scratch (sequential partitioning)')

    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l_save.try_gpu(),
              use_random_iter=True, net_name='RNN Scratch (random sampling)')
    
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()

if __name__ == '__main__':
    main()
