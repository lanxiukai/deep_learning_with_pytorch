'''
Gated Recurrent Units (GRU)
'''

import torch
from d2l_importer import d2l_save

def get_gru_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # update gate parameters
    W_xr, W_hr, b_r = three()  # reset gate parameters
    W_xh, W_hh, b_h = three()  # candidate hidden state parameters
    # output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l_save.load_data_time_machine(batch_size, num_steps)
    
    vocab_size, num_hiddens, device = len(vocab), 256, d2l_save.try_gpu()
    num_epochs, lr = 500, 1
    model = d2l_save.RNNModelScratch(len(vocab), num_hiddens, device, get_gru_params,
                                init_gru_state, gru)
    d2l_save.train_ch8(model, train_iter, vocab, lr, num_epochs, device,
                       net_name='GRU (scratch)')

    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
