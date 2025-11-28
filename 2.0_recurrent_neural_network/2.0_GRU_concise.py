'''
Gated Recurrent Units (GRU) (Concise Implementation)
'''

from torch import nn
from d2l_importer import d2l_save

batch_size, num_steps = 32, 35
train_iter, vocab = d2l_save.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l_save.try_gpu()
num_epochs, lr = 500, 1

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l_save.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l_save.train_ch8(model, train_iter, vocab, lr, num_epochs, device,
                   net_name='GRU (concise)')

# d2l_save.plt.ioff()
# d2l_save.plt.show()
