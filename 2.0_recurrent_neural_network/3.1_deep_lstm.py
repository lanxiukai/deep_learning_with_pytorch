'''
Deep Long Short-Term Memory (LSTM) (concise implementation)
'''

from torch import nn
from d2l_importer import d2l_save

batch_size, num_steps = 32, 35
train_iter, vocab = d2l_save.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l_save.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l_save.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
d2l_save.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device, net_name='Double-layer LSTM (concise)')

# d2l_save.plt.ioff()
# d2l_save.plt.show()
