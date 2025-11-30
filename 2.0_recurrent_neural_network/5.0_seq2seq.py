'''
Sequence to Sequence (Seq2Seq)
'''

import collections
import math
import torch
from torch import nn
from d2l_importer import d2l_save

#@save
class Seq2SeqEncoder(d2l_save.Encoder):
    """Recurrent neural network encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # Input 'X' shape: (batch_size, num_steps)
        X = self.embedding(X)   # Output 'X' shape: (batch_size, num_steps, embed_size)
        # In recurrent neural network models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)  # Permute the dimensions of 'X' to (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)
        # if state is not specified, it defaults to zeros
        # Shape of output: (num_steps, batch_size, num_hiddens)
        # Shape of state: (num_layers, batch_size, num_hiddens)
        return output, state

class Seq2SeqDecoder(d2l_save.Decoder):
    """Recurrent neural network decoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)  # Permute the dimensions of 'X' to (num_steps, batch_size, embed_size)
        # Broadcast context so that it has the same num_steps as X,
        # Repeating the state of the last layer (state[-1], shape: (batch_size, num_hiddens)) for each time step
        context = state[-1].repeat(X.shape[0], 1, 1)  # Shape of context: (num_steps, batch_size, num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        # Shape of X_and_context: (num_steps, batch_size, embed_size + num_hiddens)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # Shape of output: (batch_size, num_steps, vocab_size)
        # Shape of state: (num_layers, batch_size, num_hiddens)
        return output, state

#@save
def sequence_mask(X, valid_len, value=0):
    """
    Mask irrelevant items in sequences.

    Args:
        X: the input sequence (batch_size, num_steps)
        valid_len: the valid length of the input sequence (batch_size,)
        value: the value to fill the irrelevant items with (Default: 0)
    Returns:
        The masked input sequence (batch_size, num_steps),
        where the irrelevant items are filled with the value.
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]  # Shape of mask: (batch_size, num_steps)
    X[~mask] = value  # Fill the irrelevant items with the value
    return X

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """Softmax cross-entropy loss with masking."""
    # Shape of pred: (batch_size, num_steps, vocab_size)
    # Shape of label: (batch_size, num_steps)
    # Shape of valid_len: (batch_size,)
    def forward(self, pred, label, valid_len):  # Rewrite the forward method of the base class nn.CrossEntropyLoss
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'  # Set the reduction method to 'none' to return the loss for each item
        # Permute the dimensions of 'pred' to (batch_size, vocab_size, num_steps) 
        # to meet the needs of the base class nn.CrossEntropyLoss (input shape: (N, C, ...))
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)  # Call the forward method of the base class nn.CrossEntropyLoss
        # Shape of unweighted_loss: (batch_size, num_steps)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # Shape of weighted_loss: (batch_size,)
        return weighted_loss

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device, net_name=None):
    """Train a sequence-to-sequence model."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    if net_name is not None:
        print(f'\n{net_name} is training on {device} ...')
    else:
        print(f'\nTraining on {device} ...')
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l_save.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    timer = d2l_save.Timer()
    total_tokens = 0.0
    for epoch in range(num_epochs):
        metric = d2l_save.Accumulator(2)  # l.sum(), num_tokens
        for batch in data_iter:
            timer.start()
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)  # Shape of bos: (batch_size, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing (Shape of dec_input: (batch_size, num_steps + 1))
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # Perform backpropagation using the scalar loss
            d2l_save.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            timer.stop()
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        total_tokens += metric[1]
    tokens_per_sec = total_tokens / timer.sum()
    print(f'loss {metric[0] / metric[1]:.3f}, {tokens_per_sec:.1f} '
          f'tokens/sec, total time: {timer.format_time()}')

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Prediction for sequence-to-sequence model."""
    # Set net to evaluation mode during prediction
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l_save.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add a batch dimension (dim=0) using unsqueeze to make the shape of enc_X: (1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add a batch dimension (dim=0) using unsqueeze to make the shape of dec_X: (1, 1)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)  # Shape of Y: (1, 1, vocab_size)
        # Use the token with the highest predicted probability as the decoder input at the next time step
        dec_X = Y.argmax(dim=2)  # Shape of dec_X: (1, 1) (greedy search)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()  # .item() to get the scalar value
        # Save attention weights (to be discussed later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, stop generating the output sequence
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """Compute BLEU (Bilingual Evaluation Understudy)."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                # Prevent the same label n-gram sub-sequence from being matched multiple times
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def main():
    # encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # encoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long)
    # output, state = encoder(X)
    # print(output.shape)
    # print(state.shape)

    # decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # decoder.eval()
    # state = decoder.init_state(encoder(X))
    # output, state = decoder(X, state)
    # print(output.shape, state.shape)

    # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(sequence_mask(X, torch.tensor([1, 2])))

    # X = torch.ones(2, 3, 4)
    # print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    # loss = MaskedSoftmaxCELoss()
    # print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
    #      torch.tensor([4, 2, 0])))

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l_save.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l_save.load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                            dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                            dropout)
    net = d2l_save.EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, 
                  net_name='Seq2Seq-GRU')

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

if __name__ == '__main__':
    main()
