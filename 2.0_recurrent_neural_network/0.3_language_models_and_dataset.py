'''
Language Models and Dataset
'''

import random
import torch
from d2l_importer import d2l_save

tokens = d2l_save.tokenize(d2l_save.read_time_machine())
# Because each line of text is not necessarily a sentence or a paragraph
# we concatenate all the lines into a single list
corpus = [token for line in tokens for token in line]
vocab = d2l_save.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
d2l_save.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l_save.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l_save.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l_save.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a batch of subsequences using random sampling"""
    # Partition the sequence starting from a random offset, whose range includes num_steps - 1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 because we need to account for the labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # Initial indices of subsequences with length num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # During iteration with random sampling, subsequences from two adjacent random batches 
    # are not necessarily adjacent in the original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return the subsequence of length num_steps starting from index pos
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, initial_indices contains the random initial indices of subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # Yield the subsequences as tensors: (batch_size, num_steps)
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a batch of subsequences using sequential partitioning"""
    # Split the sequence starting from a random offset
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        # Yield the subsequences as tensors: (batch_size, num_steps)
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

class SeqDataLoader:  #@save
    """Iterator for loading sequence data"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = d2l_save.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and vocabulary for the Time Machine dataset"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# d2l_save.plt.ioff()
# d2l_save.plt.show()
