'''
Text Preprocessing
'''

import collections
import re
from d2l_importer import d2l_save

#@save
d2l_save.DATA_HUB['time_machine'] = (d2l_save.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the Time Machine dataset into a list of text lines"""
    with open(d2l_save.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# The total number of text lines: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens"""
    if token == 'word':
        return [line.split() for line in lines]     # 2D list [([word])]
    elif token == 'char':
        return [list(line) for line in lines]  # 2D list [([char])]
    else:
        print('Error: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
print('--------------------------------')

class Vocab:  #@save
    """Text vocabulary"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort by frequency
        counter = count_corpus(tokens)
        # Sorted by frequency in descending order, type: list of tuples [(token, frequency)]
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index of the unknown token is 0, and the reserved tokens are prepended
        # 1D list [token], the index is the position of the token in the list
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 2D dictionary {token: index}
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):  # if tokens is a single token
            # return the index of the token if it exists, otherwise return the index of the unknown token (0)
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):  # if indices is a single index
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        '''The index of the unknown token is 0'''
        return 0

    @property
    def token_freqs(self):
        '''Return a list of tuples sorted by token frequencies: [(token, frequency)]'''
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Count token frequencies"""
    # Here tokens is a 1D list or a 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a 2D list of tokens into a single list
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('Text:', tokens[i])
    print('Indices:', vocab[tokens[i]])

def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return corpus indices and vocabulary of the Time Machine dataset"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Because each line in the Time Machine dataset is not necessarily a sentence or a paragraph,
    # we flatten all text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]  # hidden flatten: 1D list [index]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
print(vocab.idx_to_token[:10])
