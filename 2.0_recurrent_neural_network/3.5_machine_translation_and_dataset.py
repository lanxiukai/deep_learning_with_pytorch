'''
Machine Translation and Dataset
'''

import os
import torch
from d2l_importer import d2l_save

#@save
d2l_save.DATA_HUB['fra-eng'] = (d2l_save.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():  # neural machine translation dataset
    """
    Load the 'English-French' dataset into a string of text.
        - The first column is the source language (English)
        - The second column is the target language (French)

    Returns:
        text (str): the raw text of the dataset
    """
    data_dir = d2l_save.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

#@save
def preprocess_nmt(text):
    """Preprocess the 'English-French' dataset"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking spaces with spaces
    # Convert uppercase letters to lowercase letters
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert spaces between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

#@save
def tokenize_nmt(text, num_examples=None):
    """
    Tokenize the 'English-French' dataset.
    
    Args:
        text (str): the raw text of the dataset
        num_examples (int): the number of examples (lines) to tokenize (Default: None)
    Returns:
        source: the source language tokens (2D list [[token]])
        target: the target language tokens (2D list [[token]])
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
print('Source:', source[:6])
print('Target:', target[:6])

#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot a histogram of list length pairs"""
    d2l_save.set_figsize()
    _, _, patches = d2l_save.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l_save.plt.xlabel(xlabel)
    d2l_save.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l_save.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);

# d2l_save.plt.ioff()
# d2l_save.plt.show()

src_vocab = d2l_save.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])  # source vocabulary
print('Length of source vocabulary:', len(src_vocab))

#@save
def truncate_pad(line, num_steps, padding_token):
    """
    Truncate or pad text sequences.
    
    Args:
        line: the text sequence
        num_steps: the number of steps (tokens)
        padding_token: the token index to pad
    Returns:
        The truncated or padded text sequence with length num_steps
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

answer = truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
print('Truncated and padded source sequence:', answer)

#@save
def build_array_nmt(lines, vocab, num_steps):
    """
    Convert machine translation text sequences into batches.
    
    Args:
        lines: the text sequences (2D list [[token]])
        vocab: the vocabulary (Vocab)
        num_steps: the number of steps (tokens)
    Returns:
        A tuple of tensors: (array, valid_len)
        - array: the indices array of tokenized text sequences (2D tensor [[index]])  (num_lines, num_steps)
        - valid_len: the valid length of each text sequence (1D tensor [length]) (num_lines)
    """
    lines = [vocab[l] for l in lines]              # convert to 2D list of indices [[index]]
    lines = [l + [vocab['<eos>']] for l in lines]  # add <eos> token index to the end of each line
    array = torch.tensor([truncate_pad(            # truncate or pad each line to num_steps
        l, num_steps, vocab['<pad>']) for l in lines])   # return a 2D tensor [[index]] (num_lines, num_steps)
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # the valid length (1D tensor [length]) (num_lines)
    return array, valid_len

#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    Return the iterator and vocabularies for the translation dataset.
    
    Args:
        batch_size: the batch size
        num_steps: the number of steps (tokens)
        num_examples: the number of examples (lines) to load (Default: 600)
    Returns:
        A tuple of: (data_iter, src_vocab, tgt_vocab)
        - data_iter: the data iterator (DataLoader)
        - src_vocab: the source vocabulary (Vocab)
        - tgt_vocab: the target vocabulary (Vocab)
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    reserved_tokens=['<pad>', '<bos>', '<eos>']
    # Build source and target vocabularies
    src_vocab = d2l_save.Vocab(source, min_freq=2, reserved_tokens=reserved_tokens)  # source vocabulary
    tgt_vocab = d2l_save.Vocab(target, min_freq=2, reserved_tokens=reserved_tokens)  # target vocabulary
    # Build source and target arrays by truncating or padding
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # Load data into data loader
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l_save.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=5, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('Valid length of X:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Valid length of Y:', Y_valid_len)
    break
