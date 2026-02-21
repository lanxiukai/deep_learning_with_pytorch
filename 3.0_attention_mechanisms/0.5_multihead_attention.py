"""
Multi-head Attention
"""

import torch
from torch import nn
from d2l_importer import d2l_save

#@save
class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l_save.DotProductAttention(dropout)
        # p_q = p_k = p_v = num_hiddens/num_heads in one head
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values shape:
        # (batch_size, num_queries or num_key_value_pairs, num_hiddens)
        # valid_lens shape:
        # (batch_size,) or (batch_size, num_queries)
        # After projection, queries/keys/values shape:
        # (batch_size*num_heads, num_queries or num_key_value_pairs,
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, repeat each item (scalar or vector) num_heads times.
            # And then copy the second item, etc.
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output shape: (batch_size*num_heads, num_queries,
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat shape: (batch_size, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

#@save
def transpose_qkv(X, num_heads):
    """Transpose for parallel multi-head attention computation."""
    # Input X shape: (batch_size, num_queries or num_key_value_pairs, num_hiddens)
    # Output X shape: (batch_size, num_queries or num_key_value_pairs, num_heads,
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Output X shape: (batch_size, num_heads, num_queries or num_key_value_pairs,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # Final output shape: (batch_size*num_heads, num_queries or num_key_value_pairs,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

#@save
def transpose_output(X, num_heads):
    """Reverse the transpose_qkv operation."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
print(attention.eval())

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens).shape)
