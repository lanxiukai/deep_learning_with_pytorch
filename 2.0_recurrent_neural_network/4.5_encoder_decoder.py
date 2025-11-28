'''
Encoder-Decoder Architecture
'''

from torch import nn

#@save
class Encoder(nn.Module):
    """Basic encoder interface for the encoder-decoder architecture"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):  # encode the input X
        raise NotImplementedError  # to be implemented by subclass

#@save
class Decoder(nn.Module):
    """Basic decoder interface for the encoder-decoder architecture"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):  # initialize the decoder state based on the encoder outputs
        raise NotImplementedError  # to be implemented by subclass

    def forward(self, X, state):  # decode the decoder input X based on the decoder state
        raise NotImplementedError  # to be implemented by subclass

#@save
class EncoderDecoder(nn.Module):
    """
    Base class for the encoder-decoder architecture.
    
    Args:
        encoder: a encoder instance (Encoder)
        decoder: a decoder instance (Decoder)
        **kwargs: additional arguments
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """
        Forward pass for the encoder-decoder architecture.
        
        Args:
            enc_X: the encoder input
            dec_X: the decoder input
            *args: additional arguments
        Returns:
            The decoder output
        """
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
