import torch
from torch import nn
from torch.nn import functional as F
from .module_stft import Encoder, Decoder, DPRNN_ME
EPS = 1e-8


class TF_DSDPRNN(nn.Module):
    def __init__(self, N, L, B, H, R, K, rnn_type='LSTM', stride=None, dropout=0, bidirectional=False, encoder_activation='relu', mask=True, mask_nonlinear='relu'):
        super().__init__()
        self.N, self.L, self.B, self.H, self.R, self.K = N, L, B, H, R, K
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder_activation = encoder_activation
        self.mask = mask
        self.encoder_noisy = Encoder(N, L, stride=stride)
        self.encoder_refer = Encoder(N, L, stride=stride)
        self.mask_estimator = DPRNN_ME(N, B, H, R, K, rnn_type, dropout, bidirectional, mask_nonlinear)
        self.decoder_audio = Decoder(N, L, stride=stride)

    def forward(self, noisy, refer, inference=False):
        # [batch_size, length]
        length = noisy.size(1)
        noisy_en = self.encoder_noisy(noisy)
        refer_en = self.encoder_refer(refer)
        clean = self.mask_estimator(noisy_en, refer_en, inference)
        clean = self.decoder_audio(noisy_en, clean)
        clean = clean[:, :length]
        return clean
