import torch
from torch import nn
from torch.nn import functional as F
from .layer_unit import DPRNN
alpha = 0.25
EPS = 1e-8


def complexMulti(a, b):
    c1 = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    c2 = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    c = torch.stack((c1, c2), -1)
    return c


class Encoder(nn.Module):
    def __init__(self, N, L, stride=None):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        if stride is None:
            self.stride = L // 2
        else:
            self.stride = stride
        self.window = nn.Parameter(torch.hamming_window(self.L), requires_grad=False)

    def pad_segment(self, inputs):
        r = self.stride - (inputs.size(-1) - self.L) % self.stride
        if r > 0:
            x = F.pad(inputs, [0, r])
        return x

    def stft(self, inputs):
        return torch.stft(inputs, self.L, self.stride, window=self.window, center=False, return_complex=False)

    def forward(self, mixture):
        mixture = self.pad_segment(mixture)
        x = self.stft(mixture)
        x = torch.transpose(x, 1, 3)
        return x


class Decoder(nn.Module):
    def __init__(self, N, L, stride=None):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        if stride is None:
            self.stride = L // 2
        else:
            self.stride = stride
        self.conv_A = nn.ConvTranspose2d(N, 1, kernel_size=[1, 5], stride=[1, 2])
        self.conv_P = nn.ConvTranspose2d(N, 2, kernel_size=[1, 5], stride=[1, 2])
        self.window = nn.Parameter(torch.hamming_window(self.L), requires_grad=False)
        self.kernel_len = self.L

    def istft(self, inputs):
        length = (inputs.size(2) - 1) * self.stride + self.kernel_len
        return torch.istft(inputs, self.L, self.stride, window=self.window, center=False, length=length, return_complex=False)

    def forward(self, inputs_en, xi):
        Am = torch.relu(self.conv_A(xi))
        P = self.conv_P(xi)
        norm_P = torch.norm(P, dim=1, keepdim=True) + EPS
        P = P / norm_P
        x = torch.norm(inputs_en, dim=1, keepdim=True) * Am * P
        x = torch.transpose(x, 1, 3)
        x = self.istft(x)
        return x


class DPRNN_ME(nn.Module):
    def __init__(self, N, B, H, R, K, rnn_type='LSTM', dropout=0, bidirectional=False, mask_nonlinear='relu'):
        super().__init__()
        self.N, self.B, self.H, self.R, self.K = N, B, H, R, K
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.conv_noisy = nn.Conv2d(2, B, [5, 5], [1, 2], bias=False)
        self.conv_refer = nn.Conv2d(2, B, [5, 5], [1, 2], bias=False)
        self.dprnn = DPRNN(N, B, H, R, K, rnn_type, dropout=dropout, bidirectional=bidirectional)
        self.output_audio = Output(B, N, mask_nonlinear)

    def forward(self, noisy, refer, inference=False):
        x = noisy
        y = refer
        x = F.pad(x, (0, 0, 4, 0))
        y = F.pad(y, (0, 0, 4, 0))
        x = self.conv_noisy(x)
        y = self.conv_refer(y)
        x = self.dprnn(x, y, inference)
        x = self.output_audio(x)
        return x


class Output(nn.Module):
    def __init__(self, B, N, mask_nonlinear='relu'):
        super().__init__()
        self.B = B
        self.N = N
        mask_nonlinear = mask_nonlinear.lower()
        if mask_nonlinear == 'relu':
            self.mask = torch.relu
        elif mask_nonlinear == 'sigmoid':
            self.mask = torch.sigmoid
        elif mask_nonlinear == 'linear':
            self.mask = lambda x: x
        elif mask_nonlinear == 'lrelu':
            self.mask = nn.LeakyReLU()
        elif mask_nonlinear == 'elu':
            self.elu = nn.ELU()
            self.mask = lambda x: self.elu(x) + 1
        else:
            raise ValueError
        self.prelu = nn.PReLU()
        self.mask_conv = nn.Conv2d(B, N, 1)

    def forward(self, inputs):
        x = self.prelu(inputs)
        x = self.mask_conv(x)
        x = self.mask(x)
        return x
