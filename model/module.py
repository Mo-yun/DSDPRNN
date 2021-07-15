import torch
from torch import nn
from torch.nn import functional as F
import math
from .layer_unit import DPRNN
alpha = 0.25
EPS = 1e-8


class Encoder(nn.Module):
    def __init__(self, N, L, stride=None, activation='relu'):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.activation = activation
        if activation is None:
            self.act = lambda x: x
        else:
            self.act = getattr(torch, activation.lower())
        if stride is None:
            self.stride = L // 2
        else:
            self.stride = stride
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=self.stride, bias=False)

    def pad_segment(self, inputs):
        r = self.stride - (inputs.size(-1) - self.L) % self.stride
        if r > 0:
            x = F.pad(inputs, [0, r])
        return x

    def forward(self, mixture):
        mixture = self.pad_segment(mixture)
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = self.conv1d_U(mixture)
        mixture_w = self.act(mixture_w)
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L, stride=None):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        if stride is None:
            self.stride = L // 2
        else:
            self.stride = stride
        self.ConvTrans = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=self.stride, bias=True)

    def forward(self, outputs_en):
        est_source = self.ConvTrans(outputs_en)
        est_source = torch.squeeze(est_source, 1)
        return est_source


class DPRNN_ME(nn.Module):
    def __init__(self, N, B, H, R, K, rnn_type='LSTM', dropout=0, bidirectional=False, mask_nonlinear='relu'):
        super().__init__()
        self.N, self.B, self.H, self.R, self.K = N, B, H, R, K
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.stride = self.K // 2
        self.conv_noisy = nn.Conv1d(N, B, 1, bias=False)
        self.conv_refer = nn.Conv1d(N, B, 1, bias=False)
        self.dprnn = DPRNN(N, B, H, R, K, rnn_type, dropout=dropout, bidirectional=bidirectional)
        self.output_audio = Output(B, N, mask_nonlinear)

    def pad_segment(self, inputs):
        r = (inputs.size(2) - self.K) % self.stride
        if r > 0:
            r = self.stride - r
            inputs = F.pad(inputs, [0, r])
        return inputs

    def split_feature(self, inputs):
        x = self.pad_segment(inputs)
        x = x.unfold(2, self.K, self.stride)
        return x

    def merge_feature(self, inputs):
        x = self.overlap_and_add(inputs, self.stride)
        return x

    @staticmethod
    def overlap_and_add(signal, frame_step):
        """Reconstructs a signal from a framed representation.

        Adds potentially overlapping frames of a signal with shape
        `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
        The resulting tensor has shape `[..., output_size]` where

            output_size = (frames - 1) * frame_step + frame_length

        Args:
            signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
            frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

        Returns:
            A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
            output_size = (frames - 1) * frame_step + frame_length

        Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
        """
        outer_dimensions = signal.size()[:-2]
        frames, frame_length = signal.size()[-2:]

        subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
        subframe_step = frame_step // subframe_length
        subframes_per_frame = frame_length // subframe_length
        output_size = frame_step * (frames - 1) + frame_length
        output_subframes = output_size // subframe_length

        subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

        frame = torch.arange(0, output_subframes, dtype=torch.int64, device=signal.device).unfold(0, subframes_per_frame, subframe_step)
        frame = frame.contiguous().view(-1)

        result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
        result.index_add_(-2, frame, subframe_signal)
        result = result.view(*outer_dimensions, -1)
        return result

    def forward(self, noisy, refer, inference=False):
        length = noisy.size(2)
        x = noisy
        y = refer
        x = self.conv_noisy(x)
        y = self.conv_refer(y)
        x = self.split_feature(x)
        y = self.split_feature(y)
        x = self.dprnn(x, y, inference)
        # batch_size, B, T, K = x.shape
        x = self.merge_feature(x)
        x = x[:, :, :length]
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
        self.mask_conv = nn.Conv1d(B, N, 1)

    def forward(self, inputs):
        x = self.prelu(inputs)
        x = self.mask_conv(x)
        x = self.mask(x)
        return x
