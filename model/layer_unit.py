import torch
from torch import nn
from torch.nn import functional as F
EPS = 1e-8


class DPRNN(nn.Module):
    def __init__(self, N, B, H, R, K, rnn_type, dropout=0, bidirectional=False):
        super().__init__()
        self.N, self.B, self.H, self.R, self.K = N, B, H, R, K
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm_x = nn.ModuleList([])
        self.row_norm_y = nn.ModuleList([])
        self.col_norm_x = nn.ModuleList([])
        self.col_norm_y = nn.ModuleList([])
        for i in range(R):
            self.row_rnn.append(ReferRNN(rnn_type, B, K, col=False, dropout=dropout, bidirectional=True))
            self.col_rnn.append(ReferRNN(rnn_type, B, K, col=True, dropout=dropout, bidirectional=bidirectional))
            if i < R - 1:
                self.row_norm_x.append(BlockGroupNorm(B, 2))
                self.col_norm_x.append(BlockGroupNorm(B, 2))
                self.row_norm_y.append(BlockGroupNorm(B, 2))
                self.col_norm_y.append(BlockGroupNorm(B, 2))
        self.output_x = nn.Sequential(nn.PReLU(), nn.Conv2d(B, B, 1))

    def forward(self, xi, yi, inference=False):
        batch_size, B, T, K = xi.shape
        x = xi.permute(0, 2, 3, 1)
        y = yi.permute(0, 2, 3, 1)
        for i in range(self.R):
            mx, my = self.row_rnn[i](x, y)
            if i < self.R - 1:
                x = self.row_norm_x[i](mx + x)
                y = self.row_norm_y[i](my + y)
            else:
                x = mx + x
                y = my + y
            mx, my = self.col_rnn[i](x, y)
            if i < self.R - 1:
                x = self.col_norm_x[i](mx + x)
                y = self.col_norm_y[i](my + y)
            else:
                x = mx + x
                y = my + y

        x = x.permute(0, 3, 1, 2)
        x = self.output_x(x)
        return x


class ReferRNN(nn.Module):
    def __init__(self, rnn_type, inputs_size, segment_size, col, dropout=0, bidirectional=False, online=False):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.inputs_size = inputs_size
        self.segment_size = segment_size
        self.col = col
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.online = online
        num_direction = 2 if bidirectional else 1
        assert inputs_size % num_direction == 0
        hidden_size = inputs_size // num_direction
        self.dpt = Dropout2d(dropout)
        self.rnn0 = getattr(nn, self.rnn_type)(inputs_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.rnn1 = getattr(nn, self.rnn_type)(inputs_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.msw0 = nn.Parameter(0.5 * torch.ones([1, 1, inputs_size]))
        self.msw1 = nn.Parameter(0.5 * torch.ones([1, 1, inputs_size]))
        if not self.col:
            self.projx = nn.Linear(segment_size * 2, segment_size)
            self.projy = nn.Linear(segment_size * 2, segment_size)
        else:
            self.projx = nn.Linear(inputs_size * 2, inputs_size)
            self.projy = nn.Linear(inputs_size * 2, inputs_size)
        self.hidden_states = None

    def reset_buffer(self):
        self.hidden_states = None

    def forward(self, xi, yi):
        self.rnn0.flatten_parameters()
        self.rnn1.flatten_parameters()
        batch_size, T, K, B = xi.size()
        if not self.col:
            mx = torch.reshape(xi, [batch_size * T, K, B])
            my = torch.reshape(yi, [batch_size * T, K, B])
            x, _ = self.rnn0(mx, None)
            y, _ = self.rnn1(my, None)
            x = torch.reshape(x, [batch_size, T, K, -1])
            y = torch.reshape(y, [batch_size, T, K, -1])
        else:
            mx = torch.transpose(xi, 1, 2)
            my = torch.transpose(yi, 1, 2)
            mx = torch.reshape(mx, [batch_size * K, T, B])
            my = torch.reshape(my, [batch_size * K, T, B])
            if self.online:
                if self.hidden_states is not None:
                    hidden_states_0, hidden_states_1 = self.hidden_states
                else:
                    hidden_states_0 = hidden_states_1 = None
                x, hidden_states_0 = self.rnn0(mx, hidden_states_0)
                y, hidden_states_1 = self.rnn1(my, hidden_states_1)
                self.hidden_states = (hidden_states_0, hidden_states_1)
            else:
                x, _ = self.rnn0(mx, None)
                y, _ = self.rnn1(my, None)
            x = torch.reshape(x, [batch_size, K, T, -1])
            y = torch.reshape(y, [batch_size, K, T, -1])
            x = torch.transpose(x, 1, 2)
            y = torch.transpose(y, 1, 2)
        x = self.dpt(x)
        y = self.dpt(y)
        mx = x + self.msw0 * y
        my = y + self.msw1 * x
        if not self.col:
            x = torch.cat([xi, mx], dim=2)
            y = torch.cat([yi, my], dim=2)
            x = torch.transpose(x, 3, 2)
            y = torch.transpose(y, 3, 2)
            x = self.projx(x)
            y = self.projy(y)
            x = torch.transpose(x, 3, 2)
            y = torch.transpose(y, 3, 2)
        else:
            x = torch.cat([xi, mx], dim=3)
            y = torch.cat([yi, my], dim=3)
            x = self.projx(x)
            y = self.projy(y)
        return x, y


class BlockGroupNorm(nn.Module):
    '''
    axis -1 should be the time index, axis 1 should be the channel index.\\
    time delay: the block step.
    '''

    def __init__(self, in_channels, num_groups, elementwise_affine=True, pow_para=0.5, online=False):
        super().__init__()
        assert in_channels % num_groups == 0, 'in_channels: {}, num_groups: {}'.format(in_channels, num_groups)
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.elementwise_affine = elementwise_affine
        self.pow_para = pow_para
        self.online = online
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones([1, 1, 1, in_channels]))  # [1, N, 1]
            self.beta = nn.Parameter(torch.zeros([1, 1, 1, in_channels]))  # [1, N, 1]
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.gamma.data.fill_(1)
            self.beta.data.zero_()

    def forward(self, inputs):
        batch_size, T, K, C = inputs.shape
        x = torch.reshape(inputs, (batch_size, T, K, self.num_groups, C // self.num_groups))
        variances, means = torch.var_mean(x, dim=[2, 4], keepdim=True, unbiased=False)
        x = (x - means) / torch.pow(variances + EPS, self.pow_para)
        x = torch.reshape(x, [batch_size, T, K, C])
        if self.elementwise_affine:
            x = x * self.gamma + self.beta
        return x


class Dropout2d(nn.Dropout2d):
    def forward(self, x):
        x = x.contiguous()
        x = x.permute(0, 3, 1, 2)
        x = super().forward(x)
        x = x.permute(0, 2, 3, 1)
        return x
