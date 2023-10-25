"""
Credits to https://github.com/jrobine/twm/nets
"""


import math
import torch
import torch.nn as nn
import numpy as np



def _calculate_gain(nonlinearity, param=None):
    if nonlinearity == 'elu':
        nonlinearity = 'selu'
        param = 1
    elif nonlinearity == 'silu':
        nonlinearity = 'relu'
        param = None
    return torch.nn.init.calculate_gain(nonlinearity, param)


def _kaiming_uniform_(tensor, gain):
    # same as torch.nn.init.kaiming_uniform_, but uses gain
    fan = torch.nn.init._calculate_correct_fan(tensor, mode='fan_in')
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    torch.nn.init._no_grad_uniform_(tensor, -bound, bound)

def _get_initializer(name, nonlinearity=None, param=None):
    if nonlinearity is None:
        assert param is None
    if name == 'kaiming_uniform':
        if nonlinearity is None:
            # defaults from PyTorch
            nonlinearity = 'leaky_relu'
            param = math.sqrt(5)
        return lambda x: _kaiming_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == 'xavier_uniform':
        if nonlinearity is None:
            nonlinearity = 'relu'
        return lambda x: torch.nn.init.xavier_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == 'orthogonal':
        if nonlinearity is None:
            nonlinearity = 'relu'
        return lambda x: torch.nn.init.orthogonal_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == 'zeros':
        return lambda x: torch.nn.init.zeros_(x)
    else:
        raise ValueError(f'Unsupported initializer: {name}')
    
def init_(mod, weight_initializer=None, bias_initializer=None, nonlinearity=None, param=None):
    weight_initializer = _get_initializer(weight_initializer, nonlinearity, param) \
        if weight_initializer is not None else lambda x: x
    bias_initializer = _get_initializer(bias_initializer, nonlinearity='linear', param=None) \
        if bias_initializer is not None else lambda x: x

    def fn(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weight_initializer(m.weight)
            if m.bias is not None:
                bias_initializer(m.bias)

    return mod.apply(fn)

def get_activation(nonlinearity, param=None):
    if nonlinearity is None or nonlinearity == 'none' or nonlinearity == 'linear':
        return nn.Identity()
    elif nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'leaky_relu':
        if param is None:
            param = 1e-2
        return nn.LeakyReLU(negative_slope=param)
    elif nonlinearity == 'elu':
        if param is None:
            param = 1.0
        return nn.ELU(alpha=param)
    elif nonlinearity == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')
    
def get_norm_1d(norm, k):
    if norm is None or norm == 'none':
        return nn.Identity()
    elif norm == 'batch_norm':
        return nn.BatchNorm1d(k)
    elif norm == 'layer_norm':
        return nn.LayerNorm(k)
    else:
        raise ValueError(f'Unsupported norm: {norm}')


def get_norm_2d(norm, c, h=None, w=None):
    if norm == 'none':
        return nn.Identity()
    elif norm == 'batch_norm':
        return nn.BatchNorm2d(c)
    elif norm == 'layer_norm':
        assert h is not None and w is not None
        return nn.LayerNorm([c, h, w])
    else:
        raise ValueError(f'Unsupported norm: {norm}')

class _MultilayerModule(nn.Module):

    def __init__(self, layer_prefix, ndim, in_dim, num_layers, nonlinearity, param,
                 norm, dropout_p, pre_activation, post_activation,
                 weight_initializer, bias_initializer, final_bias_init):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.ndim = ndim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.param = param
        self.pre_activation = pre_activation
        self.post_activation = post_activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.final_bias_init = final_bias_init

        self.has_norm = norm is not None and norm != 'none'
        self.has_dropout = dropout_p != 0
        self.unsqueeze = in_dim == 0

        self.act = get_activation(nonlinearity, param)

    def reset_parameters(self):
        init_(self, self.weight_initializer, self.bias_initializer, self.nonlinearity, self.param)
        final_layer = getattr(self, f'{self.layer_prefix}{self.num_layers}')
        if not self.post_activation:
            init_(final_layer, self.weight_initializer, self.bias_initializer, nonlinearity='linear', param=None)
        if self.final_bias_init is not None:
            def final_init(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    with torch.no_grad():
                        m.bias.data.fill_(self.final_bias_init)
            final_layer.apply(final_init)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(-self.ndim)

        if x.ndim > self.ndim + 1:
            batch_shape = x.shape[:-self.ndim]
            x = x.reshape(-1, *x.shape[-self.ndim:])
        else:
            batch_shape = None

        if self.pre_activation:
            if self.has_norm:
                x = getattr(self, 'norm0')(x)
            x = self.act(x)

        for i in range(self.num_layers - 1):
            x = getattr(self, f'{self.layer_prefix}{i + 1}')(x)
            if self.has_norm:
                x = getattr(self, f'norm{i + 1}')(x)
            x = self.act(x)
            if self.has_dropout:
                x = self.dropout(x)
        x = getattr(self, f'{self.layer_prefix}{self.num_layers}')(x)

        if self.post_activation:
            if self.has_norm:
                x = getattr(self, f'norm{self.num_layers}')(x)
            x = self.act(x)

        if batch_shape is not None:
            x = x.unflatten(0, batch_shape)
        return x


class MLP(_MultilayerModule):

    def __init__(self, in_dim, hidden_dims, out_dim, nonlinearity, param=None, norm=None, dropout_p=0, bias=True,
                 pre_activation=False, post_activation=False,
                 weight_initializer='kaiming_uniform', bias_initializer='zeros', final_bias_init=None):
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__('linear', 1, in_dim, len(dims) - 1, nonlinearity, param, norm, dropout_p,
                         pre_activation, post_activation, weight_initializer, bias_initializer, final_bias_init)
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        if pre_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, in_dim)
            self.add_module(f'norm0', norm_layer)

        for i in range(self.num_layers - 1):
            linear_layer = nn.Linear(dims[i], dims[i + 1], bias=bias)
            self.add_module(f'linear{i + 1}', linear_layer)
            if self.has_norm:
                norm_layer = get_norm_1d(norm, dims[i + 1])
                self.add_module(f'norm{i + 1}', norm_layer)

        linear_layer = nn.Linear(dims[-2], dims[-1], bias=bias)
        self.add_module(f'linear{self.num_layers}', linear_layer)

        if post_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, dims[-1])
            self.add_module(f'norm{self.num_layers}', norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "swish": nn.SiLU(),
}


class GRURNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        std=np.sqrt(2),
        bias_const=0.0,
        activation="tanh",
    ):

        super().__init__()
        assert activation in ["relu", "tanh", "gelu", "swish"]

        self.fc1 = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)), activations[activation]
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(
            activations[activation],
            layer_init(nn.Linear(hidden_dim, out_dim), std=std, bias_const=bias_const),
        )

    def forward(self, x, hidden_state=None):
        out = self.fc1(x)
        hidden_state = self.rnn(out, hidden_state)
        out = self.fc2(hidden_state)
        return out, hidden_state


