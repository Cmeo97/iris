import math
import torch
import torch.nn as nn
from functools import lru_cache
import copy
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class MovingMNISTDataset(Dataset):
    def __init__(self, file_path):
        # Load the dataset from the npy file
        self.data = np.load(file_path)
        # Convert to torch tensors
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data.shape[1]  # Assuming (num_frames, num_samples, height, width) is the shape of the loaded npy file

    def __getitem__(self, idx):
        return self.data[:, idx, :, :]  # Returning the entire sequence for a given sample

def get_dataloader(file_path, batch_size=64, shuffle=True, num_workers=6):
    dataset = MovingMNISTDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def plot_sequence(images_tensor, save_path='sequence_plot.png'):
    """
    Plot a sequence of images in a 2x10 grid and save the plot.
    
    Args:
    - images_tensor (torch.Tensor): A tensor of 20 images, each of shape (64, 64).
    - save_path (str): Path where the plot will be saved.
    """
    assert images_tensor.shape == (20, 64, 64), "The input tensor should have shape (20, 64, 64)."
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images_tensor[i].numpy(), cmap='gray')  # Convert tensor to numpy array for plotting
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save the figure with a specified dpi (resolution)
    plt.close()  # Close the figure to free up memory


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



class TransformerXLDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, max_length, mem_length, batch_first=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.mem_length = mem_length
        self.batch_first = batch_first

        self.pos_enc = PositionalEncoding(decoder_layer.dim, max_length, dropout_p=decoder_layer.dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim))
        self.v_bias = nn.Parameter(torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def init_mems(self):
        if self.mem_length > 0:
            param = next(self.parameters())
            dtype, device = param.dtype, param.device
            mems = []
            for i in range(self.num_layers + 1):
                mems.append(torch.empty(0, dtype=dtype, device=device))
            return mems
        else:
            return None

    def forward(self, x, positions, attn_mask, mems=None, tgt_length=None, return_attention=False):
        if self.batch_first:
            x = x.transpose(0, 1)

        if mems is None:
            mems = self.init_mems()
        print('mems:', mems)

        if tgt_length is None:
            tgt_length = x.shape[0]
        assert tgt_length > 0
        print('x', x)
        
        pos_enc = self.pos_enc(positions)
        hiddens = [x]
        attentions = []
        out = x
        for i, layer in enumerate(self.layers):
            out, attention = layer(out, pos_enc, self.u_bias, self.v_bias, attn_mask=attn_mask, mems=mems[i])
            hiddens.append(out)
            attentions.append(attention)

        out = out[-tgt_length:]

        if self.batch_first:
            out = out.transpose(0, 1)

        assert len(hiddens) == len(mems)
        with torch.no_grad():
            new_mems = []
            for i in range(len(hiddens)):
                cat = torch.cat([mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[-self.mem_length:].detach())
        if return_attention:
            attention = torch.stack(attentions, dim=-2)
            return out, new_mems, attention
        return out, new_mems


class TransformerXLDecoderLayer(nn.Module):

    def __init__(self, dim, feedforward_dim, head_dim, num_heads, activation, dropout_p, layer_norm_eps=1e-5):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.self_attn = RelativeMultiheadSelfAttention(dim, head_dim, num_heads, dropout_p)
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        print(x.shape)
        print(x)
        print(pos_encodings)
        print(u_bias)
        out, attention = self.self_attn(x, pos_encodings, u_bias, v_bias, attn_mask, mems)
        out = self.dropout(out)
        out = self.norm1(x + out)
        out = self.norm2(out + self._ff(out))
        return out, attention


class RelativeMultiheadSelfAttention(nn.Module):

    def __init__(self, dim, head_dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1 / (dim ** 0.5)

        self.qkv_proj = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.pos_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        tgt_length, batch_size = x.shape[:2]
        pos_len = pos_encodings.shape[0]

        if mems is not None:
            cat = torch.cat([mems, x], dim=0)
            qkv = self.qkv_proj(cat)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q[-tgt_length:]
        else:
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        pos_encodings = self.pos_proj(pos_encodings)
        print(pos_encodings)
        src_length = k.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim
        print(head_dim)

        q = q.view(tgt_length, batch_size, num_heads, head_dim)
        k = k.view(src_length, batch_size, num_heads, head_dim)
        v = v.view(src_length, batch_size, num_heads, head_dim)
        pos_encodings = pos_encodings.view(pos_len, num_heads, head_dim)
        print(pos_encodings)


        content_score = torch.einsum('ibnd,jbnd->ijbn', (q + u_bias, k))
        print(content_score)
        pos_score = torch.einsum('ibnd,jnd->ijbn', (q + v_bias, pos_encodings))
        pos_score = self._rel_shift(pos_score)
        print(pos_score)

        # [tgt_length x src_length x batch_size x num_heads]
        attn_score = content_score + pos_score
        attn_score.mul_(self.scale)
   

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_score = attn_score.masked_fill(attn_mask[:, :, None, None], -float('inf'))
            elif attn_mask.ndim == 3:
                attn_score = attn_score.masked_fill(attn_mask[:, :, :, None], -float('inf'))

        # [tgt_length x src_length x batch_size x num_heads]
        attn = F.softmax(attn_score, dim=1)
        return_attn = attn
        attn = self.dropout(attn)

        context = torch.einsum('ijbn,jbnd->ibnd', (attn, v))
        context = context.reshape(context.shape[0], context.shape[1], num_heads * head_dim)
        return self.out_proj(context), return_attn


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_length, dropout_p=0, batch_first=False):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        encodings = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encodings', encodings)

    def forward(self, positions):
        print('positions',positions)
        print('##################################################################################################')
        print(self.encodings.shape)
        out = self.encodings[positions]
        out = self.dropout(out)
        return out.unsqueeze(0) if self.batch_first else out.unsqueeze(1)

def same_batch_shape(tensors, ndim=2):
    batch_shape = tensors[0].shape[:ndim]
    assert all(t.ndim >= ndim for t in tensors)
    return all(tensors[i].shape[:ndim] == batch_shape for i in range(1, len(tensors)))


class PredictionNet(nn.Module):

    def __init__(self, modality_order, num_current, out_heads, embed_dim, activation, norm, dropout_p,
                 feedforward_dim, head_dim, num_heads, num_layers, memory_length, max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_length = memory_length
        self.modality_order = tuple(modality_order)
        self.num_current = num_current 
        self.z_dim = 128

        #self.embeds = nn.ModuleDict({
        #    name: nn.Embedding(embed['in_dim'], embed_dim) if embed.get('categorical', False) else
        #    MLP(embed['in_dim'], [], embed_dim, activation, norm=norm, dropout_p=dropout_p, post_activation=True)
        #    for name, embed in embeds.items()
        #})

        decoder_layer = TransformerXLDecoderLayer(
            embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p)

        num_modalities = len(modality_order)
        max_length = max_length * num_modalities + self.num_current
        mem_length = memory_length * num_modalities + self.num_current
        self.transformer = TransformerXLDecoder(decoder_layer, num_layers, max_length, mem_length, batch_first=True)

        self.out_heads = nn.ModuleDict({
            name: MLP(embed_dim, head['hidden_dims'], head['out_dim'], activation, norm=norm, dropout_p=dropout_p,
                      pre_activation=True, final_bias_init=head.get('final_bias_init', None))
            for name, head in out_heads.items()
        })

    @lru_cache(maxsize=20)
    def _get_base_mask(self, src_length, tgt_length, device):
        src_mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=device)
        num_modalities = len(self.modality_order)
        for tgt_index in range(tgt_length):
            # the last indices are always 'current'
            start_index = src_length - self.num_current
            src_index = src_length - tgt_length + tgt_index
            modality_index = (src_index - start_index) % num_modalities
            if modality_index < self.num_current:
                start = max(src_index - (self.memory_length + 1) * num_modalities, 0)
            else:
                start = max(src_index - modality_index - self.memory_length * num_modalities, 0)
            src_mask[tgt_index, start:src_index + 1] = False
        return src_mask

    def _get_mask(self, src_length, tgt_length, device, stop_mask):
        # prevent attention over episode ends using stop_mask
        num_modalities = len(self.modality_order)
        print(stop_mask.shape[1])
        print(num_modalities)
        print(self.num_current)
        print(src_length)
        assert stop_mask.shape[1] * num_modalities + self.num_current == src_length

        src_mask = self._get_base_mask(src_length, tgt_length, device)
        print(src_mask)

        batch_size, seq_length = stop_mask.shape
        stop_mask = stop_mask.t()
        stop_mask_shift_right = torch.cat([stop_mask.new_zeros(1, batch_size), stop_mask], dim=0)
        stop_mask_shift_left = torch.cat([stop_mask, stop_mask.new_zeros(1, batch_size)], dim=0)

        tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril()
        src = torch.logical_and(stop_mask_shift_left.unsqueeze(0), tril.unsqueeze(-1))
        src = torch.cummax(src.flip(1), dim=1).values.flip(1)

        shifted_tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril(diagonal=-1)
        tgt = torch.logical_and(stop_mask_shift_right.unsqueeze(1), shifted_tril.unsqueeze(-1))
        tgt = torch.cummax(tgt, dim=0).values

        idx = torch.logical_and(src, tgt)

        i, j, k = idx.shape
        idx = idx.reshape(i, 1, j, 1, k).expand(i, num_modalities, j, num_modalities, k) \
            .reshape(i * num_modalities, j * num_modalities, k)

        offset = num_modalities - self.num_current
        if offset > 0:
            idx = idx[:-offset, :-offset]
        idx = idx[-tgt_length:]

        src_mask = src_mask.unsqueeze(-1).tile(1, 1, batch_size)
        print(src_mask)
        src_mask[idx] = True
        print(src_mask)
        return src_mask

    def forward(self, inputs, tgt_length, stop_mask, heads=None, mems=None, return_attention=False):
        modality_order = self.modality_order
        num_modalities = len(modality_order)
        num_current = self.num_current

        #assert same_batch_shape([inputs[name] for name in modality_order[:num_current]])
        #if num_modalities > num_current:
        #    assert same_batch_shape([inputs[name] for name in modality_order[num_current:]])

        #embeds = {name: mod(inputs[name]) for name, mod in self.embeds.items()}

        #def cat_modalities(xs):
        #    batch_size, seq_len, dim = xs[0].shape
        #    return torch.cat(xs, dim=2).reshape(batch_size, seq_len * len(xs), dim)
        print('MEMS:', mems)
        if mems is None:
            history_length = inputs['z'].shape[1] - 1
            #if num_modalities == num_current:
            #    inputs = cat_modalities([embeds[name] for name in modality_order])
            #else:
            #    history = cat_modalities([embeds[name][:, :history_length] for name in modality_order])
            #    current = cat_modalities([embeds[name][:, history_length:] for name in modality_order[:num_current]])
            #    inputs = torch.cat([history, current], dim=1)
            tgt_length = (tgt_length - 1) * num_modalities + num_current
            src_length = history_length * num_modalities + num_current
            print(src_length)
            print(inputs['z'].shape)
            assert inputs['z'].shape[1] == src_length
            src_mask = self._get_mask(src_length, src_length, inputs['z'].device, stop_mask)
            
        else:
            sequence_length = inputs['z'].shape[1] - 1
            # switch order so that 'currents' are last
            #inputs = cat_modalities(
            #    [embeds[name] for name in (modality_order[num_current:] + modality_order[:num_current])])
            tgt_length = tgt_length * num_modalities
            mem_length = mems[0].shape[0]
            src_length = mem_length + sequence_length * num_modalities
            src_mask = self._get_mask(src_length, tgt_length, inputs['z'].device, stop_mask)

        positions = torch.arange(src_length - 1, -1, -1, device=inputs['z'].device)
        print(inputs)
        print(positions)
        #return  inputs['z'], positions, src_mask, mems, tgt_length, return_attention
        outputs = self.transformer(
            inputs['z'], positions, attn_mask=src_mask, mems=mems, tgt_length=tgt_length, return_attention=return_attention)
        hiddens, mems, attention = outputs if return_attention else (outputs + (None,))
#
        ## take outputs at last current
        assert hiddens.shape[1] == tgt_length
        out_idx = torch.arange(tgt_length - 1, -1, -num_modalities, device=inputs['z'].device).flip([0])
        hiddens = hiddens[:, out_idx]
        if return_attention:
            attention = attention[out_idx]
#
        if heads is None:
            heads = self.out_heads.keys()
#
        out = {name: self.out_heads[name](hiddens) for name in heads}

        return (out, hiddens, mems, attention) if return_attention else (out, hiddens, mems)
    

def check_no_grad(*tensors):
    return all((t is None or not t.requires_grad) for t in tensors)

class DynamicsModel(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
  
        
        modality_order = ['z']
        num_current = 1
        self.modality_order = modality_order
        out_heads = {'z': {'hidden_dims': [512], 'out_dim': z_dim}}
        memory_length = 16
        max_length = 1 + 17  # 1 for context
        self.prediction_net = PredictionNet(
            modality_order, num_current, out_heads, embed_dim=128,
            activation='silu', norm='none', dropout_p=0.1,
            feedforward_dim=1024, head_dim=64,
            num_heads=4, num_layers=10,
            memory_length=memory_length, max_length=max_length)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss =  nn.MSELoss()
    @property
    def h_dim(self):
        return self.prediction_net.embed_dim

    def predict(self, z, d, tgt_length, heads=None, mems=None, return_attention=False):
        
        assert check_no_grad(z, d)
        assert mems is None or check_no_grad(*mems)
      
        inputs = {'z': z}
        heads = tuple(heads) if heads is not None else ('z')

        outputs = self.prediction_net(
            inputs, tgt_length, stop_mask=d, heads=heads, mems=mems, return_attention=return_attention)
        out, h, mems, attention = outputs if return_attention else (outputs + (None,))

        
        preds = {}
        
        return (out['z'], h, mems, attention) if return_attention else (preds, h, mems)

    def compute_dynamics_loss(self, preds, h, gt_z):
        assert check_no_grad(gt_z)

        losses = []
        metrics = {}

        metrics['h_norm'] = h.norm(dim=-1, p=2).mean().detach()

        kl = self.kl_loss(preds,gt_z)
        print(kl.shape)
        mse = self.mse_loss(preds, gt_z)
        print(mse.shape)
        metrics['z_kl'] = kl
        metrics['z_mse'] = mse

        losses.append(kl)
        losses.append(mse)

        loss = sum(losses)
        metrics['dyn_loss'] = loss.detach()
        return loss, metrics
    

# Observational Model 
class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 64, 8, 8)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x
    
    
class BasicEncoder(nn.Module):
    """basic encoder as baseline
    
    Args:
        `do_flatten`: whether to flatten the input
        `embedding_size`: size of the embedding
    
    Inputs:
        `x`: image of shape [N, in_channels=1|3, 64, 64]
    Output:
        `output`: output of the encoder; shape: [N, embedding_size]"""
    def __init__(self, embedding_size=128, in_channels=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.in_channels = in_channels
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm() # Shape: (batch_size, 64, 6, 6)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64*6*6, embedding_size),  # Shape: [N, 6*6*64] -> [N, embedding_size]
            nn.ELU(),
            LayerNorm(),
        )
    def forward(self, x):
        x = self.conv_layer(x) # Shape: [N, 64, 6, 6]
        x = nn.Flatten(start_dim=1)(x) # Shape: [N, 64, 6, 6] -> [N, 64*6*6]
        x = self.mlp(x) # Shape: [N, 64*6*6] -> [N, embedding_size]
        return x.unsqueeze(1) # Shape: [N, 1, embedding_size]

class BasicDecoder(nn.Module):
    """Basic Upsampling Conv decoder that accepts concatenated hidden state vectors to decode an image
    
    Args:
        `embedding_size`: size of the embedding

    Inputs:
        `x`: hidden state of shape [N, embedding_size]

    Outputs:
        `output`: output of the decoder; shape: [N, 1, 64, 64]
    """
    def __init__(self, embedding_size=128, out_channels=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            nn.Sigmoid(),
            LayerNorm(),
            nn.Linear(self.embedding_size, 4096), # Shape: [N, embedding_size] -> [N, 4096]
            nn.ReLU(),
            LayerNorm(),
            UnFlatten(), # Shape: [N, 4096] -> [N, 64, 8, 8]
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, self.out_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() # Shape; [N, 1, 64, 64]
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
class ObservationModel(nn.Module):

    def __init__(self):
        super().__init__()
     
        self.z_dim = 128

        self.encoder, self.decoder = self.load_model()
        self.criterion = nn.MSELoss()



    def load_model(self, path="autoencoder.pth"):
        """
        Load the encoder and decoder model parameters.

        Args:
        - encoder (nn.Module): The encoder model.
        - decoder (nn.Module): The decoder model.
        - path (str): Path to load the model parameters from.

        Returns:
        - encoder (nn.Module): The encoder model with loaded parameters.
        - decoder (nn.Module): The decoder model with loaded parameters.
        """
        encoder = BasicEncoder()
        decoder = BasicDecoder()
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        return encoder, decoder

    def encode(self, o):
        assert check_no_grad(o)
        shape = o.shape[:2]
        print(shape)
        o = o.flatten(0, 1)
        z = self.encoder(o)
        z.unflatten(0, shape)
        z = z.reshape(*shape, *z.shape[2:])
        return z

    def decode(self, z):

        shape = z.shape[:2]
        z = z.flatten(0, 1)
        recons = self.decoder(z)
        recons = recons.unflatten(0, shape)
      
        return recons

    def compute_decoder_loss(self, recons, o):
        metrics = {}
        loss = self.criterion(recons, o)
        metrics['dec_loss'] = loss.detach()
        return loss, metrics


## World Model
class AdamOptim:

    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_clip=0):
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        self.optimizer.step()

def update_metrics(metrics, new_metrics, prefix=None):
    def process(key, t):
        if isinstance(t, (int, float)):
            return t
        assert torch.is_tensor(t), key
        assert not t.requires_grad, key
        assert t.ndim == 0 or t.shape == (1,), key
        return t.clone()

    if prefix is None:
        metrics.update({key: process(key, value) for key, value in new_metrics.items()})
    else:
        metrics.update({f'{prefix}{key}': process(key, value) for key, value in new_metrics.items()})
    return metrics

def combine_metrics(metrics, prefix=None):
    result = {}
    if prefix is None:
        for met in metrics:
            update_metrics(result, met)
    else:
        for met, pre in zip(metrics, prefix):
            update_metrics(result, met, pre)
    return result


class WorldModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_actions = 0

        self.obs_model = ObservationModel()
        self.dyn_model = DynamicsModel(self.obs_model.z_dim)

        self.obs_optimizer = AdamOptim(
            self.obs_model.parameters(), lr=config['obs_lr'], eps=config['obs_eps'], weight_decay=config['obs_wd'],
            grad_clip=config['obs_grad_clip'])
        self.dyn_optimizer = AdamOptim(
            self.dyn_model.parameters(), lr=config['dyn_lr'], eps=config['dyn_eps'], weight_decay=config['dyn_wd'],
            grad_clip=config['dyn_grad_clip'])

    @property
    def z_dim(self):
        return self.obs_model.z_dim

    @property
    def h_dim(self):
        return self.dyn_model.h_dim


    def optimize(self, o, mem=None):
   
        obs_model = self.obs_model
        dyn_model = self.dyn_model

        self.eval()
        with torch.no_grad():
           
            context_z = obs_model.encode(o[:, :1])
            next_z = obs_model.encode(o[:, -1:])
        
        
        if mem == None:
            mem = context_z

        self.train()

        # observation model
        o = o[:, 1:-1]
        z = obs_model.encode(o)
        print('z shape:', z.shape)
        recons = obs_model.decode(z)

        dec_loss, dec_met = obs_model.compute_decoder_loss(recons, o)
    
        # dynamics model
        z = z.detach()
        z = torch.cat([mem, z], dim=1)
        target_z = torch.cat([z[:, 1:].detach(), next_z.detach()], dim=1)
        tgt_length = target_z.shape[1]
        
        ones = torch.ones(tgt_length - 1)
        zeros = torch.zeros(tgt_length - 1)
        ones[-1] -= 1
        stop_mask = torch.logical_or(ones, zeros).unsqueeze(0).repeat(z.shape[0], 1)
        print(stop_mask)
        print(stop_mask.requires_grad)

        preds, h, mem = dyn_model.predict(z, stop_mask, tgt_length)
        #return  dyn_model.predict(z, stop_mask, tgt_length)

        dyn_loss, dyn_met = dyn_model.compute_dynamics_loss(
            preds, h, target_z=target_z)
        self.dyn_optimizer.step(dyn_loss)

        #z_hat = preds['z_hat'].detach()
        #con_loss, con_met = obs_model.compute_consistency_loss(z, z_hat)

        obs_loss = dec_loss #+ ent_loss + con_loss
        self.obs_optimizer.step(obs_loss)

        metrics = combine_metrics([dec_met, dyn_met])
        metrics['obs_loss'] = obs_loss.detach()

        return z, h, mem, metrics
    


if __name__ == "__main__":
    config = {
    # buffer
    'buffer_capacity': 100000,
    'buffer_temperature': 20.0,
    'buffer_prefill': 5000,

    # training
    'budget': 1000000000,
    'pretrain_budget': 50000000,
    'pretrain_obs_p': 0.6,
    'pretrain_dyn_p': 0.3,

    # evaluation
    'eval_every': 5000,
    'eval_episodes': 10,
    'final_eval_episodes': 100,

    # environment
    'env_frame_size': 64,
    'env_frame_skip': 4,
    'env_frame_stack': 4,
    'env_grayscale': True,
    'env_noop_max': 30,
    'env_time_limit': 27000,
    'env_episodic_lives': True,
    'env_reward_transform': 'tanh',
    'env_discount_factor': 0.99,
    'env_discount_lambda': 0.95,

    # world model
    'wm_batch_size': 100,
    'wm_sequence_length': 16,
    'wm_train_steps': 1,
    'wm_memory_length': 16,
    'wm_discount_threshold': 0.1,

    'z_categoricals': 32,
    'z_categories': 32,
    'obs_channels': 48,
    'obs_act': 'silu',
    'obs_norm': 'none',
    'obs_dropout': 0,
    'obs_lr': 1e-4,
    'obs_wd': 1e-6,
    'obs_eps': 1e-5,
    'obs_grad_clip': 100,
    'obs_entropy_coef': 5,
    'obs_entropy_threshold': 0.1,
    'obs_consistency_coef': 0.01,
    'obs_decoder_coef': 1,

    'dyn_embed_dim': 256,
    'dyn_num_heads': 4,
    'dyn_num_layers': 10,
    'dyn_feedforward_dim': 1024,
    'dyn_head_dim': 64,
    'dyn_z_dims': [512, 512, 512, 512],
    'dyn_reward_dims': [256, 256, 256, 256],
    'dyn_discount_dims': [256, 256, 256, 256],
    'dyn_input_rewards': True,
    'dyn_input_discounts': False,
    'dyn_act': 'silu',
    'dyn_norm': 'none',
    'dyn_dropout': 0.1,
    'dyn_lr': 1e-4,
    'dyn_wd': 1e-6,
    'dyn_eps': 1e-5,
    'dyn_grad_clip': 100,
    'dyn_z_coef': 1,
    'dyn_reward_coef': 10,
    'dyn_discount_coef': 50,

    # actor-critic
    'ac_batch_size': 400,
    'ac_horizon': 15,
    'ac_act': 'silu',
    'ac_norm': 'none',
    'ac_dropout': 0,
    'ac_input_h': False,
    'ac_h_norm': 'none',
    'ac_normalize_advantages': False,

    'actor_dims': [512, 512, 512, 512],
    'actor_lr': 1e-4,
    'actor_eps': 1e-5,
    'actor_wd': 1e-6,
    'actor_entropy_coef': 1e-2,
    'actor_entropy_threshold': 0.1,
    'actor_grad_clip': 1,

    'critic_dims': [512, 512, 512, 512],
    'critic_lr': 1e-5,
    'critic_eps': 1e-5,
    'critic_wd': 1e-6,
    'critic_grad_clip': 1,
    'critic_target_interval': 1
}


device = 'cuda:5'
# Example usage
file_path = 'mnist_test_seq.npy'
dataloader = get_dataloader(file_path)
for batch in dataloader:
    print(batch.shape)  # Expecting shape: (
    break
plot_sequence(batch[0], save_path="my_plot.png")

wm = WorldModel(config).to(device)

for epoch in range(10):
    for batch in dataloader:
        out = wm.optimize(batch.unsqueeze(2).to(device))
        break 
    break
