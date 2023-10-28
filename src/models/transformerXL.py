"""
Credits to https://github.com/jrobine/twm
"""

import copy
from functools import lru_cache
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .nets import * 
from .kv_caching import KeysValues, KVCache
from einops import rearrange
    
class TransformerXLDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, max_length, mem_length, batch_first=True):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.mem_length = mem_length
        self.batch_first = batch_first

        self.pos_enc = PositionalEncoding(decoder_layer.embed_dim, max_length, dropout_p=decoder_layer.dropout_p)
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

        if tgt_length is None:
            tgt_length = x.shape[0]
        assert tgt_length > 0
      
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

    def __init__(self, embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p, layer_norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.self_attn = RelativeMultiheadSelfAttention(embed_dim, head_dim, num_heads, dropout_p)
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
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

        src_length = k.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = q.view(tgt_length, batch_size, num_heads, head_dim)
        k = k.view(src_length, batch_size, num_heads, head_dim)
        v = v.view(src_length, batch_size, num_heads, head_dim)
        pos_encodings = pos_encodings.view(pos_len, num_heads, head_dim)

        content_score = torch.einsum('ibnd,jbnd->ijbn', (q + u_bias, k))
        pos_score = torch.einsum('ibnd,jnd->ijbn', (q + v_bias, pos_encodings))
        pos_score = self._rel_shift(pos_score)

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
        out = self.encodings[positions]
        out = self.dropout(out)
        return out.unsqueeze(0) if self.batch_first else out.unsqueeze(1)


class TransformerXL(nn.Module):

    def __init__(self, modality_order, num_current, embeds, embed_dim, activation, norm, dropout_p,
                 feedforward_dim, head_dim, num_heads, num_layers, memory_length, max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_length = memory_length
        self.modality_order = tuple(modality_order)
        self.num_current = num_current
        self.num_tokens = num_current
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        decoder_layer = TransformerXLDecoderLayer(
            embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p)

        num_modalities = len(modality_order)
        max_length = max_length * num_modalities + self.num_current
        mem_length = memory_length * num_modalities + self.num_current
        self.transformer = TransformerXLDecoder(decoder_layer, num_layers, max_length, mem_length, batch_first=True)


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
        #assert stop_mask.shape[1] * (self.num_tokens - 1) + (self.num_tokens - 1) == src_length

        src_mask = self._get_base_mask(src_length, tgt_length, device)

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

        idx = torch.logical_and(src, tgt)[:-1, :-1] # remove extra dimension 

        i, j, k = idx.shape 
        idx = idx.reshape(i, 1, j, 1, k).expand(i, (self.num_current), j, (self.num_current), k) \
            .reshape(i * (self.num_current), j * (self.num_current), k)

        src_mask = src_mask.unsqueeze(-1).tile(1, 1, batch_size)
        src_mask[idx] = True
        return src_mask
    
    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.transformer.u_bias.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.num_heads, max_tokens, self.embed_dim, self.num_layers, device)

    def forward(self, embeds, tgt_length, stop_mask, mems=None, return_attention=False):

        def cat_modalities(xs):
            batch_size, seq_len, num_tokens, dim = xs[0].shape  # xs[0] is z 
            xs[1] = xs[1].unsqueeze(2)
            m_xs = torch.cat(xs, dim=2)
            return m_xs.reshape(batch_size, seq_len * (num_tokens+1), dim)

        if 'a' not in embeds.keys():
            self.num_current= self.num_tokens - 1 # remove the action token 
        else:
            self.num_current = self.num_tokens
        
        if mems is None:
            history_length = embeds[self.modality_order[0]].shape[1] - 1  # assuming dim is (B, L, T, embed_dim (or encodings_dim if continuos Tf_XL))
            if 'a' not in embeds.keys():
                inputs = rearrange(embeds['z'], 'b l t e -> b (l t) e')
            elif self.num_tokens == self.num_current:
                inputs = cat_modalities([embeds[name] for name in self.modality_order]) # (B, L*num_modalities, dim) , modalities = [z, a], num_modalities = 2
            else:
                history = cat_modalities([embeds[name][:, :history_length] for name in self.modality_order]) # (B, L*num_modalities, dim) , modalities = [z, a], num_modalities = 2
                current = cat_modalities([embeds[name][:, history_length:] for name in self.modality_order[:self.num_current]])
                inputs = torch.cat([history, current], dim=1)
            tgt_length = (tgt_length - 1) * self.num_tokens + self.num_current 
            src_length = history_length * self.num_tokens + self.num_current
            assert inputs.shape[1] == src_length 
            src_mask = self._get_mask(src_length, src_length, inputs.device, stop_mask)
        else:
            sequence_length = embeds['z'].shape[1] # assuming dim is (B, L, embed_dim (or encodings_dim if continuos Tf_XL))
            # switch order so that 'currents' are last
            inputs = cat_modalities(
                [embeds[name] for name in (self.modality_order[self.num_current:] + self.modality_order[:self.num_current])])
            tgt_length = tgt_length * self.num_tokens
            mem_length = mems[0].shape[0]
            src_length = mem_length + sequence_length * self.num_tokens
            src_mask = self._get_mask(src_length, tgt_length, inputs.device, stop_mask) # (num_tokens * inner_sequence, num_tokens * inner_sequence, batch)

        positions = torch.arange(src_length - 1, -1, -1, device=inputs.device) # num_tokens * inner_sequence
        outputs = self.transformer(
            inputs, positions, attn_mask=src_mask, mems=mems, tgt_length=tgt_length, return_attention=return_attention)
        hiddens, mems, attention = outputs if return_attention else (outputs + (None,))

        # take outputs at last current
        assert hiddens.shape[1] == tgt_length
        #out_idx = torch.arange(tgt_length - 1, -1, - self.num_tokens, device=inputs.device).flip([0])
        #hiddens = hiddens[:, out_idx]
        #if return_attention:
        #    attention = attention[out_idx]

        #if heads is None:
        #    heads = self.out_heads.keys()

        #out = {name: self.out_heads[name](hiddens) for name in heads}

        return (hiddens, mems, attention) if return_attention else (hiddens, mems)
