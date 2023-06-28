"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import List
from math import sqrt
from typing import List, Optional, Union, Tuple
from torch import Tensor
import torch
import torch.nn as nn
from omegaconf import ListConfig


@dataclass
class EncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float


class Encoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.ch_mult)
        temb_ch = 0  # timestep embedding #channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(config.in_channels,
                                       config.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.config.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        temb = None  # timestep embedding

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
# class SpatialBroadcastDecoder(nn.Module):
#     def __init__(
#         self, config: EncoderDecoderConfig) -> None:
#         super().__init__()

#         params = SBDConfig()

#         self.conv_bone = []
#         self.config = config
#         self.params = params

#         self.num_resolutions = len(config.ch_mult)
#         input_channels = config.z_channels
#         width = height = config.resolution // 2 ** (self.num_resolutions - 1)
#         assert len(params.channels) == len(params.kernels) == len(params.strides) == len(params.paddings)
#         if params.conv_transposes:
#             assert len(params.channels) == len(params.output_paddings)
#         self.pos_embedding = PositionalEmbedding(width, height, input_channels)
#         self.width = width
#         self.height = height

#         self.conv_bone = self.make_sequential_from_config(
#             input_channels,
#             try_inplace_activation=True,
#         )



#     def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         x = self.pos_embedding(x)
#         output = self.conv_bone(x)
#         img, mask = output[:, :3], output[:, -1:]
#         return img, mask
    

#     def make_sequential_from_config(self,
#         input_channels: int,
#         return_params: bool = False,
#         try_inplace_activation: bool = True,
#     ) -> Union[nn.Sequential, Tuple[nn.Sequential, dict]]:
#         # Make copy of locals and expand scalars to lists
#         params = {k: v for k, v in locals().items()}
#         # params = self._scalars_to_list(params)

#         # Make sequential with the following order:
#         # - Conv or conv transpose
#         # - Optional batchnorm (optionally affine)
#         # - Optional activation
#         layers = []
#         layer_infos = zip(
#             self.params.channels,
#             self.params.batchnorms,
#             self.params.bn_affines,
#             self.params.kernels,
#             self.params.strides,
#             self.params.paddings,
#             self.params.activations,
#             self.params.conv_transposes,
#             self.params.output_paddings,
#         )
#         for (
#             channel,
#             bn,
#             bn_affine,
#             kernel,
#             stride,
#             padding,
#             activation,
#             conv_transpose,
#             o_padding,
#         ) in layer_infos:
#             if conv_transpose:
#                 layers.append(
#                     nn.ConvTranspose2d(
#                         input_channels, channel, kernel, stride, padding, o_padding
#                     )
#                 )
#             else:
#                 layers.append(nn.Conv2d(input_channels, channel, kernel, stride, padding))

#             if bn:
#                 layers.append(nn.BatchNorm2d(channel, affine=bn_affine))
#             if activation is not None:
#                 layers.append(
#                     self.get_activation_module(activation, try_inplace=try_inplace_activation)
#                 )

#             # Input for next layer has half the channels of the current layer if using GLU.
#             input_channels = channel
#             if activation == "glu":
#                 input_channels //= 2

#         if return_params:
#             return nn.Sequential(*layers), params
#         else:
#             return nn.Sequential(*layers)
        
#     def get_activation_module(self, activation_name: str, try_inplace: bool = True) -> nn.Module:
#         if activation_name == "leakyrelu":
#             act = torch.nn.LeakyReLU()
#         elif activation_name == "elu":
#             act = torch.nn.ELU()
#         elif activation_name == "relu":
#             act = torch.nn.ReLU(inplace=try_inplace)
#         elif activation_name == "glu":
#             act = torch.nn.GLU(dim=1)  # channel dimension in images
#         elif activation_name == "sigmoid":
#             act = torch.nn.Sigmoid()
#         elif activation_name == "tanh":
#             act = torch.nn.Tanh()
#         else:
#             raise ValueError(f"Unknown activation name '{activation_name}'")
#         return act
        
#     def _scalars_to_list(self, params: dict) -> dict:
#         # Channels must be a list
#         list_size = len(params["channels"])
#         # All these must be in `params` and should be expanded to list
#         allow_list = [
#             "kernels",
#             "batchnorms",
#             "bn_affines",
#             "paddings",
#             "strides",
#             "activations",
#             "output_paddings",
#             "conv_transposes",
#         ]
#         for k in allow_list:
#             if not isinstance(params[k], (tuple, list, ListConfig)):
#                 params[k] = [params[k]] * list_size
#         return params
    

    
class PositionalEmbedding(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x



class Decoder(nn.Module):
   def __init__(self, config: EncoderDecoderConfig) -> None:
       super().__init__()
       self.config = config
       temb_ch = 0
       self.num_resolutions = len(config.ch_mult)

       # compute in_ch_mult, block_in and curr_res at lowest res
       in_ch_mult = (1,) + tuple(config.ch_mult)
       block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
       curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
       print(f"Tokenizer : shape of latent is {config.z_channels, curr_res, curr_res}.")

       # z to block_in
       self.conv_in = torch.nn.Conv2d(config.z_channels,
                                      block_in,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

       # middle
       self.mid = nn.Module()
       self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                      out_channels=block_in,
                                      temb_channels=temb_ch,
                                      dropout=config.dropout)
       self.mid.attn_1 = AttnBlock(block_in)
       self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                      out_channels=block_in,
                                      temb_channels=temb_ch,
                                      dropout=config.dropout)

       # upsampling
       self.up = nn.ModuleList()
       for i_level in reversed(range(self.num_resolutions)):
           block = nn.ModuleList()
           attn = nn.ModuleList()
           block_out = config.ch * config.ch_mult[i_level]
           for i_block in range(config.num_res_blocks + 1):
               block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=temb_ch,
                                        dropout=config.dropout))
               block_in = block_out
               if curr_res in config.attn_resolutions:
                   attn.append(AttnBlock(block_in))
           up = nn.Module()
           up.block = block
           up.attn = attn
           if i_level != 0:
               up.upsample = Upsample(block_in, with_conv=True)
               curr_res = curr_res * 2
           self.up.insert(0, up)  # prepend to get consistent order

       # end
       self.norm_out = Normalize(block_in)
       self.conv_out = torch.nn.Conv2d(block_in,
                                       config.out_ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

   def forward(self, z: torch.Tensor) -> torch.Tensor:
       temb = None  # timestep embedding

       # z to block_in
       h = self.conv_in(z)

       # middle
       h = self.mid.block_1(h, temb)
       h = self.mid.attn_1(h)
       h = self.mid.block_2(h, temb)

       # upsampling
       for i_level in reversed(range(self.num_resolutions)):
           for i_block in range(self.config.num_res_blocks + 1):
               h = self.up[i_level].block[i_block](h, temb)
               if len(self.up[i_level].attn) > 0:
                   h = self.up[i_level].attn[i_block](h)
           if i_level != 0:
               h = self.up[i_level].upsample(h)

       # end
       h = self.norm_out(h)
       h = nonlinearity(h)
       h = self.conv_out(h)
       return h


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = None, conv_shortcut: bool = False,
                 dropout: float, temb_channels: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)      # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)        # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

@dataclass
class SAEncoderConfig:
    channels: List[int] = (32, 32, 32, 512)
    kernels: List[int] = (5, 5, 5, 3)
    strides: List[int] = (1, 1, 1, 1)
    paddings: List[int] = (2, 2, 2, 1)

@dataclass
class SBDecoderConfig:
    resolution: int
    num_slots: int
    hidden_dim: int

@dataclass
class SAConfig:
    resolution: int
    num_slots: int
    iters: int
    hidden_dim: int

import numpy as np
def build_grid(resolution):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Integer specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid.to(inputs.device))
        return inputs + grid

class SAEncoder(nn.Module):
    def __init__(self, resolution, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.encoder_pos = SoftPositionEmbed(hidden_dim, resolution)

    def forward(self, x):
        x = self.layers(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x
    
class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, config: SBDecoderConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        resolution = config.resolution
        self.config = config

        if hidden_dim == 64:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, 4, 3, stride=(1, 1), padding=1),
            )
            self.decoder_initial_size = (8, 8)
        elif hidden_dim == 32:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, 4, 3, stride=(1, 1), padding=1),
            )
            self.decoder_initial_size = (64, 64)
        self.decoder_pos = SoftPositionEmbed(hidden_dim, self.decoder_initial_size)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution

    def forward(self, slots):
        bs = slots.shape[0]
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, self.decoder_initial_size[0], self.decoder_initial_size[1], 1))

        x = self.decoder_pos(slots)
        x = x.permute(0,3,1,2)
        x = self.layers(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)

        # Undo combination of slot and batch dimension; split alpha masks.
        colors, masks = x.reshape(bs, -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `colors` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        rec = torch.sum(colors * masks, dim=1)  # Recombine image.
        rec = rec.permute(0,3,1,2)
        # `rec` has shape: [batch_size, width, height, num_channels].
        return rec

class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters=3, eps=1e-8, hidden_dim=128) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.rand(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim, eps=0.001)
        self.norm_slots = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)
        self.dim = dim

    def forward(self, inputs: Tensor, num_slots: Optional[int] = None) -> Tensor:
        b, n = inputs.shape[:2]
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim)
            )

            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

class SlotAttention(nn.Module):
    def __init__(self, config: SAConfig) -> None:
        """Builds the Slot Attention-based auto-encoder.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.resolution = config.resolution
        self.num_slots = config.num_slots
        self.iters = config.iters

        self.encoder_cnn = SAEncoder(config.resolution, config.hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.slot_attention = SlotAttentionModule(
            num_slots=config.num_slots,
            dim=config.hidden_dim,
            iters=config.iters,
            eps=1e-8, 
            hidden_dim=128)
        
    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(x.device)(x)
        x = self.fc(x) # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return slots