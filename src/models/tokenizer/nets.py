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

@dataclass
class OCEncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float
    dec_hidden_dim: int # SBDecoder

class Encoder(nn.Module):
    def __init__(self, config: Union[EncoderDecoderConfig, OCEncoderDecoderConfig]) -> None:
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
    def __init__(self, resolution: List[int], channels: int):
        super().__init__()
        height, width = resolution
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
class SAConfig:
    num_slots: int
    tokens_per_slot: int
    iters: int
    channels_enc: int
    token_dim: int

    @property
    def slot_dim(self):
        return self.tokens_per_slot * self.token_dim

class SAEncoder(nn.Module):
    def __init__(self, config: OCEncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        resolution = config.resolution

        self.conv_bone = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, config.z_channels, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.pos_embedding = PositionalEmbedding(resolution, config.z_channels)
        self.lnorm = nn.GroupNorm(1, config.z_channels, affine=True, eps=0.001)
        self.conv_1x1 = nn.Sequential(
            nn.Conv1d(config.z_channels, config.z_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(config.z_channels, config.z_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv_bone(x)
        out = self.pos_embedding(conv_output)
        out = out.flatten(2, 3) # bs x c x (w * h)
        out = self.lnorm(out)
        out = self.conv_1x1(out)
        out = conv_output.reshape(conv_output.shape) # bs x c x w x h
        return out
    
class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, config: OCEncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.dec_hidden_dim
        resolution = config.resolution

        if hidden_dim == 64:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(config.z_channels, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, config.out_ch, 3, stride=(1, 1), padding=1),
            )
        elif hidden_dim == 32:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(config.z_channels, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, config.out_ch, 3, stride=(1, 1), padding=1),
            )
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.pos_embedding = PositionalEmbedding(resolution, config.z_channels)
        self.resolution = resolution
        print('Decoder initialized')

    def forward(self, x: torch.Tensor, return_indiv_slots=False) -> torch.Tensor:
        x = self.spatial_broadcast(x.permute(0,2,3,1))
        bs, K, nts, td, w, h = x.shape #K: num slots, nts: num tokens per slots, td: token_dim
        x = self.pos_embedding(x.reshape(-1, td, w, h))
        x = self.layers(x)

        # Undo combination of slot and batch dimension; split alpha masks.
        colors, masks = x[:, :3], x[:, -1:]
        colors = colors.reshape(bs, -1, 3, self.resolution[0], self.resolution[1])
        masks = masks.reshape(bs, K, -1, 1, self.resolution[0], self.resolution[1])
        masks = masks.softmax(dim=1)
        masks = masks.reshape(bs, -1, 1, self.resolution[0],  self.resolution[1])
        rec = (colors * masks).sum(dim=1)

        if return_indiv_slots:
            return rec, colors, masks

        return rec

    def spatial_broadcast(self, slot: torch.Tensor) -> torch.Tensor:
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, 1, 1, self.resolution[0], self.resolution[1])

class SlotAttention(nn.Module):
    def __init__(self, config: SAConfig, eps=1e-8, hidden_dim=128) -> None:
        super().__init__()
        assert config.slot_dim % config.tokens_per_slot == 0
        self.config = config
        self.num_slots = config.num_slots
        self.tokens_per_slot = config.tokens_per_slot
        self.iters = config.iters
        self.eps = eps
        self.scale = config.slot_dim**-0.5

        self.slots_mu = nn.Parameter(torch.rand(1, 1, config.slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, config.slot_dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + config.slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        self.to_q = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.to_k = nn.Linear(config.channels_enc, config.slot_dim, bias=False)
        self.to_v = nn.Linear(config.channels_enc, config.slot_dim, bias=False)

        self.gru = nn.GRUCell(config.slot_dim, config.slot_dim)

        hidden_dim = max(config.slot_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(config.slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, config.slot_dim),
        )

        self.norm_input = nn.LayerNorm(config.channels_enc, eps=0.001)
        self.norm_slots = nn.LayerNorm(config.slot_dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(config.slot_dim, eps=0.001)
        self.slot_dim = config.slot_dim

    def forward(self, inputs: torch.Tensor, num_slots: Optional[int] = None) -> torch.Tensor:
        b, n, d = inputs.shape
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
                updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim)
            )

            slots = slots.reshape(b, -1, self.slot_dim)
            slots = (slots + self.mlp(self.norm_pre_ff(slots)))

        return slots

# class SlotAttention(nn.Module):
#     def __init__(self, config) -> None:
#         """Builds the Slot Attention-based auto-encoder.
#         """
#         super().__init__()
#         self.config = config
#         self.hidden_dim = config.hidden_dim
#         self.resolution = config.resolution
#         self.num_slots = config.num_slots
#         self.iters = config.iters

#         self.encoder_cnn = Encoder(config)

#         self.fc = nn.Sequential(
#             nn.Linear(config.hidden_dim, config.hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(config.hidden_dim, config.hidden_dim),
#         )

#         self.slot_attention = SlotAttentionModule(
#             num_slots=config.num_slots,
#             dim=config.hidden_dim,
#             iters=config.iters,
#             eps=1e-8, 
#             hidden_dim=128)
        
#     def forward(self, image):
#         # `image` has shape: [batch_size, num_channels, width, height].

#         # Convolutional encoder with position embedding.
#         x = self.encoder_cnn(image)  # CNN Backbone.
#         x = nn.LayerNorm(x.shape[1:]).to(x.device)(x)
#         x = self.fc(x) # Feedforward network on set.
#         # `x` has shape: [batch_size, width*height, input_size].

#         # Slot Attention module.
#         slots = self.slot_attention(x)
#         # `slots` has shape: [batch_size, num_slots, slot_size].

#         return slots