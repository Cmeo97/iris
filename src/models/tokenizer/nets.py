"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import List
from math import sqrt
from typing import List, Optional, Union, Tuple
from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import ListConfig
from utils import LossWithIntermediateLosses

from .transformer_utils import *



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
    dec_input_dim: int # SBDecoder
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
    prior_class: str
    pred_prior_from: str

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
        self._init_params()

    def _init_params(self):
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv_bone(x)
        out = self.pos_embedding(conv_output)
        out = out.flatten(2, 3) # bs x c x (w * h)
        out = self.lnorm(out)
        out = self.conv_1x1(out)
        return out


@dataclass
class SBDConfig:
    resolution: int
    dec_input_dim: int # SBDecoder
    dec_hidden_dim: int # SBDecoder
    out_ch: int


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, config: SBDConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.dec_hidden_dim
        resolution = config.resolution

        if hidden_dim == 64:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(config.dec_input_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                # nn.ReLU(inplace=True),
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
                # nn.ConvTranspose2d(config.dec_input_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.Conv2d(config.dec_input_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.Conv2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.Conv2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, config.out_ch, 3, stride=(1, 1), padding=1),
                nn.Conv2d(hidden_dim, config.out_ch, 3, stride=(1, 1), padding=1),
            )
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        # self.init_resolution = resolution if hidden_dim == 32 else (8, 8)
        self.init_resolution = resolution if hidden_dim == 32 else (28, 28)
        self.pos_embedding = PositionalEmbedding(self.init_resolution, config.dec_input_dim)
        self.resolution = resolution
        self._init_params()
        print('Decoder initialized')

    def _init_params(self):
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)

    def __repr__(self) -> str:
        return "image_decoder"
    
    def forward(self, x: torch.Tensor, return_indiv_slots=False) -> torch.Tensor:
        bs = x.shape[0]
        K = x.shape[2] * x.shape[3]
        x = self.spatial_broadcast(x.permute(0,2,3,1))
        x = self.pos_embedding(x)
        x = self.layers(x)

        # Undo combination of slot and batch dimension; split alpha masks.
        colors, masks = x[:, :3], x[:, -1:]
        colors = colors.reshape(bs, K, 3, self.resolution[0], self.resolution[1])
        masks = masks.reshape(bs, K, 1, self.resolution[0], self.resolution[1])
        masks = masks.softmax(dim=1)
        rec = (colors * masks).sum(dim=1)

        if return_indiv_slots:
            return rec, colors, masks

        return rec

    def spatial_broadcast(self, slot: torch.Tensor) -> torch.Tensor:
        slot = slot.reshape(-1, slot.shape[-1])
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.init_resolution[0], self.init_resolution[1])
    
    def compute_loss(self, batch, z, **kwargs) -> LossWithIntermediateLosses:
        observations = batch['observations']
        shape = observations.shape

        reconstructions = self(rearrange(z, 'b t k d -> (b t) d k').unsqueeze(-1))
        reconstructions = rearrange(reconstructions, '(b t) c h w -> b t c h w', b=shape[0])

        loss = torch.pow(observations - reconstructions, 2).mean()

        return LossWithIntermediateLosses(reconstruction_loss=loss)
    
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)


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
        # self.slots_mu = nn.Linear(config.channels_enc, config.slot_dim)
        # self.slots_log_sigma = nn.Linear(config.channels_enc, config.slot_dim)
        # self.slots_mu2 = nn.Linear(4096, config.num_slots)
        # self.slots_log_sigma2 = nn.Linear(4096, config.num_slots)

        self.to_q = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.to_k = nn.Linear(config.channels_enc, config.slot_dim, bias=False)
        self.to_v = nn.Linear(config.channels_enc, config.slot_dim, bias=False)

        self.gru = nn.GRUCell(config.slot_dim, config.slot_dim)

        hidden_dim = max(config.slot_dim, hidden_dim)

        # self.mlp = nn.Sequential(
        #     nn.Linear(config.slot_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, config.slot_dim),
        # )
        self.mlp = nn.Sequential(
            nn.Linear(config.slot_dim, config.slot_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(config.slot_dim*4, config.slot_dim),
        )

        self.norm_input = nn.LayerNorm(config.channels_enc)
        self.norm_slots = nn.LayerNorm(config.slot_dim)
        self.norm_pre_ff = nn.LayerNorm(config.slot_dim)
        self.slot_dim = config.slot_dim
        
        self._init_params()

        self.is_video = False

    def _init_params(self):
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                torch.nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)
        torch.nn.init.zeros_(self.gru.bias_ih)
        torch.nn.init.zeros_(self.gru.bias_hh)
        torch.nn.init.orthogonal_(self.gru.weight_hh)

    def forward(self, inputs: torch.Tensor, num_slots: Optional[int] = None) -> torch.Tensor:
        b, n, d = inputs.shape
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        # mu = self.slots_mu(inputs)
        # sigma = self.slots_log_sigma(inputs)
        # mu = self.slots_mu2(mu.permute(0, 2, 1)).permute(0, 2, 1)
        # sigma = self.slots_log_sigma2(sigma.permute(0, 2, 1)).permute(0, 2, 1).exp()
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


class SlotAttentionVideo(nn.Module):
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
            nn.Linear(config.slot_dim, config.slot_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(config.slot_dim*4, config.slot_dim),
        )

        if config.prior_class.lower() == 'mlp':
            self.prior = nn.Sequential(
                nn.Linear(config.slot_dim, config.slot_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.slot_dim, config.slot_dim),
            )
        elif config.prior_class.lower() == 'gru':
            self.prior = nn.GRU(config.slot_dim, config.slot_dim, batch_first=True)
        elif config.prior_class.lower() == 'grucell':
            self.prior = nn.GRUCell(config.slot_dim, config.slot_dim)
        elif config.prior_class.lower() == 'none' or config.prior_class.lower() == 'tf':
            self.prior = None
        else:
            raise NotImplementedError("prior class not implemented")
        self.prior_class = config.prior_class
        self.pred_prior_from = config.pred_prior_from

        self.predictor = TransformerEncoder(num_blocks=1, d_model=config.slot_dim, num_heads=4, dropout=0.1)

        self.norm_input = nn.LayerNorm(config.channels_enc)
        self.norm_slots = nn.LayerNorm(config.slot_dim)
        self.norm_pre_ff = nn.LayerNorm(config.slot_dim)
        self.slot_dim = config.slot_dim
        
        self._init_params()

        self.is_video = True

    def _init_params(self):
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                torch.nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)
        torch.nn.init.zeros_(self.gru.bias_ih)
        torch.nn.init.zeros_(self.gru.bias_hh)
        torch.nn.init.orthogonal_(self.gru.weight_hh)

    def forward(self, inputs: torch.Tensor, num_slots: Optional[int] = None) -> torch.Tensor:
        assert len(inputs.shape) == 4
        b, T, n, d = inputs.shape
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k *= self.scale

        slots_list = []
        dots_list = []
        hidden = None
        for t in range(T):
            slots_init = slots

            for i in range(self.iters):
                slots_prev = slots

                slots = self.norm_slots(slots)
                q = self.to_q(slots)
                dots = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn = dots.softmax(dim=-1) + self.eps

                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])

                slots = self.gru(updates.view(-1, self.slot_dim),
                                 slots_prev.view(-1, self.slot_dim))
                slots = slots.view(-1, self.num_slots, self.slot_dim)

                # use MLP only when more than one iterations
                if i < self.iters - 1:
                    slots = slots + self.mlp(self.norm_pre_ff(slots))
                
            slots_list += [slots]
            dots_list += [dots.softmax(dim=-1)]

            # if prior_class = tf and pred_prior_from = last, STEVE
            slots_to_predict_from = slots_init if self.pred_prior_from.lower() == 'init' else slots
            if self.prior_class.lower() == 'mlp':
                slots = self.prior(slots_to_predict_from)
            elif self.prior_class.lower() == 'gru':
                slots, hidden = self.prior(slots_to_predict_from, hidden)
            elif self.prior_class.lower() == 'grucell':
                slots = self.prior(slots_to_predict_from.view(-1, self.slot_dim), hidden)
                slots = slots.view(-1, self.num_slots, self.slot_dim)
            elif self.prior_class.lower() == 'tf':
                slots = self.predictor(slots_to_predict_from)
            elif self.prior_class.lower() == 'none':
                pass

        slots_list = torch.stack(slots_list, dim=1)   # B, T, num_slots, slot_size
        dots_list = torch.stack(dots_list, dim=1)

        return slots_list, dots_list
    

def resize_patches_to_image(patches: torch.Tensor, size: Optional[int] = None, 
                            scale_factor: Optional[float] = None, resize_mode: str = "bilinear") -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = sqrt(n_patches)
    patch_size = int(sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


@dataclass
class OCEncoderDecoderWithViTConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float
    vit_model_name: str
    vit_use_pretrained: bool
    vit_freeze: bool
    vit_feature_level: Union[int, str, List[Union[int, str]]]
    vit_num_patches: int
    dec_input_dim: int # MLPDecoder
    dec_hidden_layers: List[int] # MLPDecoder
    dec_output_dim: int # MLPDecoder
    

class MLPDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.dec_input_dim = config.dec_input_dim
        self.dec_hidden_layers = config.dec_hidden_layers
        self.dec_output_dim = config.dec_output_dim

        self.vit_num_patches = config.vit_num_patches
        
        layers = []
        current_dim = config.dec_input_dim
    
        for dec_hidden_dim in config.dec_hidden_layers:
            layers.append(nn.Linear(current_dim, dec_hidden_dim))
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU(inplace=True))
            current_dim = dec_hidden_dim

        layers.append(nn.Linear(current_dim, config.dec_output_dim + 1))
        nn.init.zeros_(layers[-1].bias)
        
        self.layers = nn.Sequential(*layers)

        self.pos_embed = nn.Parameter(torch.randn(1, config.vit_num_patches, config.dec_input_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z (bt, k, d)
        init_shape = z.shape[:-1]
        z = z.flatten(0, -2)
        z = z.unsqueeze(1).expand(-1, self.vit_num_patches, -1)

        # Simple learned additive embedding as in ViT
        z = z + self.pos_embed
        out = self.layers(z)
        out = out.unflatten(0, init_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.dec_output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        masks_as_image = resize_patches_to_image(masks, size=self.config.resolution, resize_mode="bilinear")

        return reconstruction, masks, masks_as_image

