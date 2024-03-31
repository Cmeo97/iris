"""
Credits to https://github.com/CompVis/taming-transformers
"""

import enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from einops import rearrange
import math
import torch
import torch.nn as nn
import timm

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder, SlotAttention, SpatialBroadcastDecoder, SAEncoder, SlotAttentionVideo, MLPDecoder
from .quantizer import *
from utils import LossWithIntermediateLosses
import torch.nn.functional as F

@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True, consistency_loss_reg: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None
        self.slot_based = False
        self.consistency_loss_reg = consistency_loss_reg


    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        b, t, _, _, _ = batch['observations'].shape
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        
        if self.consistency_loss_reg:
            tokens_consistency_loss = 0.1 * self.compute_tokens_consistency_loss(z_quantized.reshape(b,t,*z_quantized.shape[1:]))
        else:
            tokens_consistency_loss = 0 * commitment_loss

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss, tokens_consistency_loss=tokens_consistency_loss)

    def compute_tokens_consistency_loss(self, z_q: Any, **kwargs: Any) -> Any:
        
        b, l, e, c,_ = z_q.shape
        z_q_next = z_q[:, 1:].reshape(b*(l-1), e, -1)
        z_q_current = z_q[:, :-1].reshape(b*(l-1), e, -1).transpose(1,2)
        mat = torch.bmm(z_q_current, z_q_next)
        mat = F.softmax(mat, dim=-1)
        cosine_distance = F.cross_entropy(mat, torch.arange(c**2, device=mat.device).expand(mat.shape[0], -1))/(z_q_current.shape[1])
        
        return cosine_distance

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec


    @torch.no_grad()
    def get_post_zq(self, z_q: torch.Tensor) -> torch.Tensor:
        shape = z_q.shape
        z_q = z_q.reshape(-1, *shape[-3:])
        h = self.post_quant_conv(z_q)
        return h

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)


class OCContinuousTokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Union[Encoder, SAEncoder], decoder: SpatialBroadcastDecoder, slot_attn: SlotAttention, with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.slot_attn = slot_attn
        self.quantizer = SkipVQ(slot_attn.config.num_slots, slot_attn.config.tokens_per_slot)

        self.lpips = LPIPS().eval() if with_lpips else None
        self.width = encoder.config.resolution
        self.height = encoder.config.resolution
        self.num_slots = slot_attn.config.num_slots
        self.tokens_per_slot = slot_attn.config.tokens_per_slot
        self.slot_based = True

    def __repr__(self) -> str:
        return "tokenizer"
    
    def set_tau(self):
        pass

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        reconstruction_loss = torch.pow(observations - reconstructions, 2).mean()
        return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        b, e, c = z.shape
        z = rearrange(z, 'b e c -> b c e')
        z = self.slot_attn(z)
        
        z = rearrange(z, 'b k (t e) -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot)
        
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])

        return TokenizerEncoderOutput(z, None, None)

    def decode(self, z: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z.shape  # (..., E, h, w)
        z = z.view(-1, *shape[-3:])
        rec = self.decoder(z)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
    
    def decode_slots(self, z: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z.shape  # (..., E, h, w)
        z = z.view(-1, *shape[-3:])
        rec, color, mask = self.decoder(z, return_indiv_slots=True)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        color = color.reshape(*shape[:-3], *color.shape[1:])
        mask = mask.reshape(*shape[:-3], *mask.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec, color, mask

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z = self.encode(x, should_preprocess).z
        return self.decode(z, should_postprocess)
    
    @torch.no_grad()
    def encode_decode_slots(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, use_hard: bool = False) -> torch.Tensor:
        z = self.encode(x, should_preprocess).z
        if use_hard:
            pass
        return self.decode_slots(z, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)


class OCTokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Union[Encoder, SAEncoder], decoder: SpatialBroadcastDecoder, slot_attn: SlotAttention, with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = None #nn.Linear(slot_attn.config.token_dim, embed_dim) if slot_attn.config.token_dim != embed_dim else None
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = None #nn.Linear(embed_dim, decoder.config.dec_input_dim) if embed_dim != decoder.config.dec_input_dim else None
        self.decoder = decoder
        self.slot_attn = slot_attn
        # self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # self.quantizer = SkipVQ(slot_attn.config.num_slots, slot_attn.config.tokens_per_slot)
        # self.quantizer = BaseVQ(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = VQEMA(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = VQNearestEmbEMA(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = VQEMAwCodeReset(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = VmfVQ(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = PerceiverVQ(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = VQwithReparameterization(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = SlicedVQ(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        # self.quantizer = BinderQuantization(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
        self.quantizer = SlicedPerceiverVQ(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)

        self.lpips = LPIPS().eval() if with_lpips else None
        self.width = encoder.config.resolution
        self.height = encoder.config.resolution
        self.num_slots = slot_attn.config.num_slots
        self.tokens_per_slot = slot_attn.config.tokens_per_slot
        self.slot_based = True

        self.global_step = 0
        self.tau = 0.0
        self.tau_init = 1.0
        self.tau_final = 0.1
        self.tau_start_step = 0
        self.tau_final_step = 50000

    def __repr__(self) -> str:
        return "tokenizer"
    
    def set_tau(self): # cosine annealing
        # if self.global_step < self.tau_start_step:
        #     value = self.tau_init
        # elif self.global_step >= self.tau_final_step:
        #     value = self.tau_final
        # else:
        #     a = 0.5 * (self.tau_init - self.tau_final)
        #     b = 0.5 * (self.tau_init + self.tau_final)
        #     progress = (self.global_step - self.tau_start_step) / (self.tau_final_step - self.tau_start_step)
        #     value = a * math.cos(math.pi * progress) + b
        
        # self.tau = value
        # self.global_step += 1
        pass

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach() * self.tau
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        commitment_loss = self.quantizer.compute_loss(z, z_quantized) 
        if commitment_loss is not None:
            commitment_loss *= self.tau
        reconstruction_loss = torch.pow(observations - reconstructions, 2).mean()
        if commitment_loss is None:
            return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss)
        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        b, e, c = z.shape
        z = rearrange(z, 'b e c -> b c e')
        z = self.slot_attn(z)
        if self.pre_quant_conv is not None:
            z = self.pre_quant_conv(z)
        z_flattened = rearrange(z, 'b k (t e) -> (b k t) e', t=self.tokens_per_slot)

        tokens, z_q = self.quantizer(z_flattened, self.tau)
        z_q = rearrange(z_q, '(b k t) e -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot).contiguous()

        z = rearrange(z, 'b k (t e) -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot)
        
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        if self.post_quant_conv is not None:
            z_q = rearrange(z_q, 'b e k t -> b k t e', k=self.num_slots, t=self.tokens_per_slot)
            z_q = self.post_quant_conv(z_q)
            z_q = rearrange(z_q, 'b k t e -> b e k t', k=self.num_slots, t=self.tokens_per_slot)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
    
    def decode_slots(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        if self.post_quant_conv is not None:
            z_q = rearrange(z_q, 'b e k t -> b k t e', k=self.num_slots, t=self.tokens_per_slot)
            z_q = self.post_quant_conv(z_q)
            z_q = rearrange(z_q, 'b k t e -> b e k t', k=self.num_slots, t=self.tokens_per_slot)
        rec, color, mask = self.decoder(z_q, return_indiv_slots=True)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        color = color.reshape(*shape[:-3], *color.shape[1:])
        mask = mask.reshape(*shape[:-3], *mask.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec, color, mask

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)
    
    @torch.no_grad()
    def encode_decode_slots(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, use_hard: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z
        if use_hard:
            # z_q = self.encode(x, should_preprocess).z_quantized
            z_q = self.quantizer.get_embedding(self.encode(x, should_preprocess).tokens)
            z_q = rearrange(z_q, 'b (k t) e -> b e k t', k=self.num_slots, t=self.tokens_per_slot)
        return self.decode_slots(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
    

@dataclass
class TokenizerSeparateEncoderOutput:
    z: torch.FloatTensor
    z_target: torch.FloatTensor
    z_quantized: torch.FloatTensor
    z_quantized_hard: torch.FloatTensor
    tokens: torch.LongTensor

class OCTokenizerSeparate(OCTokenizer):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Union[Encoder, SAEncoder], decoder: SpatialBroadcastDecoder, slot_attn: SlotAttention, with_lpips: bool = True) -> None:
        super().__init__(vocab_size, embed_dim, encoder, decoder, slot_attn, with_lpips)
        self.quantizer = dVAE(vocab_size, embed_dim, slot_attn.config.num_slots, slot_attn.config.tokens_per_slot, beta=0.25)
    
        self.global_step = 0
        self.tau = 1.0
        self.tau_init = 1.0
        self.tau_final = 0.1
        self.tau_start_step = 0
        self.tau_final_step = 50000

    def __repr__(self) -> str:
        return "tokenizer"
    
    def set_tau(self): # cosine annealing
        if self.global_step < self.tau_start_step:
            value = self.tau_init
        elif self.global_step >= self.tau_final_step:
            value = self.tau_final
        else:
            a = 0.5 * (self.tau_init - self.tau_final)
            b = 0.5 * (self.tau_init + self.tau_final)
            progress = (self.global_step - self.tau_start_step) / (self.tau_final_step - self.tau_start_step)
            value = a * math.cos(math.pi * progress) + b
        
        self.tau = value
        self.global_step += 1

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z_target, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z_target, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        reconstructions_from_quantized = self.decode(z_quantized, should_postprocess=False)
        commitment_loss = self.quantizer.compute_loss(z_target, z_quantized) + torch.pow(observations - reconstructions_from_quantized, 2).mean() * self.quantizer.beta
        reconstruction_loss = torch.pow(observations - reconstructions, 2).mean()
        if commitment_loss is None:
            return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss)
        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        # if self.pre_quant_conv is not None:
        #     z = self.pre_quant_conv(z)
        b, e, c = z.shape
        z = rearrange(z, 'b e c -> b c e')
        z = self.slot_attn(z)

        z_targ = z.detach()
        z_targ = rearrange(z_targ, 'b k (t e) -> (b k t) e', t=self.tokens_per_slot)
        tokens, z_q, z_hard = self.quantizer(z_targ, self.tau)

        z = rearrange(z, 'b k (t e) -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot)
        z_targ = rearrange(z_targ, '(b k t) e -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot)
        z_q = rearrange(z_q, '(b k t) e -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot).contiguous()
        z_hard = rearrange(z_hard, '(b k t) e -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot).contiguous()
        
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_targ = z_targ.reshape(*shape[:-3], *z_targ.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        z_hard = z_hard.reshape(*shape[:-3], *z_hard.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerSeparateEncoderOutput(z, z_targ, z_q, z_hard, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        if self.post_quant_conv is not None:
            z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
    
    def decode_slots(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        if self.post_quant_conv is not None:
            z_q = self.post_quant_conv(z_q)
        rec, color, mask = self.decoder(z_q, return_indiv_slots=True)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        color = color.reshape(*shape[:-3], *color.shape[1:])
        mask = mask.reshape(*shape[:-3], *mask.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec, color, mask

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized #TODO: z_quantized or z?
        return self.decode(z_q, should_postprocess)
    
    @torch.no_grad()
    def encode_decode_slots(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, use_hard: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized #TODO: z_quantized or z?
        if use_hard:
            z_q = self.encode(x, should_preprocess).z_quantized_hard
        return self.decode_slots(z_q, should_postprocess)

    def encode_logits(self, z):
        bs = z.shape[0]
        z = rearrange(z.detach(), 'b e k t -> (b k t) e')
        z_logits = F.log_softmax(self.quantizer.pre_vq_linear(z), dim=1)
        z_logits = rearrange(z_logits, '(b k) e -> b k e', b=bs)
        return z_logits

    def decode_logits(self, logits):
        z_soft = self.quantizer.gumbel_softmax(logits, self.tau, False, dim=1)
        z_q = self.quantizer.post_vq_linear(z_soft)
        return z_q
    
    def decode_tokens(self, tokens):
        z_hard = F.one_hot(tokens, num_classes=self.vocab_size).float()
        z_q = self.quantizer.post_vq_linear(z_hard)
        return z_q

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)


@dataclass
class TokenizerWithPosEncoderOutput:
    z: torch.FloatTensor
    pos: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor

class OCTokenizerWithSeparatePos(OCTokenizer):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Union[Encoder, SAEncoder], decoder: SpatialBroadcastDecoder, slot_attn: SlotAttention, with_lpips: bool = True) -> None:
        super().__init__(vocab_size, embed_dim, encoder, decoder, slot_attn, with_lpips)

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach() * self.tau
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        commitment_loss = self.quantizer.compute_loss(z, z_quantized) * self.tau
        reconstruction_loss = torch.pow(observations - reconstructions, 2).mean()
        if commitment_loss is None:
            return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss)
        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        b, e, c = z.shape
        z = rearrange(z, 'b e c -> b c e')
        z, pos = self.slot_attn(z)
        if self.pre_quant_conv is not None:
            z = self.pre_quant_conv(z)
        z_flattened = rearrange(z, 'b k (t e) -> (b k t) e', t=self.tokens_per_slot)

        tokens, z_q = self.quantizer(z_flattened, self.tau)
        z_q = rearrange(z_q, '(b k t) e -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot).contiguous()

        z = rearrange(z, 'b k (t e) -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot)
        pos = rearrange(pos, 'b k (t e) -> b e k t', b=b, k=self.num_slots, t=self.tokens_per_slot)
        
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        pos = pos.reshape(*shape[:-3], *pos.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerWithPosEncoderOutput(z, pos, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        if self.post_quant_conv is not None:
            z_q = rearrange(z_q, 'b e k t -> b k t e', k=self.num_slots, t=self.tokens_per_slot)
            z_q = self.post_quant_conv(z_q)
            z_q = rearrange(z_q, 'b k t e -> b e k t', k=self.num_slots, t=self.tokens_per_slot)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
    
    def decode_slots(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        if self.post_quant_conv is not None:
            z_q = rearrange(z_q, 'b e k t -> b k t e', k=self.num_slots, t=self.tokens_per_slot)
            z_q = self.post_quant_conv(z_q)
            z_q = rearrange(z_q, 'b k t e -> b e k t', k=self.num_slots, t=self.tokens_per_slot)
        rec, color, mask = self.decoder(z_q, return_indiv_slots=True)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        color = color.reshape(*shape[:-3], *color.shape[1:])
        mask = mask.reshape(*shape[:-3], *mask.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec, color, mask

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)
    
    @torch.no_grad()
    def encode_decode_slots(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, use_hard: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z
        if use_hard:
            z_q = self.encode(x, should_preprocess).z_quantized
            # z_q = self.quantizer.get_embedding(self.encode(x, should_preprocess).tokens)
            # z_q = rearrange(z_q, 'b (k t) e -> b e k t', k=self.num_slots, t=self.tokens_per_slot)
        return self.decode_slots(z_q, should_postprocess)


class _VitFeatureType(enum.Enum):
    BLOCK = 1
    KEY = 2
    VALUE = 3
    QUERY = 4
    CLS = 5


class _VitFeatureHook:
    """Auxilliary class used to extract features from timm ViT models."""

    def __init__(self, feature_type: _VitFeatureType, block: int, drop_cls_token: bool = True):
        """Initialize VitFeatureHook.

        Args:
            feature_type: Type of feature to extract.
            block: Number of block to extract features from. Note that this is not zero-indexed.
            drop_cls_token: Drop the cls token from the features. This assumes the cls token to
                be the first token of the sequence.
        """
        assert isinstance(feature_type, _VitFeatureType)
        self.feature_type = feature_type
        self.block = block
        self.drop_cls_token = drop_cls_token
        self.name = f"{feature_type.name.lower()}{block}"
        self.remove_handle = None  # Can be used to remove this hook from the model again

        self._features = None

    @staticmethod
    def create_hook_from_feature_level(feature_level: Union[int, str]):
        feature_level = str(feature_level)
        prefixes = ("key", "query", "value", "block", "cls")
        for prefix in prefixes:
            if feature_level.startswith(prefix):
                _, _, block = feature_level.partition(prefix)
                feature_type = _VitFeatureType[prefix.upper()]
                block = int(block)
                break
        else:
            feature_type = _VitFeatureType.BLOCK
            try:
                block = int(feature_level)
            except ValueError:
                raise ValueError(f"Can not interpret feature_level '{feature_level}'.")

        return _VitFeatureHook(feature_type, block)

    def register_with(self, model):
        supported_models = (
            timm.models.vision_transformer.VisionTransformer,
            timm.models.beit.Beit,
            timm.models.vision_transformer_sam.VisionTransformerSAM,
        )
        model_names = ["vit", "beit", "samvit"]

        if not isinstance(model, supported_models):
            raise ValueError(
                f"This hook only supports classes {', '.join(str(cl) for cl in supported_models)}."
            )

        if self.block > len(model.blocks):
            raise ValueError(
                f"Trying to extract features of block {self.block}, but model only has "
                f"{len(model.blocks)} blocks"
            )

        block = model.blocks[self.block - 1]
        if self.feature_type == _VitFeatureType.BLOCK:
            self.remove_handle = block.register_forward_hook(self)
        else:
            if isinstance(block, timm.models.vision_transformer.ParallelBlock):
                raise ValueError(
                    f"ViT with `ParallelBlock` not supported for {self.feature_type} extraction."
                )
            elif isinstance(model, timm.models.beit.Beit):
                raise ValueError(f"BEIT not supported for {self.feature_type} extraction.")
            self.remove_handle = block.attn.qkv.register_forward_hook(self)

        model_name_map = dict(zip(supported_models, model_names))
        self.model_name = model_name_map.get(type(model), None)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp):
        if self.feature_type == _VitFeatureType.BLOCK:
            features = outp
            if self.drop_cls_token:
                # First token is CLS token.
                if self.model_name == "samvit":
                    # reshape outp (B,H,W,C) -> (B,H*W,C)
                    features = outp.flatten(1,2)
                else:
                    features = features[:, 1:]
        elif self.feature_type in {
            _VitFeatureType.KEY,
            _VitFeatureType.QUERY,
            _VitFeatureType.VALUE,
        }:
            # This part is adapted from the timm implementation. Unfortunately, there is no more
            # elegant way to access keys, values, or queries.
            B, N, C = inp[0].shape
            qkv = outp.reshape(B, N, 3, C)  # outp has shape B, N, 3 * H * (C // H)
            q, k, v = qkv.unbind(2)

            if self.feature_type == _VitFeatureType.QUERY:
                features = q
            elif self.feature_type == _VitFeatureType.KEY:
                features = k
            else:
                features = v
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type == _VitFeatureType.CLS:
            # We ignore self.drop_cls_token in this case as it doesn't make any sense.
            features = outp[:, 0]  # Only get class token.
        else:
            raise ValueError("Invalid VitFeatureType provided.")

        self._features = features


@dataclass
class TokenizerWithSAMEncoderOutput:
    z: torch.FloatTensor
    z_vit: torch.FloatTensor
    attns: torch.FloatTensor
    tokens: torch.FloatTensor

class OCSAMTokenizer(OCTokenizer):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Union[Encoder, SAEncoder], decoder: MLPDecoder, slot_attn: Union[SlotAttention, SlotAttentionVideo], const_type: str, const_coef: float, with_lpips: bool = True) -> None:
        super().__init__(vocab_size, embed_dim, encoder, decoder, slot_attn, with_lpips)
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.slot_attn = slot_attn
        self.vit_model_name = encoder.config.vit_model_name
        self.vit_use_pretrained = encoder.config.vit_use_pretrained
        self.vit_freeze = encoder.config.vit_freeze
        self.vit_feature_level = encoder.config.vit_feature_level

        self._init_vit()
        self._init_pos_embed(encoder.config.dec_output_dim, slot_attn.config.token_dim)

        self.width = encoder.config.resolution
        self.height = encoder.config.resolution
        self.num_slots = slot_attn.config.num_slots
        self.tokens_per_slot = slot_attn.config.tokens_per_slot
        self.slot_based = True

        self.const_type = const_type
        self.const_coef = const_coef

    def __repr__(self) -> str:
        return "tokenizer"
    
    def _init_vit(self):
        def feature_level_to_list(feature_level):
            if feature_level is None:
                return []
            elif isinstance(feature_level, (int, str)):
                return [feature_level]
            else:
                return list(feature_level)

        self.feature_levels = feature_level_to_list(self.vit_feature_level)

        model = timm.create_model(self.vit_model_name, pretrained=self.vit_use_pretrained)
        # Delete unused parameters from classification head
        if hasattr(model, "head"):
            del model.head
        if hasattr(model, "fc_norm"):
            del model.fc_norm

        if len(self.feature_levels) > 0:
            self._feature_hooks = [
                _VitFeatureHook.create_hook_from_feature_level(level).register_with(model) for level in self.feature_levels
            ]
            feature_dim = model.num_features * len(self.feature_levels)

            # Remove modules not needed in computation of features
            max_block = max(hook.block for hook in self._feature_hooks)
            new_blocks = model.blocks[:max_block]  # Creates a copy
            del model.blocks
            model.blocks = new_blocks
            model.norm = nn.Identity()

        self.vit = model
        self._feature_dim = feature_dim

        if self.vit_freeze:
            self.vit.requires_grad_(False)
            # BatchNorm layers update their statistics in train mode. This is probably not desired
            # when the model is supposed to be frozen.
            contains_bn = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                for m in self.vit.modules()
            )
            self.run_in_eval_mode = contains_bn
        else:
            self.run_in_eval_mode = False

    def _init_pos_embed(self, encoder_output_dim, token_dim):
        layers = []
        layers.append(nn.LayerNorm(encoder_output_dim))
        layers.append(nn.Linear(encoder_output_dim, encoder_output_dim))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(encoder_output_dim, token_dim))
        nn.init.zeros_(layers[-1].bias)
        self.pos_embed = nn.Sequential(*layers)
    
    def set_tau(self): # cosine annealing
        pass

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.attns, outputs.z_vit, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        if self.slot_attn.is_video:
            observations = rearrange(observations, '(b t) c h w -> b t c h w', b=batch['observations'].shape[0]) # video
        z, attns, z_vit, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        reconstruction_loss = torch.pow(z_vit - reconstructions, 2).mean()

        # cosine similarity loss between consecutive frames
        if not self.slot_attn.is_video:
            z = rearrange(z, '(b t) k e -> b t k e', b=batch['observations'].shape[0])
        cosine_loss = 0.
        if self.const_type == "repr":
            for t in range(z.shape[1]-1):
                z_curr = z[:, t]
                z_next = z[:, t+1]
                z_curr = z_curr / z_curr.norm(dim=-1, keepdim=True)
                z_next = z_next / z_next.norm(dim=-1, keepdim=True)
                # matrix of cosine similarities between all pairs of slots for each sample in batch
                mat = torch.bmm(z_curr, z_next.transpose(1, 2))
                # softmax of mat
                mat = F.softmax(mat, dim=-1)
                # cross entropy loss between mat and identity matrix
                cosine_loss += F.cross_entropy(mat, torch.arange(self.num_slots, device=mat.device).expand(mat.shape[0], -1))
            cosine_loss /= (z.shape[1]-1)
        elif self.const_type == "attn":
            B, T, _, num_slots = attns.shape
            for t in range(attns.shape[1]-1):
                attn_curr = attns[:, t]
                attn_next = attns[:, t+1]
                attn_curr = attn_curr / attn_curr.norm(dim=-2, keepdim=True)
                attn_next = attn_next / attn_next.norm(dim=-2, keepdim=True)
                pairwise_sim = torch.bmm(attn_curr.transpose(1, 2), attn_next)
                loss = torch.pow(torch.diagonal(pairwise_sim, dim1=-2, dim2=-1) - torch.ones(num_slots, device=pairwise_sim.device).expand(B, -1), 2).mean()
                cosine_loss += loss
            cosine_loss /= (attns.shape[1]-1)

        return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss, cosine_loss=cosine_loss * self.const_coef)
    
    def _transformer_compute_positions(self, features):
        """Compute positions for Transformer features."""
        n_tokens = features.shape[1]
        image_size = math.sqrt(n_tokens)
        image_size_int = int(image_size)
        assert (
            image_size_int == image_size
        ), "Position computation for Transformers requires square image"

        spatial_dims = (image_size_int, image_size_int)
        positions = torch.cartesian_prod(
            *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
        )
        return positions

    def vit_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.run_in_eval_mode and self.training:
            self.eval()

        if self.vit_freeze:
            # Speed things up a bit by not requiring grad computation.
            with torch.no_grad():
                features = self.vit.forward_features(x)
        else:
            features = self.vit.forward_features(x)

        if self._feature_hooks is not None:
            hook_features = [hook.pop() for hook in self._feature_hooks]

        if len(self.feature_levels) == 0:
            # Remove class token when not using hooks.
            features = features[:, 1:]
            positions = self._transformer_compute_positions(features)
        else:
            features = hook_features[: len(self.feature_levels)]
            positions = self._transformer_compute_positions(features[0])
            features = torch.cat(features, dim=-1)

        return features, positions

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        # z = self.encoder(x)
        # b, e, c = z.shape
        # z = rearrange(z, 'b e c -> b c e')
        z, _ = self.vit_encode(x)
        z = self.pos_embed(z)

        if self.slot_attn.is_video:
            z = z.reshape(*shape[:-3], *z.shape[1:]) # video
            if z.dim() == 3:
                z = z.unsqueeze(1)
        z, attns = self.slot_attn(z)
        if self.slot_attn.is_video:
            z = z.view(-1, *z.shape[-2:]) # video

        z_vit, _ = self.vit_encode(x)
        
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_vit = z_vit.reshape(*shape[:-3], *z_vit.shape[1:])
        tokens = z

        return TokenizerWithSAMEncoderOutput(z, z_vit, attns, tokens)

    def decode(self, z: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z.shape  # (..., C, D)
        z = z.view(-1, *shape[-2:])
        rec, _, _ = self.decoder(z)
        rec = rec.reshape(*shape[:-2], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
    
    def decode_slots(self, z: torch.Tensor, x: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z.shape  # (..., C, D)
        z = z.view(-1, *shape[-2:])
        rec, masks, masks_as_image = self.decoder(z)
        rec = rec.reshape(*shape[:-2], *rec.shape[1:])
        masks = masks.reshape(*shape[:-2], *masks.shape[1:])
        masks_as_image = masks_as_image.reshape(*shape[:-2], *masks_as_image.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)

        colors = x.unsqueeze(-4).expand(-1, self.num_slots, -1, -1, -1) if len(x.shape) == 4 else x.unsqueeze(-4).expand(-1, -1, self.num_slots, -1, -1, -1)
      
        return x, colors, masks_as_image.unsqueeze(-3)
    
    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        # z_q = self.encode(x, should_preprocess).z_quantized
        # return self.decode(z_q, should_postprocess)
        return x
    
    @torch.no_grad()
    def encode_decode_slots(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, use_hard: bool = False) -> torch.Tensor:
        z = self.encode(x, should_preprocess).z
        return self.decode_slots(z, x, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)