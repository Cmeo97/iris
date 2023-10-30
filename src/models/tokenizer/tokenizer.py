"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple, Union

from einops import rearrange
import math
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder, SlotAttention, SpatialBroadcastDecoder, SAEncoder, SlotAttentionSeparate
from .quantizer import *
from utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
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

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

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
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

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
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Union[Encoder, SAEncoder], decoder: SpatialBroadcastDecoder, slot_attn: SlotAttentionSeparate, with_lpips: bool = True) -> None:
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
