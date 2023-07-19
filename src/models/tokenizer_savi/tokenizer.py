"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .lpips import LPIPS
from utils import LossWithIntermediateLosses
from .steve import STEVEEncoder, STEVEDecoder, dVAE, SlotAttentionVideo, gumbel_softmax, linear


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, seed: int = 0, batch_size: int = 24, num_workers: int = 4, image_size: int = 128, img_channels: int = 3, 
             ep_len: int = 3, checkpoint_path: str = 'checkpoint.pt.tar', data_path: str = 'data/*', log_path: str = 'logs/', 
             lr_dvae: float = 0.0003, lr_enc: float = 0.0001, lr_dec: float = 0.0003, lr_warmup_steps: int = 30000, 
             lr_half_life: int = 250000, clip: float = 0.05, epochs: int = 500, steps: int = 200000, num_iterations: int = 2, 
             num_slots: int = 15, cnn_hidden_size: int = 64, slot_size: int = 192, mlp_hidden_size: int = 192, 
             num_predictor_blocks: int = 1, num_predictor_heads: int = 4, predictor_dropout: float = 0.0, vocab_size: int = 4096, 
             num_decoder_blocks: int = 8, num_decoder_heads: int = 4, d_model: int = 192, dropout: float = 0.1, tau_start: float = 1.0, 
             tau_final: float = 0.1, tau_steps: int = 30000, hard: bool = False, use_dp: bool = True, with_lpips: bool = True, 
             embed_dim: int = 256, dim_slot: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = STEVEEncoder(img_channels,
                                    cnn_hidden_size,
                                    image_size,
                                    d_model,
                                    num_iterations,
                                    num_slots,
                                    slot_size,
                                    mlp_hidden_size,
                                    num_predictor_blocks,
                                    num_predictor_heads,
                                    predictor_dropout)
        
        self.decoder = STEVEDecoder(d_model,
                                    image_size,
                                    num_decoder_blocks,
                                    num_decoder_heads,
                                    dropout)
        self.slot_attn = SlotAttentionVideo()
        self.dvae = dVAE(vocab_size, img_channels)
        self.slot_proj = linear(slot_size, d_model, bias=False)

        #self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        #self.embedding = nn.Embedding(vocab_size, embed_dim)
        #self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        #self.decoder = decoder
        #self.slot_attn = SA(num_slots, embed_dim, dim_slot)
        #self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.img_channels = img_channels
        self.ep_len = ep_len
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.log_path = log_path
        self.lr_dvae = lr_dvae
        self.lr_enc = lr_enc
        self.lr_dec = lr_dec
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_half_life = lr_half_life
        self.clip = clip
        self.epochs = epochs
        self.steps = steps
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.cnn_hidden_size = cnn_hidden_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_predictor_blocks = num_predictor_blocks
        self.num_predictor_heads = num_predictor_heads
        self.predictor_dropout = predictor_dropout
        self.vocab_size = vocab_size
        self.num_decoder_blocks = num_decoder_blocks
        self.num_decoder_heads = num_decoder_heads
        self.d_model = d_model
        self.dropout = dropout
        self.tau_start = tau_start
        self.tau_final = tau_final
        self.tau_steps = tau_steps
        self.hard = hard
        self.use_dp = use_dp
        self.with_lpips = with_lpips
        self.embed_dim = embed_dim
        self.dim_slot = dim_slot
      

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        z_hard, z_soft = self.dvae(x)
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_hard, reconstructions, dvae_reconstructions = self(observations, should_preprocess=False, should_postprocess=False) #TODO: need at adjust forward()

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        # beta = 1.0
        # commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        # reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        b = observations.shape[0]
        dvae_mse = ((observations - dvae_reconstructions) ** 2).sum() / b
        cross_entropy = -(z_hard * torch.log_softmax(reconstructions, dim=-1)).sum() / b

        return LossWithIntermediateLosses(commitment_loss=cross_entropy, reconstruction_loss=dvae_mse)

    def dvae(self, x: torch.Tensor, tau: float, hard: bool = False) -> Tuple[torch.Tensor]:
        shape = x.shape # (..., C, H, W)
        z_soft, z_hard = self.dvae(x, tau, hard)
        b, e, h, w = z_hard.shape
        z_hard = rearrange(z_hard, 'b e h w -> b (h w) e', b=b, e=e, h=h, w=w)          # B * T, H_enc * W_enc, vocab_size
        z_emb = self.decoder.dict(z_hard)                                               # B * T, H_enc * W_enc, d_model
        z_emb = torch.cat([self.decoder.bos.expand(shape[0], -1, -1), z_emb], dim=1)    # B * T, 1 + H_enc * W_enc, d_model
        z_emb = self.decoder.pos(z_emb)

        return z_hard, z_soft, z_emb

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z_soft, z_hard, z_emb = self.dvae(x) #TODO: tau, hard
        z = self.encoder(x)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> b (h w) e')
        z, _ = self.slot_attn(z_flattened)

        # dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())

        # tokens = dist_to_embeddings.argmin(dim=-1)
        # z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=self.num_slots, w=int(h*w/self.num_slots)).contiguous()

        # Reshape to original
        # z = z.reshape(*shape[:-3], *z_q.shape[1:])
        # z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        # tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, ..., ...) #TODO:

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        #z_q = self.post_quant_conv(z_q)
        img_slots, masks = self.decoder(z_q)
        img_slots = img_slots.view(z_q.shape[0], self.num_slots*z_q.shape[3], 3, self.width, self.height)
        masks = masks.view(z_q.shape[0], self.num_slots, z_q.shape[3], 1, self.width, self.height)
        masks = masks.softmax(dim=1)
        masks = masks.view(z_q.shape[0], self.num_slots*z_q.shape[3], 1, self.width, self.height)
        recon_slots_masked = img_slots * masks
        rec = recon_slots_masked.sum(dim=1)
        #rec = self.decoder(z_q)
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
