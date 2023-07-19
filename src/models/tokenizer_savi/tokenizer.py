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
from utils import LossWithIntermediateLosses, cosine_anneal
from .steve import dVAE


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor

@dataclass
class TokenizerConfig:
    img_channels: int = 3
    vocab_size: int = 4096
    tau_start: float = 1.0
    tau_final: float = 0.1
    tau_steps: int = 30000
    hard: bool = False


class Tokenizer(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        self.dVAE = dVAE(configs.vocab_size, configs.img_channels)
       
        self.tau_start = configs.tau_start
        self.tau_final = configs.tau_final
        self.tau_steps = configs.tau_steps
        self.hard = configs.hard
      
      

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, tau: float = 0.1) -> Tuple[torch.Tensor]:
        #B, T, C, H, W = x.shape
        #z_hard, z_soft, z_emb = self.dvae(x, tau, self.hard)
        #dvae_recon = self.dVAE.decoder(z_soft).reshape(B, T, C, H, W) 
        #dvae_mse = ((x - dvae_recon) ** 2).sum() / (B * T)  
        outputs = self.encode(x, should_preprocess, tau)
        #decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(outputs.z, should_postprocess, x.shape)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, global_step: int = 0 , **kwargs: Any) -> LossWithIntermediateLosses:
        tau = cosine_anneal(global_step, self.tau_start, self.tau_final, 0, self.tau_steps)
        
        observations = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')
        z, z_hard, dvae_reconstructions = self(observations, tau=tau)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        # beta = 1.0
        # commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        # reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        b = observations.shape[0]
        dvae_mse = ((observations - dvae_reconstructions) ** 2).sum() / b
        #cross_entropy = -(z_hard * torch.log_softmax(reconstructions, dim=-1)).sum() / b

        return LossWithIntermediateLosses(reconstruction_loss=dvae_mse)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False, tau: float = 0.1) -> TokenizerEncoderOutput:
        #if should_preprocess:
        #    x = self.preprocess_input(x)
        #shape = x.shape  # (..., C, H, W)
        #x = x.view(-1, *shape[-3:])
        
        #B, T, C, H, W = x.size()
        x = x.flatten(end_dim=1)
        z, z_q, tokens = self.dVAE(x, tau, self.hard) 
        #z = self.encoder(x)
        #b, e, h, w = z.shape
        #z_flattened = rearrange(z, 'b e h w -> b (h w) e')
        #z, _ = self.slot_attn(z_flattened)

        # dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())

        # tokens = dist_to_embeddings.argmin(dim=-1)
        # z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=self.num_slots, w=int(h*w/self.num_slots)).contiguous()

        # Reshape to original
        # z = z.reshape(*shape[:-3], *z_q.shape[1:])
        # z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        # tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z: torch.Tensor, should_postprocess: bool = False, shape: Tuple = ()) -> torch.Tensor:
        B, T, C, H, W = shape
        dvae_recon = self.dVAE.decoder(z).reshape(B, T, C, H, W)
        #shape = z_q.shape  # (..., E, h, w)
        #z_q = z_q.view(-1, *shape[-3:])
        ##z_q = self.post_quant_conv(z_q)
        #img_slots, masks = self.decoder(z_q)
        #img_slots = img_slots.view(z_q.shape[0], self.num_slots*z_q.shape[3], 3, self.width, self.height)
        #masks = masks.view(z_q.shape[0], self.num_slots, z_q.shape[3], 1, self.width, self.height)
        #masks = masks.softmax(dim=1)
        #masks = masks.view(z_q.shape[0], self.num_slots*z_q.shape[3], 1, self.width, self.height)
        #recon_slots_masked = img_slots * masks
        #rec = recon_slots_masked.sum(dim=1)
        ##rec = self.decoder(z_q)
        #rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        #if should_postprocess:
        #    rec = self.postprocess_output(rec)
        return dvae_recon

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z = self.encode(x, should_preprocess).z
        return self.decode(z, should_postprocess, x.shape)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
