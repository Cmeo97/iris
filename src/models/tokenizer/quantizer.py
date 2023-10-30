from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SkipVQ(nn.Module): # For SlotAttention
    def __init__(self, num_slots, tokens_per_slot):
        super().__init__()

        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.save_after = 25
        self.pca = None

    def compute_loss(self, z, z_quantized):
        return None

    def forward(self, z, tau=None):
        self.slot_repr = z.detach()
        return z, z

    def plot_count(self, epoch, save_dir):
        pass

    def plot_slot_dist(self, epoch, save_dir):
        if epoch >= self.save_after and self.slot_repr is not None:
            slot_repr = rearrange(self.slot_repr, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot)
            dist = torch.cdist(slot_repr, slot_repr)[0]
            fig, ax = plt.subplots()
            im = ax.imshow(dist.cpu().numpy())
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(save_dir / f"slot_dist_{epoch}.png")
            plt.close()

    @torch.no_grad()
    def plot_codebook(self, epoch, save_dir):
        if epoch >= self.save_after:
            z = self.slot_repr.cpu().numpy()
            if self.pca is None:
                self.pca = PCA(n_components=2)
                self.pca.fit(z)
            z_pca = self.pca.transform(z)

            plt.scatter(z_pca[:, 0], z_pca[:, 1])
            plt.savefig(save_dir / f"slot_repr_{epoch}.png")
            plt.close()


class dVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta

        self.pre_vq_linear = nn.Linear(embed_dim, vocab_size)
        self.post_vq_linear = nn.Linear(vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.ref_count = torch.zeros(vocab_size)
        self.ref_count_log = []
        self.save_after = 25
        self.save_count_every = 25
        self.pca = None

    def gumbel_softmax(self, logits, tau=1., hard=False, dim=-1):
        gumbels = -(torch.empty_like(logits).exp() + 1e-8).log()
        gumbels = (logits + gumbels) / tau
        
        y_soft = F.softmax(gumbels, dim)
        
        if hard:
            index = y_soft.argmax(dim, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
        
    def compute_loss(self, z, z_quantized):
        return torch.pow(z - z_quantized, 2).mean() * self.beta

    def forward(self, z, tau, hard=False):
        z_logits = F.log_softmax(self.pre_vq_linear(z), dim=1) # don't need log?
        z_soft = self.gumbel_softmax(z_logits, tau, hard, dim=1)
        z_hard = self.gumbel_softmax(z_logits, tau, True, dim=1)
        z_q = self.post_vq_linear(z_soft)
        tokens = z_hard.argmax(dim=-1)

        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z_logits.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_targ = z.detach()
        self.slot_repr = z_q.detach()
        return tokens, z_q, self.post_vq_linear(z_hard)

    def _reset_count(self):
        self.ref_count = torch.zeros(self.vocab_size)
        self.ref_count_log = []
    
    def plot_count(self, epoch, save_dir):
        self.ref_count_log.append(self.ref_count.numpy())
        if epoch >= self.save_after and epoch % self.save_count_every == 0:
            plt.figure(figsize=(10,1))
            plt.imshow(self.ref_count_log)
            plt.tight_layout()
            plt.savefig(save_dir / f"ref_count_{epoch}.png")
            plt.close()
            self._reset_count()

    def plot_slot_dist(self, epoch, save_dir):
        if epoch >= self.save_after and self.slot_repr is not None:
            slot_targ = rearrange(self.slot_targ, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot)
            slot_repr = rearrange(self.slot_repr, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot)
            dist = torch.cdist(slot_targ, slot_targ)[0]
            fig, ax = plt.subplots()
            im = ax.imshow(dist.cpu().numpy())
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(save_dir / f"slot_targ_dist_{epoch}.png")
            plt.close()

            dist = torch.cdist(slot_repr, slot_repr)[0]
            fig, ax = plt.subplots()
            im = ax.imshow(dist.cpu().numpy())
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(save_dir / f"slot_dist_{epoch}.png")
            plt.close()

    @torch.no_grad()
    def plot_codebook(self, epoch, save_dir):
        if epoch >= self.save_after:
            z = self.slot_repr.cpu().numpy()
            if self.pca is None:
                self.pca = PCA(n_components=2)
                self.pca.fit(z)
            z_pca = self.pca.transform(z)

            plt.scatter(z_pca[:, 0], z_pca[:, 1])
            plt.savefig(save_dir / f"slot_repr_{epoch}.png")
            plt.close()


class BaseVQ(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # self.embedding.weight.data.normal_()

        # limit = np.sqrt(6.0 / (1 + tokens_per_slot * embed_dim))
        # mu = torch.rand((vocab_size, embed_dim)).uniform_(-limit, limit)
        # log_sigma = torch.rand((vocab_size, embed_dim)).uniform_(-limit, limit).exp()
        # init_codes = torch.normal(mu, log_sigma)
        # self.embedding.weight.data.copy_(init_codes)

        self.pre_quant_conv = nn.Linear(64, embed_dim)
        self.post_quant_conv = nn.Linear(embed_dim, 64)

        self.ref_count = torch.zeros(vocab_size*tokens_per_slot)
        self.ref_count_log = []
        self.save_count_every = 25
        self.save_after = 25
        self.slot_repr = None
        self.pca = None

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + self.beta * (z - z_quantized.detach()).pow(2).mean()

        return commitment_loss

    def forward(self, z, tau=None):
        z = self.pre_quant_conv(z)

        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        # probabilities = F.softmax(-dist_to_embeddings / tau, dim=-1)
        # tokens = probabilities.argmax(dim=-1)
        z_q = self.embedding(tokens)

        z_q = self.post_quant_conv(z_q)

        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q

    def get_embedding(self, tokens):
        # shape of tokens: (b k)
        z_q = self.embedding(tokens)
        z_q = self.post_quant_conv(z_q)

        return z_q

    def _reset_count(self):
        self.ref_count = torch.zeros(self.vocab_size)
        self.ref_count_log = []

    def plot_count(self, epoch, save_dir):
        self.ref_count_log.append(self.ref_count.numpy())
        if epoch >= self.save_after and epoch % self.save_count_every == 0:
            plt.figure(figsize=(10,1))
            plt.imshow(self.ref_count_log)
            plt.tight_layout()
            plt.savefig(save_dir / f"ref_count_{epoch}.png")
            plt.close()
            self._reset_count()

    def plot_slot_dist(self, epoch, save_dir):
        if epoch >= self.save_after and epoch % self.save_count_every == 0 and self.slot_repr is not None:
            dist = torch.cdist(self.slot_repr, self.slot_repr)[0]
            fig, ax = plt.subplots()
            im = ax.imshow(dist.cpu().numpy())
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(save_dir / f"slot_dist_{epoch}.png")
            plt.close()

    @torch.no_grad()
    def plot_codebook(self, epoch, save_dir):
        if epoch >= self.save_after:
            self.slot_repr = rearrange(self.slot_repr, 'b (k t) e -> (b k t) e', k=self.num_slots, t=self.tokens_per_slot)
            dist_to_embeddings = torch.sum(self.slot_repr ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(self.slot_repr, self.embedding.weight.t())
            tokens = dist_to_embeddings.argmin(dim=-1)
            z_q = self.embedding(tokens).cpu().numpy()

            emb = self.embedding.weight.detach().cpu().numpy()
            if self.pca is None:
                self.pca = PCA(n_components=2)
                self.pca.fit(emb)
            emb_pca = self.pca.transform(emb)
            z_pca = self.pca.transform(z_q)

            plt.scatter(emb_pca[:, 0], emb_pca[:, 1])
            plt.scatter(z_pca[:, 0], z_pca[:, 1])
            plt.savefig(save_dir / f"codebook_{epoch}.png")
            plt.close()

            fig = sns.displot(dist_to_embeddings.detach().cpu().numpy().T, kind='kde')
            fig._legend.remove()
            plt.savefig(save_dir / f"dist_{epoch}.png")
            plt.close()


class VQEMA(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.register_buffer('cluster_size', torch.zeros(vocab_size))
        self.register_buffer('ema_embed', torch.Tensor(vocab_size, embed_dim))
        self.ema_embed.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # self.ema_embed.data.normal_()
        self.decay = 0.99
        self.eps = 1e-5

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = self.beta * (z - z_quantized.detach()).pow(2).mean() # EMA

        return commitment_loss

    def ema(self, z, tokens):
        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z.requires_grad: # if training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()
            
            self.cluster_size.mul_(self.decay).add_(ref_count, alpha=1-self.decay)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.cluster_size.data)
            smoothing_cluster_size = (self.cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n

            dw = z.T @ tokens_onehot
            self.ema_embed.data.mul_(self.decay).add_(dw.T, alpha=1-self.decay)
            self.embedding.weight.data.copy_(self.ema_embed / smoothing_cluster_size.unsqueeze(1))

    def forward(self, z):
        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = self.embedding(tokens)

        if z.requires_grad:
            self.ema(z, tokens)

        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q

class VQNearestEmbEMA(VQEMA):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)

        self.register_buffer('prev_cluster', torch.zeros(vocab_size))
        self.resample_every = 20 # original paper uses 200
        self.seen_batches = 0

    def ema(self, z, tokens):
        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z.requires_grad:
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()
            self.prev_cluster.data.add_(ref_count)
            
            self.cluster_size.mul_(self.decay).add_(ref_count, alpha=1-self.decay)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.cluster_size.data)
            smoothing_cluster_size = (self.cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n

            dw = z.T @ tokens_onehot
            self.ema_embed.data.mul_(self.decay).add_(dw.T, alpha=1-self.decay)
            self.embedding.weight.data.copy_(self.ema_embed / smoothing_cluster_size.unsqueeze(1))

        self.seen_batches += 1

    @torch.no_grad()
    def code_resampling(self, z_q):
        updated = 0
        if self.seen_batches % self.resample_every == 0:
            for idx, eq in enumerate(self.prev_cluster):
                if eq == 0:
                    dist = torch.norm(z_q - self.embedding.weight[idx], dim=1)
                    probs = dist / (torch.sum(dist, dim=0, keepdim=True) + 1e-6)
                    if probs.sum() == 0:
                        break
                    rand_idx = torch.multinomial(probs, 1)
                    self.embedding.weight.data[idx].copy_(z_q[rand_idx].squeeze())
                    self.ema_embed.data[idx].copy_(z_q[rand_idx].squeeze())

                    updated += 1

            self.prev_cluster.data.mul_(0.)
            self.seen_batches = 0

    def forward(self, z):
        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = self.embedding(tokens)

        if z.requires_grad:
            self.ema(z, tokens)
            self.code_resampling(z_q)

        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q


class VQEMAwCodeReset(VQEMA):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)

        self.register_buffer('prev_cluster', torch.zeros(vocab_size))
        self.resample_every = 20
        self.seen_batches = 0

    def ema(self, z, tokens):
        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z.requires_grad:
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()
            self.prev_cluster.data.add_(ref_count)
            
            self.cluster_size.mul_(self.decay).add_(ref_count, alpha=1-self.decay)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.cluster_size.data)
            smoothing_cluster_size = (self.cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n

            dw = z.T @ tokens_onehot
            self.ema_embed.data.mul_(self.decay).add_(dw.T, alpha=1-self.decay)
            self.embedding.weight.data.copy_(self.ema_embed / smoothing_cluster_size.unsqueeze(1))

        self.seen_batches += 1

    @torch.no_grad()
    def codebook_reset(self):
        if self.seen_batches % self.resample_every == 0:
            max_count, most_used_idx = torch.max(self.prev_cluster, dim=0)
            most_used_code = self.embedding.weight.data[most_used_idx]
            frac_usage = self.prev_cluster / max_count
            min_frac_usage, min_usage_idx = torch.min(frac_usage, dim=0)
            if min_frac_usage < 0.03:
                moved_code = most_used_code + torch.randn_like(most_used_code) / 100
                self.embedding.weight.data[min_usage_idx].copy_(moved_code)
            self.prev_cluster.data.mul_(0.)
            self.seen_batches = 0

    def forward(self, z):
        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = self.embedding(tokens)

        if z.requires_grad:
            self.ema(z, tokens)
            self.codebook_reset()

        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q


class VmfVQ(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta, temperature=0.5):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)

        self.temperature = temperature
        self.register_buffer("log_param_q_scalar", torch.tensor(-2.995732273553991))
        self.register_buffer("kappa_q", self.log_param_q_scalar.exp() + torch.tensor([1.]))
        self.step = 0

    def _gumbel_softmax_sample(self, logits, eps=1e-10):
        U = torch.rand(logits.size()).to(logits.device)
        g = -torch.log(-torch.log(U + eps) + eps)
        y = logits + g
        return F.softmax(y / self.temperature, dim=-1)
    
    def compute_loss(self, z, z_quantized):
        # Latent loss
        emb_norm = F.normalize(self.embedding.weight, p=2.0, dim=1)
        logit = -self._calc_distance_bw_enc_codes(z, emb_norm)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        kld_discrete = (probabilities * log_probabilities).mean()
        kld_continuous = self._calc_distance_bw_enc_dec(z, z_quantized).mean()        
        loss = kld_discrete + kld_continuous
        
        return loss

    def forward(self, z):
        z = F.normalize(z, p=2.0, dim=1)
        emb_norm = F.normalize(self.embedding.weight, p=2.0, dim=1)

        logit = -self._calc_distance_bw_enc_codes(z, emb_norm)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if z.requires_grad:
            tokens = self._gumbel_softmax_sample(logit)
            z_q = torch.mm(tokens, emb_norm)
            
            self._set_temperature()
        else:
            # if False:
            #     indices = torch.argmax(logit, dim=1).unsqueeze(1)
            #     tokens = torch.zeros(indices.shape[0], self.vocab_size).to(z.device)
            #     tokens.scatter_(1, indices, 1)
            # else:
            dist = Categorical(probabilities)
            indices = dist.sample()
            tokens = F.one_hot(indices, num_classes=self.vocab_size).float()
            z_q = torch.matmul(tokens, emb_norm)


        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q
 
    def _calc_distance_bw_enc_codes(self, z, codebook):
        distances = -self.kappa_q * torch.matmul(z, codebook.t())

        return distances
    
    def _calc_distance_bw_enc_dec(self, x1, x2):
        return torch.sum(x1 * (x1-x2) * self.kappa_q, dim=1)
    
    def _set_temperature(self):
        self.temperature = np.max([1.0 * np.exp(-0.00001*self.step), 0.0])
        self.step += 1

class PerceiverVQ(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # self.embedding.weight.data.normal_()

        self.scale = embed_dim**-0.5

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm_slots = nn.LayerNorm(embed_dim, eps=0.001)
        self.norm_embeds = nn.LayerNorm(embed_dim, eps=0.001)

        self.ref_count = torch.zeros(vocab_size)
        self.ref_count_log = []
        self.save_count_every = 25
        self.save_after = 25
        self.slot_repr = None
        self.pca = None

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + self.beta * (z - z_quantized.detach()).pow(2).mean()

        return commitment_loss

    def forward(self, z, tau=None):
        embedding = repeat(self.embedding.weight.data, 'n d -> b n d', b=z.shape[0])
        embedding = self.norm_embeds(embedding)
        q = self.to_q(embedding)

        z = self.norm_slots(z)
        k, v = self.to_k(z), self.to_v(z)

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        tokens = attn.argmax(dim=1)
        z_q = self.embedding(tokens)

        tokens_onehot = F.one_hot(tokens.flatten().detach(), num_classes=self.vocab_size).float()
        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = z.detach()
        return tokens, z_q


class VQwithReparameterization(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # self.embedding.weight.data.normal_()

        # limit = np.sqrt(6.0 / (1 + tokens_per_slot * embed_dim))
        # mu = torch.rand((vocab_size, embed_dim)).uniform_(-limit, limit)
        # log_sigma = torch.rand((vocab_size, embed_dim)).uniform_(-limit, limit).exp()
        # init_codes = torch.normal(mu, log_sigma)
        # self.embedding.weight.data.copy_(init_codes)

        self.pre_quant_conv_mu = nn.Linear(64, 64)
        self.pre_quant_conv_sigma = nn.Linear(64, 64)
        self.post_quant_conv = nn.Linear(64, 64)

        self.ref_count = torch.zeros(vocab_size)
        self.ref_count_log = []
        self.save_count_every = 25
        self.save_after = 25
        self.slot_repr = None
        self.pca = None

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + self.beta * (z - z_quantized.detach()).pow(2).mean()

        return commitment_loss + self.kl

    def _sample_z(self, mean, std):
        if mean.requires_grad:
            epsilon = torch.randn(mean.shape).to(mean.device)
            return mean + std * epsilon
        else:
            return mean

    def torch_log(self, x):
        return torch.log(torch.clamp(x, min=1e-10))

    def forward(self, z, tau=None):
        z_mu = self.pre_quant_conv_mu(z)
        z_sigma = F.softplus(self.pre_quant_conv_sigma(z))
        z = self._sample_z(z_mu, z_sigma)

        self.kl = -0.5 * torch.mean(torch.sum(1 + self.torch_log(z_sigma**2) - z_mu**2 - z_sigma**2, dim=1))

        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        # probabilities = F.softmax(-dist_to_embeddings / tau, dim=-1)
        # tokens = probabilities.argmax(dim=-1)
        z_q = self.embedding(tokens)

        z_q = self.post_quant_conv(z_q)

        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q


class SlicedVQ(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta

        self.embeddings = nn.ModuleList([])
        for i in range(tokens_per_slot):
            net = nn.Embedding(vocab_size, embed_dim)
            net.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
            self.embeddings.append(net)

        self.ref_count = torch.zeros(tokens_per_slot, vocab_size)
        self.ref_count_log = []
        self.save_count_every = 25
        self.save_after = 25
        self.slot_repr = None
        self.pca = None

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + self.beta * (z - z_quantized.detach()).pow(2).mean()

        return commitment_loss

    def forward(self, z, tau=None):
        # shape of z: ((b k t) e)
        # (b k t) e -> (b k) t e, then for loop over t
        z = rearrange(z, '(b k t) e -> (b k) t e', k=self.num_slots, t=self.tokens_per_slot)
        z = self.pre_quant_conv(z)
        
        z_q = []
        tokens = []
        tokens_onehot = []
        for t in range(self.tokens_per_slot):
            z_t = z[:, t, :]
            dist_to_embeddings = torch.sum(z_t ** 2, dim=1, keepdim=True) + torch.sum(self.embeddings[t].weight**2, dim=1) - 2 * torch.matmul(z_t, self.embeddings[t].weight.t())
            token = dist_to_embeddings.argmin(dim=-1)
            z_q_t = self.embeddings[t](token)
            z_q.append(z_q_t)
            tokens.append(token)

            token_onehot = F.one_hot(token, num_classes=self.vocab_size).float()
            tokens_onehot.append(token_onehot)
        z_q = torch.stack(z_q, dim=1)
        tokens = torch.stack(tokens, dim=1).flatten(0,1)
        tokens_onehot = torch.stack(tokens_onehot, dim=1)

        z_q = rearrange(z_q, '(b k) t e -> (b k t) e', k=self.num_slots, t=self.tokens_per_slot)

        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = z.detach()
        return tokens, z_q
    
    def get_embedding(self, tokens):
        # shape of tokens: (b (k t))
        tokens = rearrange(tokens, 'b (k t) -> (b k) t', k=self.num_slots, t=self.tokens_per_slot)
        z_q = []
        for t in range(self.tokens_per_slot):
            z_q_t = self.embeddings[t](tokens[:, t])
            z_q.append(z_q_t)
        z_q = torch.stack(z_q, dim=1)
        z_q = rearrange(z_q, '(b k) t e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot)

        return z_q

    def _reset_count(self):
        self.ref_count = torch.zeros(self.tokens_per_slot, self.vocab_size)
        self.ref_count_log = []
        
    def plot_count(self, epoch, save_dir):
        self.ref_count_log.append(self.ref_count.numpy())
        if epoch >= self.save_after and epoch % self.save_count_every == 0:
            self.ref_count_log = np.array(self.ref_count_log)
            plt.figure(figsize=(10,1))
            for t in range(self.tokens_per_slot):
                plt.subplot(1,2,t+1)
                plt.imshow(self.ref_count_log[:, t])
            plt.tight_layout()
            plt.savefig(save_dir / f"ref_count_{epoch}.png")
            plt.close()
            self._reset_count()


class BlockAttention(nn.Module):
    def __init__(self, vocab_size, tokens_per_slot):
        super().__init__()

        self.vocab_size = vocab_size
        self.tokens_per_slot = tokens_per_slot

    def forward(self, q, k, v):
        """
        q: (bs,k,t,d), k/v: (bs,vocab_size,t,d), attn: (bs,t,k,vocab_size)
        return: out (bs,k,t,d), tokens (bs,k,t)
        """
        bs, K = q.shape[:2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)

        tokens = []
        for t in range(self.tokens_per_slot):
            token = attn[:, t].argmax(dim=-1)
            tokens.append(token)
        tokens = torch.stack(tokens, dim=-1)

        output = torch.matmul(attn, v).transpose(1, 2)

        return output, tokens


class BinderQuantization(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta
        
        # block prototype memory
        self.embeddings = nn.Parameter(torch.zeros(1, vocab_size, tokens_per_slot, self.embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.embeddings)
        self.mem_proj = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.embed_dim, self.embed_dim)
        )

        # norms
        self.norm_mem = nn.LayerNorm(self.embed_dim, elementwise_affine=False)
        self.norm_query = nn.LayerNorm(self.embed_dim, elementwise_affine=False)

        # block attention
        self.attn = BlockAttention(vocab_size, tokens_per_slot)

        self.ref_count = torch.zeros(tokens_per_slot, vocab_size)
        self.ref_count_log = []
        self.save_count_every = 25
        self.save_after = 25
        self.slot_repr = None
        self.pca = None

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + self.beta * (z - z_quantized.detach()).pow(2).mean()

        return commitment_loss

    def forward(self, z, tau=None):
        z = rearrange(z, '(b k t) e -> b k t e', k=self.num_slots, t=self.tokens_per_slot)

        # get memories
        mem = self.mem_proj(self.embeddings)  # (1,vocab_size,tokens_per_slot,embed_dim)

        # norms
        mem = self.norm_mem(mem)  # (1,vocab_size,tokens_per_slot,embed_dim)
        queries = self.norm_query(z)  # (bs,num_slots,tokens_per_slot,embed_dim)

        # broadcast
        mem = mem.expand(z.shape[0], -1, -1, -1)  # (bs,num_prototypes,tokens_per_slot,embed_dim)

        # read
        z_q, tokens = self.attn(queries, mem, mem)  # (bs,num_slots,tokens_per_slot,embed_dim), (bs,num_slots,tokens_per_slot)

        tokens_onehot = []
        for t in range(self.tokens_per_slot):
            token = tokens[:, :, t].flatten(0, 1)
            token_onehot = F.one_hot(token, num_classes=self.vocab_size).float()
            tokens_onehot.append(token_onehot)
        tokens_onehot = torch.stack(tokens_onehot, dim=1)
        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = rearrange(z, 'b k t e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()

        z_q = rearrange(z_q, 'b k t e -> (b k t) e', k=self.num_slots, t=self.tokens_per_slot)
        tokens = rearrange(tokens, 'b k t -> (b k t)', k=self.num_slots, t=self.tokens_per_slot)

        return tokens, z_q
    
    def get_embedding(self, tokens):
        # shape of tokens: (b (k t))
        tokens = rearrange(tokens, 'b (k t) -> (b k) t', k=self.num_slots, t=self.tokens_per_slot)
        
        z_q = []
        for t in range(self.tokens_per_slot):
            z_q_t = self.embeddings[0, tokens[:, t], t]
            z_q.append(z_q_t)
        z_q = torch.stack(z_q, dim=1)
        z_q = rearrange(z_q, '(b k) t e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot)

        return z_q
    
    def _reset_count(self):
        self.ref_count = torch.zeros(self.tokens_per_slot, self.vocab_size)
        self.ref_count_log = []
        
    def plot_count(self, epoch, save_dir):
        self.ref_count_log.append(self.ref_count.numpy())
        if epoch >= self.save_after and epoch % self.save_count_every == 0:
            self.ref_count_log = np.array(self.ref_count_log)
            plt.figure(figsize=(10,1))
            for t in range(self.tokens_per_slot):
                plt.subplot(1,2,t+1)
                plt.imshow(self.ref_count_log[:, t])
            plt.tight_layout()
            plt.savefig(save_dir / f"ref_count_{epoch}.png")
            plt.close()
            self._reset_count()


class SlicedPerceiverVQ(BaseVQ):
    def __init__(self, vocab_size, embed_dim, num_slots, tokens_per_slot, beta):
        super().__init__(vocab_size, embed_dim, num_slots, tokens_per_slot, beta)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot
        self.beta = beta

        self.embeddings = nn.ModuleList([])
        for i in range(tokens_per_slot):
            net = nn.Embedding(vocab_size, embed_dim)
            net.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
            self.embeddings.append(net)

        self.scale = embed_dim**-0.5

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm_slots = nn.LayerNorm(embed_dim, eps=0.001)
        self.norm_embeds = nn.LayerNorm(embed_dim, eps=0.001)

        self.ref_count = torch.zeros(tokens_per_slot, vocab_size)
        self.ref_count_log = []
        self.save_count_every = 25
        self.save_after = 25
        self.slot_repr = None
        self.pca = None

    def compute_loss(self, z, z_quantized):
        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + self.beta * (z - z_quantized.detach()).pow(2).mean()

        return commitment_loss

    def forward(self, z, tau=None):
        # shape of z: ((b k t) e)
        # (b k t) e -> (b k) t e, then for loop over t
        z = rearrange(z, '(b k t) e -> (b k) t e', k=self.num_slots, t=self.tokens_per_slot)
        z = self.norm_slots(z)
        k = self.to_k(z)
        
        z_q = []
        tokens = []
        tokens_onehot = []
        for t in range(self.tokens_per_slot):
            k_t = k[:, t, :].unsqueeze(1)

            embedding = repeat(self.embeddings[t].weight.data, 'n d -> b n d', b=z.shape[0])
            embedding = self.norm_embeds(embedding)
            q_t = self.to_q(embedding)

            dots = torch.einsum("bid,bjd->bij", q_t, k_t) * self.scale
            attn = dots.softmax(dim=-1)
            token = attn.argmax(dim=1).squeeze(1)
            z_q_t = self.embeddings[t](token)
            z_q.append(z_q_t)
            tokens.append(token)

            token_onehot = F.one_hot(token, num_classes=self.vocab_size).float()
            tokens_onehot.append(token_onehot)
        z_q = torch.stack(z_q, dim=1)
        tokens = torch.stack(tokens, dim=1).flatten(0,1)
        tokens_onehot = torch.stack(tokens_onehot, dim=1)

        z_q = rearrange(z_q, '(b k) t e -> (b k t) e', k=self.num_slots, t=self.tokens_per_slot)

        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = z.detach()
        return tokens, z_q
    
    def get_embedding(self, tokens):
        # shape of tokens: (b (k t))
        tokens = rearrange(tokens, 'b (k t) -> (b k) t', k=self.num_slots, t=self.tokens_per_slot)
        z_q = []
        for t in range(self.tokens_per_slot):
            z_q_t = self.embeddings[t](tokens[:, t])
            z_q.append(z_q_t)
        z_q = torch.stack(z_q, dim=1)
        z_q = rearrange(z_q, '(b k) t e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot)

        return z_q

    def _reset_count(self):
        self.ref_count = torch.zeros(self.tokens_per_slot, self.vocab_size)
        self.ref_count_log = []

    def plot_count(self, epoch, save_dir):
        self.ref_count_log.append(self.ref_count.numpy())
        if epoch >= self.save_after and epoch % self.save_count_every == 0:
            self.ref_count_log = np.array(self.ref_count_log)
            plt.figure(figsize=(10,1))
            for t in range(self.tokens_per_slot):
                plt.subplot(1,2,t+1)
                plt.imshow(self.ref_count_log[:, t])
            plt.tight_layout()
            plt.savefig(save_dir / f"ref_count_{epoch}.png")
            plt.close()
            self._reset_count()