from einops import rearrange
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

    def forward(self, z):
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

        self.save_after = 25
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
        z_logits = F.log_softmax(self.pre_vq_linear(z), dim=1)
        z_soft = self.gumbel_softmax(z_logits, tau, hard, dim=1)
        z_hard = self.gumbel_softmax(z_logits, tau, True, dim=1)
        z_q = self.post_vq_linear(z_soft)
        tokens = z_hard.argmax(dim=-1)

        self.slot_targ = z.detach()
        self.slot_repr = z_q.detach()
        return tokens, z_q
    
    def decode_tokens(self, tokens):
        z_hard = F.one_hot(tokens, num_classes=self.vocab_size).float()
        z_q = self.post_vq_linear(z_hard)
        return z_q

    def plot_count(self, epoch, save_dir):
        pass

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
        # self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # self.embedding.weight.data.normal_()

        limit = np.sqrt(6.0 / (1 + tokens_per_slot * embed_dim))
        mu = torch.rand((vocab_size, embed_dim)).uniform_(-limit, limit)
        log_sigma = torch.rand((vocab_size, embed_dim)).uniform_(-limit, limit).exp()
        init_codes = torch.normal(mu, log_sigma)
        self.embedding.weight.data.copy_(init_codes)

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

    def forward(self, z):
        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = self.embedding(tokens)

        tokens_onehot = F.one_hot(tokens, num_classes=self.vocab_size).float()
        if z.requires_grad: # during training
            ref_count = torch.sum(tokens_onehot, dim=0)
            self.ref_count += ref_count.detach().cpu()

        self.slot_repr = rearrange(z, '(b k t) e -> b (k t) e', k=self.num_slots, t=self.tokens_per_slot).detach()
        return tokens, z_q

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
