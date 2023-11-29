from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses
from .transformerXL import *
from torch.distributions.categorical import Categorical

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    mems: torch.FloatTensor

@dataclass
class ContinuosWorldModelOutput:
    output_sequence: torch.FloatTensor
    encodings: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor

class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern


        if self.config.model== 'iris':
            self.transformer = Transformer(config)
            self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
            self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
            )
            

        # Transformer XL, from tmw https://github.com/jrobine/twm
        # ---------------------------------------------------------------------------
        else:

            modality_order = ['z', 'a']
            num_current = self.config.tokens_per_block # num obs_tokens + act_tokens = 17
            self.modality_order = modality_order
            memory_length = config.wm_memory_length
            max_length = 1 + config.wm_sequence_length  # 1 for context


            if self.config.model == 'irisXL-discrete': #to reproduce iris with transformer XL
                embeds = {
                    'z': {'in_dim': obs_vocab_size, 'categorical': True},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                }

                self.embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], config.dyn_embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], config.dyn_embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in embeds.items()
                })

                self.transformer = TransformerXL(
                modality_order, num_current, embed_dim=config.dyn_embed_dim,
                activation=config.dyn_act, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length)


            elif self.config.model == 'irisXL-continuos': #to reproduce iris with transformer XL
                embeds = {
                    'z': {'in_dim': config.embed_dim, 'categorical': False},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                }

                self.embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], config.dyn_embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], config.dyn_embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in embeds.items()
                })


                self.transformer = TransformerXL(
                modality_order, num_current, embed_dim=config.dyn_embed_dim,
                activation=config.dyn_act, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length)



            elif self.config.model == 'Asymmetric': #for Asymmetric Transformer XL approach 
                continuos_embeds = {
                    'z': {'in_dim': config.embed_dim, 'categorical': False},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                } 
                discrete_embeds = {
                    'z': {'in_dim': obs_vocab_size, 'categorical': True},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                }

                self.continuos_embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], config.dyn_embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], config.dyn_embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in continuos_embeds.items()
                })

                self.discrete_embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], config.dyn_embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], config.dyn_embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in discrete_embeds.items()
                })

                self.continuos_transformer = TransformerXL(
                modality_order, num_current, continuos_embeds, embed_dim=config.dyn_embed_dim,
                activation=config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length)

                self.discrete_transformer = TransformerXL(
                modality_order, num_current, discrete_embeds, embed_dim=config.dyn_embed_dim,
                activation=config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length)

            else:
                raise ValueError(f'Unsupported model: {self.config.model}')

        #---------------------------------------------------------------------------------



        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )
        
        self.head_embeddings = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, inputs: dict, past_keys_values: Optional[KeysValues] = None, mems: Optional[list] = None, stop_mask: Optional[bool] = None, tgt_length: Optional[int] = None, embedding_input:Optional[bool] = False) -> WorldModelOutput:

        if self.config.model == 'iris':
            if isinstance(inputs, torch.cuda.LongTensor):
                tokens = inputs
            else:
                tokens = inputs['tokens']
            num_steps = tokens.size(1)  # (B, T)
            assert num_steps <= self.config.max_tokens
            prev_steps = 0 if past_keys_values is None else past_keys_values.size

            sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

            x = self.transformer(sequences, past_keys_values)

            logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

            return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, None)

        elif self.config.model == 'irisXL-discrete' or self.config.model == 'irisXL-continuos':

            if inputs['z'].size(1) == 1:
                num_steps = 1
            else:
                num_steps = inputs['z'].size(1)*(inputs['z'].size(2)+1)  # (B, L, T) -> L*(T+1), 1 for action token
      
            assert num_steps <= self.config.max_tokens
            prev_steps = 0 if past_keys_values is None else past_keys_values.size
            if mems is not None and 'a' not in inputs.keys():
                if embedding_input:
                    inputs = {'z': inputs['h']} # using directly transformer output from previous timestep
                else:
                    inputs = {'z': self.embedder['z'](inputs['z'])}  # fix embedder, and inputs format for actor critic part
                prev_steps = 16 + mems[0].shape[0] # num default # of obs tokens
            elif embedding_input and mems is not None:
                    inputs = {'z': inputs['h']} # using directly transformer output from previous timestep
            
            h, mems = self.transformer(inputs, tgt_length, stop_mask, mems)

            logits_rewards = self.head_rewards(h, num_steps=num_steps, prev_steps=prev_steps)
            logits_ends = self.head_ends(h, num_steps=num_steps, prev_steps=prev_steps)
            if self.config.model == 'irisXL-discrete':
                logits_observations = self.head_observations(h, num_steps=num_steps, prev_steps=prev_steps)
                if embedding_input:
                    embeddings = self.head_embeddings(h, num_steps=num_steps, prev_steps=prev_steps)
                else: 
                    embeddings = h
                return WorldModelOutput(embeddings, logits_observations, logits_rewards, logits_ends, mems)
            else:
                encodings = self.head_embeddings(h, num_steps=num_steps, prev_steps=prev_steps)
                return ContinuosWorldModelOutput(h, encodings, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        regularization = kwargs.pop('regularization')
        self.embedding_input = kwargs.pop('embedding_input')
        if self.config.model == 'iris':
            with torch.no_grad():
                obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)

            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
            tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
            inputs = {'tokens': tokens}

            outputs = self(inputs)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        elif self.config.model == 'irisXL-continuos':
            with torch.no_grad():
                obs_encodings = tokenizer.encode(batch['observations'], should_preprocess=True).z_quantized  # (B, L, K, E)

            inputs = {'z': obs_encodings, 'a': batch['actions']}
        
            outputs = self(inputs, stop_mask=batch['ends'], tgt_length=obs_encodings.shape[1])

            labels_embeddings, labels_rewards, labels_ends = self.compute_labels_continuos_world_model(obs_encodings, batch['rewards'], batch['ends'], batch['mask_padding'])

            embeddings = rearrange(outputs.embeddings[:, :-1], 'b t o e -> (b t) o e')
            loss_embedding = F.mse_loss(embeddings, labels_embeddings)
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
            
            return LossWithIntermediateLosses(loss_rewards=loss_rewards, loss_ends=loss_ends, loss_embeddings=loss_embedding)

        elif self.config.model == 'irisXL-discrete':
            with torch.no_grad():
                obs_encoded = tokenizer.encode(batch['observations'], should_preprocess=True)  # (B, L, K)
                obs_tokens, obs_zq = obs_encoded.tokens, obs_encoded.z_quantized
               
            tokens = {'z': obs_tokens, 'a': batch['actions']}
            inputs = {name: mod(tokens[name]) for name, mod in self.embedder.items()}

            outputs = self(inputs, stop_mask=batch['ends'], tgt_length=obs_tokens.shape[1], embedding_input=self.embedding_input)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
            if self.embedding_input:
                embeddings = outputs.output_sequence[:, :-1]
                loss_embedding = F.mse_loss(embeddings, rearrange(inputs['z'], 'b s t o -> b (s t) o')[:, 1:])
                
                return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends, loss_embeddings=loss_embedding)
            
            
            
        elif self.config.model == 'irisXL-discrete' and regularization:
            with torch.no_grad():
                obs_encoded = tokenizer.encode(batch['observations'], should_preprocess=True)  # (B, L, K)
                obs_tokens, obs_zq = obs_encoded.tokens, obs_encoded.z_quantized
                post_quant_zq = tokenizer.get_post_zq(obs_zq)
               
            tokens = {'z': obs_tokens, 'a': batch['actions']}
            inputs = {name: mod(tokens[name]) for name, mod in self.embedder.items()}

            for i in range(obs_tokens.shape[2]+1):  # extra iteration to account for action token 
                outputs = self(inputs, stop_mask=batch['ends'], tgt_length=obs_tokens.shape[1], embedding_input=self.embedding_input)
                tokens = {'z': Categorical(logits=outputs.logits_observations).sample(), 'a': batch['actions']}
                inputs = {name: mod(tokens[name]) for name, mod in self.embedder.items()}
                if self.embedding_input:
                    inputs['h'] = outputs.output_sequence
                
                    

            i = i - 1 # one extra iteration is done to account for action token, but to remove tokens of the last observation I need only i - 1 then 
            labels_observations, _, _ = self.compute_labels_world_model(obs_tokens[:, i:], batch['rewards'], batch['ends'], batch['mask_padding'][:, i:])

            logits_observations = rearrange(outputs.logits_observations[:, :-(1 + i)], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            
            tokens = Categorical(logits=outputs.logits_observations).sample()
            post_quant_z = tokenizer.post_quant_conv(tokenizer.embedding(tokens))
            loss_post_quant = F.mse_loss(post_quant_z[:, :-(1 + i)], post_quant_zq[:, (1 + i):])
            
            return LossWithIntermediateLosses(loss_obs=loss_obs, loss_regularization=loss_post_quant)
            

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)

    def compute_labels_continuos_world_model(self, encodings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(encodings.masked_fill(mask_fill.unsqueeze(-1).expand_as(encodings), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)


