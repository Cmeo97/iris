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
#from .transformerXL import *


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
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


        if self.config['model'] != 'iris':
            modality_order = ['z', 'a']
            num_current = 2
            self.modality_order = modality_order
            memory_length = config['wm_memory_length']
            max_length = 1 + config['wm_sequence_length']  # 1 for context
            self.transformer = Transformer(config)
            self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
            self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
            )

        # Transformer XL, from tmw https://github.com/jrobine/twm
        # ---------------------------------------------------------------------------

        if self.config['model'] == 'irisXL-discrete': #to reproduce iris with transformer XL
            embeds = {
                'z': {'in_dim': config.embed_dim, 'categorical': False},
                'a': {'in_dim': act_vocab_size, 'categorical': True}
            }


            self.transformer = TransformerXL(
            modality_order, num_current, embeds, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)

            ## Not sure if needed
            self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim)])
            )

        elif self.config['model'] == 'irisXL-continuos': #to reproduce iris with transformer XL
            embeds = {
                'a': {'in_dim': act_vocab_size, 'categorical': True}
            }

            self.transformer = TransformerXL(
            modality_order, num_current, embeds, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)

            ## Not sure if needed
            self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim)])
            )



        elif self.config['model'] == 'Asymmetric': #for Asymmetric Transformer XL approach 
            continuos_embeds = {
                'a': {'in_dim': act_vocab_size, 'categorical': True}
            } 
            discrete_embeds = {
                'z': {'in_dim': config.embed_dim, 'categorical': False},
                'a': {'in_dim': act_vocab_size, 'categorical': True}
            }
        
            self.discrete_transformer = TransformerXL(
            modality_order, num_current, discrete_embeds, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)

            self.continuos_transformer = TransformerXL(
            modality_order, num_current, continuos_embeds, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)


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

    def forward(self, inputs: dict, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        if self.config['model'] == 'iris':
            tokens = inputs['tokens']
            num_steps = tokens.size(1)  # (B, T)
            assert num_steps <= self.config.max_tokens
            prev_steps = 0 if past_keys_values is None else past_keys_values.size

            sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

            x = self.transformer(sequences, past_keys_values)

            logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        elif self.config['model'] == 'irisXL-discrete' or self.config['model'] == 'irisXL-continuos':
            x = self.transformer(inputs)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:


        if self.config['model'] == 'iris':
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
        elif self.config['model'] == 'irisXL-continuos':
            with torch.no_grad():
                obs_encodings = tokenizer.encode(batch['observations'], should_preprocess=True).z  # (B, L, K)

            tokens_act = batch['actions'] # (B, L(1))
            inputs = {'z': obs_encodings, 'a': tokens_act}

            with torch.no_grad():
                context_z = obs_encodings[:, :1] 
                next_z = obs_encodings[:, :-1]
             
            z = torch.cat([context_z, z], dim=1).detach()
        
            target_z = torch.cat([z[:, 1:], next_z], dim=1).detach()
            stop_mask = batch['ends'] # or , batch['mask_padding']
     
            tgt_length = target_z.shape[1]

            outputs = self(inputs, tgt_length, stop_mask)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        elif self.config['model'] == 'irisXL-discrete':
            with torch.no_grad():
                tokens_obs = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

            tokens_act = batch['actions'] # (B, L(1))
            inputs = {'z': tokens_obs, 'a': tokens_act}

            outputs = self(inputs)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)




class DynamicsModel(nn.Module):

    def __init__(self, config, z_dim, num_actions):
        super().__init__()
        self.config = config

        if self.config['discrete']:
            embeds = {
                'z': {'in_dim': z_dim, 'categorical': False},
                'a': {'in_dim': num_actions, 'categorical': True}
            }
        else:
            embeds = {
                'a': {'in_dim': num_actions, 'categorical': True}
            } 

        modality_order = ['z', 'a']
        num_current = 2

        #if config['dyn_input_rewards']:
        #    embeds['r'] = {'in_dim': 0, 'categorical': False}
        #    modality_order.append('r')
#
        #if config['dyn_input_discounts']:
        #    embeds['g'] = {'in_dim': 0, 'categorical': False}
        #    modality_order.append('g')

        self.modality_order = modality_order

        #out_heads = {
        #    'z': {'hidden_dims': config['dyn_z_dims'], 'out_dim': z_dim},
        #    'r': {'hidden_dims': config['dyn_reward_dims'], 'out_dim': 1, 'final_bias_init': 0.0},
        #    'g': {'hidden_dims': config['dyn_discount_dims'], 'out_dim': 1,
        #          'final_bias_init': config['env_discount_factor']}
        #}

        memory_length = config['wm_memory_length']
        max_length = 1 + config['wm_sequence_length']  # 1 for context
        self.prediction_net = PredictionNet(
            modality_order, num_current, embeds, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)

    @property
    def h_dim(self):
        return self.prediction_net.embed_dim

    def predict(self, z, a, r, d, tgt_length, mems=None, return_attention=False,
               ):
        assert utils.check_no_grad(z, a, r, d)
        assert mems is None or utils.check_no_grad(*mems)

        #if compute_consistency:
        #    tgt_length += 1  # add 1 timestep for context

        inputs = {'z': z, 'a': a}
        #heads = tuple(heads) if heads is not None else ('z', 'r', 'g')

        outputs = self.prediction_net(
            inputs, tgt_length, stop_mask=d, mems=mems, return_attention=return_attention)
        out, h, mems, attention = outputs if return_attention else (outputs + (None,))


    