from dataclasses import dataclass
from typing import Any, Optional, Tuple
import math

from einops import rearrange
import matplotlib.pyplot as plt
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
class WorldModelOutputEmbs:
    output_sequence: torch.FloatTensor
    embeddings: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    mems: torch.FloatTensor

@dataclass
class ContinuosWorldModelOutput:
    output_sequence: torch.FloatTensor
    embeddings: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    mems: torch.FloatTensor

class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        if config.model == 'OC-irisXL':
            all_but_last_obs_tokens_pattern[-1] = 0
        else:
            all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern
        self.regularization_post_quant = config.regularization_post_quant
        self.embedding_input = config.embedding_input
        self.regularization_tokens = config.regularization_tokens
        self.slot_regularization = config.slot_regularization
            
        
        self.regularization_k_pred = config.regularization_k_pred
        self.regularization_embeddings = config.regularization_embeddings
        self.regularization = (self.regularization_post_quant or self.regularization_tokens or self.regularization_embeddings or self.regularization_k_pred or self.slot_regularization)
        if self.regularization_tokens:
            self.loss_iter = []
            
        if self.regularization_embeddings:
            self.loss_iter_embeds = []

        
        embed_dim = config.continuos_embed_dim if (self.config.model == 'irisXL-continuos' or self.config.model == 'OC-irisXL') else config.embed_dim
        self.slot_based = True if self.config.model == 'OC-irisXL' else False
       
            
        if self.config.model== 'iris':
            self.transformer = Transformer(config)
            self.pos_emb = nn.Embedding(config.max_tokens, embed_dim)
            self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, embed_dim), nn.Embedding(obs_vocab_size, embed_dim)])
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
                name: nn.Embedding(embed['in_dim'], embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in embeds.items()
                })

                self.transformer = TransformerXL(
                modality_order, num_current, embed_dim=embed_dim,
                activation=config.dyn_act, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length)


            elif self.config.model == 'irisXL-continuos' or self.config.model == 'OC-irisXL': #to reproduce iris with transformer XL
                embeds = {
                    'z': {'in_dim': config.continuos_embed_dim, 'categorical': False},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                }

                
                self.embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in embeds.items()
                })


                self.transformer = TransformerXL(
                modality_order, num_current, embed_dim=embed_dim,
                activation=config.dyn_act, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length, slot_based=(self.config.model == 'OC-irisXL'))

            elif self.config.model == 'Asymmetric': #for Asymmetric Transformer XL approach 
                continuos_embeds = {
                    'z': {'in_dim': embed_dim, 'categorical': False},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                } 
                discrete_embeds = {
                    'z': {'in_dim': obs_vocab_size, 'categorical': True},
                    'a': {'in_dim': act_vocab_size, 'categorical': True}
                }

                self.continuos_embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in continuos_embeds.items()
                })

                self.discrete_embedder = nn.ModuleDict({
                name: nn.Embedding(embed['in_dim'], embed_dim) if embed.get('categorical', False) else
                MLP(embed['in_dim'], [], embed_dim, config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout, post_activation=True)
                for name, embed in discrete_embeds.items()
                })

                self.continuos_transformer = TransformerXL(
                modality_order, num_current, continuos_embeds, embed_dim=embed_dim,
                activation=config.dyn_act, norm=config.dyn_norm, dropout_p=config.dyn_dropout,
                feedforward_dim=config.dyn_feedforward_dim, head_dim=config.dyn_head_dim,
                num_heads=config.dyn_num_heads, num_layers=config.dyn_num_layers,
                memory_length=memory_length, max_length=max_length)

                self.discrete_transformer = TransformerXL(
                modality_order, num_current, discrete_embeds, embed_dim=embed_dim,
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
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, obs_vocab_size)
            )
        )
        
        self.head_embeddings = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, config.continuos_embed_dim)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 2)
            )
        )

        self.shift_action_token = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.ReLU(),
            )
        )


        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, inputs: dict, past_keys_values: Optional[KeysValues] = None, mems: Optional[list] = None, stop_mask: Optional[bool] = None, tgt_length: Optional[int] = None, embedding_input:Optional[bool] = False, generation: Optional[bool] = False) -> WorldModelOutput:

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
        
        elif self.config.model == 'OC-irisXL':

            if generation:
                #inputs = {'z': self.embedder['z'](inputs['z'])} if (mems is None or mems[0].shape[0]%7 == 0) else {'z': inputs['z']}  # fix embedder, and inputs format for actor critic part
                num_steps = inputs['z'].size(1)*(inputs['z'].size(2)+inputs['a'].size(2))
                prev_steps =  0 if mems is None else mems[0].shape[0] # num default # of obs tokens
            else: 
                if self.regularization and mems is not None:
                    actions = inputs['a']
                    inputs = inputs['z']
                    num_steps, prev_steps = inputs.shape[1], mems[0].shape[0]
                    inputs[:, self.shift_action_token.compute_slice(num_steps, prev_steps)] = actions
                    inputs = {'z': inputs}
                else:
                    num_steps, prev_steps = inputs['z'].size(1)*(inputs['z'].size(2)+1), 0  # (B, L, T) -> L*(T+1), 1 for action token 
            
            h, mems = self.transformer(inputs, tgt_length, stop_mask, mems, generation)

            logits_rewards = self.head_rewards(h, num_steps=num_steps, prev_steps=prev_steps)
            logits_ends = self.head_ends(h, num_steps=num_steps, prev_steps=prev_steps)
            embeddings = self.head_embeddings(h, num_steps=num_steps, prev_steps=prev_steps)
            return ContinuosWorldModelOutput(h, embeddings, logits_rewards, logits_ends, mems)

        elif self.config.model == 'irisXL-continuos':

            if generation:
                inputs = {'z': self.embedder['z'](inputs['z'])} if (mems is None or mems[0].shape[0]%self.config.tokens_per_block-1 == 0) else {'z': inputs['z']}  # fix embedder, and inputs format for actor critic part
                num_steps = inputs['z'].size(1) if inputs['z'].dim() == 3 else inputs['z'].size(1)*inputs['z'].size(2)
                prev_steps =  self.config.tokens_per_block-1 if mems is None else self.config.tokens_per_block-1 + mems[0].shape[0] # num default # of obs tokens
            else: 
                if self.regularization and mems is not None:
                    actions = inputs['a']
                    inputs = inputs['z']
                    num_steps, prev_steps = inputs.shape[1], mems[0].shape[0]
                    inputs[:, self.shift_action_token.compute_slice(num_steps, prev_steps)] = actions
                    inputs = {'z': inputs}
                else:
                    num_steps, prev_steps = inputs['z'].size(1)*(inputs['z'].size(2)+1), 0  # (B, L, T) -> L*(T+1), 1 for action token 
            
            h, mems = self.transformer(inputs, tgt_length, stop_mask, mems, generation)

            logits_rewards = self.head_rewards(h, num_steps=num_steps, prev_steps=prev_steps)
            logits_ends = self.head_ends(h, num_steps=num_steps, prev_steps=prev_steps)
            embeddings = self.head_embeddings(h, num_steps=num_steps, prev_steps=prev_steps)
            return ContinuosWorldModelOutput(h, embeddings, logits_rewards, logits_ends, mems)

        elif self.config.model == 'irisXL-discrete':

            if generation:
                if embedding_input:
                    inputs = {'z': inputs['h']} # using directly transformer output from previous timestep
                else:
                    inputs = {'z': self.embedder['z'](inputs['z'])}  # fix embedder, and inputs format for actor critic part
                num_steps = inputs['z'].size(1) if inputs['z'].dim() == 3 else inputs['z'].size(1)*inputs['z'].size(2)
                prev_steps = 16 + mems[0].shape[0] # num default # of obs tokens
            else: 
                if self.regularization and mems is not None:
                    actions = inputs['a']
                    inputs = inputs['z'] 
                    num_steps, prev_steps = inputs.shape[1], mems[0].shape[0]
                    inputs[:, self.shift_action_token.compute_slice(num_steps, prev_steps)] = actions
                    inputs = {'z': inputs}
                else:
                    num_steps, prev_steps = inputs['z'].size(1)*(inputs['z'].size(2)+1), 0  # (B, L, T) -> L*(T+1), 1 for action token 
            
            h, mems = self.transformer(inputs, tgt_length, stop_mask, mems, generation)

            logits_rewards = self.head_rewards(h, num_steps=num_steps, prev_steps=prev_steps)
            logits_ends = self.head_ends(h, num_steps=num_steps, prev_steps=prev_steps)
            logits_observations = self.head_observations(h, num_steps=num_steps, prev_steps=prev_steps)
            if embedding_input:
                embeddings = self.head_embeddings(h, num_steps=num_steps, prev_steps=prev_steps)
                return WorldModelOutputEmbs(h, embeddings, logits_observations, logits_rewards, logits_ends, mems)
            else: 
                return WorldModelOutput(h, logits_observations, logits_rewards, logits_ends, mems)


    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        
        
        if self.config.model == 'iris':
            with torch.no_grad():
                obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)

            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
            tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
            inputs = {'tokens': tokens}

            outputs = self(inputs)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations.reshape(-1))
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
            loss = LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

        elif self.config.model == 'irisXL-continuos':
            
            with torch.no_grad():
                obs_embeddings = tokenizer.encode(batch['observations'], should_preprocess=True).z_quantized  # (B, L, E, K^1/2, K^1/2)
            
            obs_embeddings = rearrange(obs_embeddings, 'b t e o p -> b t (o p) e')

            inputs_ = {'z': obs_embeddings, 'a': batch['actions']}
            inputs = {name: mod(inputs_[name]) for name, mod in self.embedder.items()}
            outputs = self(inputs, stop_mask=batch['ends'], tgt_length=obs_embeddings.shape[1])

            end_positions, labels_rewards, labels_ends = self.compute_labels_continuos_world_model(obs_embeddings, batch['rewards'], batch['ends'], batch['mask_padding'])

            loss_embeddings = F.mse_loss(outputs.embeddings[:, :-1], rearrange(obs_embeddings, 'b t e o  -> b (t e) o')[:, 1:], reduction='none')
            loss_embeddings[end_positions] *= 0
           
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
            
            loss = LossWithIntermediateLosses(loss_rewards=loss_rewards, loss_ends=loss_ends, loss_embeddings=loss_embeddings.mean())
            
        elif self.config.model == 'OC-irisXL' and not self.slot_regularization:
            
            with torch.no_grad():
                obs_embeddings = tokenizer.encode(batch['observations'], should_preprocess=True).z  # (B, L, Num_slots, slot_dim)

            inputs_ = {'z': obs_embeddings, 'a': batch['actions']}
            inputs = {name: mod(inputs_[name]) for name, mod in self.embedder.items()}
            outputs = self(inputs, stop_mask=batch['ends'], tgt_length=obs_embeddings.shape[1])

            end_positions, labels_rewards, labels_ends = self.compute_labels_OC_world_model(obs_embeddings, batch['rewards'], batch['ends'], batch['mask_padding'])

            loss_embeddings = F.mse_loss(outputs.embeddings[:, :-self.config.tokens_per_block], rearrange(obs_embeddings, 'b t e o  -> b (t e) o')[:, self.config.tokens_per_block:], reduction='none')
            loss_embeddings[end_positions] *= 0
           
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
            
            loss = LossWithIntermediateLosses(loss_rewards=loss_rewards, loss_ends=loss_ends, loss_embeddings=loss_embeddings.mean())
            
        elif self.config.model == 'OC-irisXL' and self.slot_regularization:
            
            
            with torch.no_grad():
                obs_embeddings = tokenizer.encode(batch['observations'], should_preprocess=True).z  # (B, L, E, K^1/2, K^1/2)
                sam_features = tokenizer.decode(obs_embeddings[:, 1:])
            
            obs_embeddings = rearrange(obs_embeddings, 'b t e o p -> b t (o p) e')

            inputs_ = {'z': obs_embeddings, 'a': batch['actions']}
            inputs = {name: mod(inputs_[name]) for name, mod in self.embedder.items()}
            mems = None
            k = 6
            for i in range(self.config.tokens_per_block):  
                outputs = self(inputs, mems=mems, stop_mask=batch['ends'], tgt_length=obs_tokens.shape[1])

                if i == 0:
                    end_positions, labels_rewards, labels_ends = self.compute_labels_continuos_world_model(obs_embeddings, batch['rewards'], batch['ends'], batch['mask_padding'])
                    loss_embeddings = F.mse_loss(outputs.embeddings[:, :-1], rearrange(obs_embeddings, 'b t e o  -> b (t e) o')[:, 1:], reduction='none')
                    loss_embeddings[end_positions] *= 0
                    loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
                    loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
                    labels_embeddings = rearrange(inputs['z'], 'b s t o -> b (s t) o')[:, 1:]
            
                next_input = outputs.output_sequence
                next_input[:, self.head_observations.compute_slice(num_steps=next_input.shape[1], prev_steps=i)] = self.embedder['z'](outputs.embeddings)
                inputs = {'z': next_input , 'a': actions}
                mems = outputs.mems
            
                if i%k == 0 and self.regularization_k_pred:
                    labels_embeddings_reg = rearrange(labels_embeddings[:, i:], 'b t o -> (b t) o')
                    embeds = outputs.embeddings[:, :-1] if i < 16 else outputs.embeddings[:, :-2]
                    embeds_observations_reg = rearrange(embeds, 'b t o -> (b t) o') 
                    loss_pred_k = F.mse_loss(embeds_observations_reg, labels_embeddings_reg)
                    
                    
            if self.regularization_post_quant:
                pred_slots = outputs.embeddings[:, :-1]
                decoder = tokenizer.decode.eval()
                decoded_features = decoder(pred_slots)
                #post_quant_z = tokenizer.decode(rearrange(tokenizer.embedding(tokens), 'b (l t) e -> (b l) e t', l = 19).reshape(*post_quant_z_gt.shape).contiguous())
                loss_post_quant = F.mse_loss(sam_features, decoded_features)
                loss.loss_total += loss_post_quant 
                loss.intermediate_losses['loss_reg_post_quant'] = loss_post_quant 

            
            
         

        elif self.config.model == 'irisXL-discrete' and not self.regularization:
            with torch.no_grad():
                obs_encoded = tokenizer.encode(batch['observations'], should_preprocess=True)  # (B, L, K)
                obs_tokens, obs_zq = obs_encoded.tokens, obs_encoded.z_quantized
               
            tokens = {'z': obs_tokens, 'a': batch['actions']}
            inputs = {name: mod(tokens[name]) for name, mod in self.embedder.items()}


            outputs = self(inputs, stop_mask=batch['ends'], tgt_length=obs_tokens.shape[1], embedding_input=self.embedding_input)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations.reshape(-1))
            loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
            loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

            loss = LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

            if self.embedding_input:
                embeddings = outputs.embeddings[:, :-1]
                loss_embedding = F.mse_loss(embeddings, rearrange(inputs['z'], 'b s t o -> b (s t) o')[:, 1:])
                loss.loss_total += loss_embedding
                loss.intermediate_losses['loss_embeddings'] = loss_embedding    
       
            
        elif self.config.model == 'irisXL-discrete' and self.regularization:
            with torch.no_grad():
                obs_encoded = tokenizer.encode(batch['observations'], should_preprocess=True)  # (B, L, K)
                obs_tokens, obs_zq = obs_encoded.tokens, obs_encoded.z_quantized
                if self.regularization_post_quant:
                    post_quant_z_gt = tokenizer.get_post_zq(obs_zq[:,1:])
            
            tokens = {'z': obs_tokens, 'a': batch['actions']}
            inputs = {name: mod(tokens[name]) for name, mod in self.embedder.items()}
            mems = None
            actions = self.embedder['a'](batch['actions'])

            for i in range(self.config.tokens_per_block):  
                outputs = self(inputs, mems=mems, stop_mask=batch['ends'], tgt_length=obs_tokens.shape[1], embedding_input=(self.embedding_input or self.regularization_embeddings), generation=False)

                if i == 0:
                    labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])
                    logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
                    loss_obs = F.cross_entropy(logits_observations, labels_observations.reshape(-1))
                    loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
                    loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
                    if self.embedding_input or self.regularization_embeddings:
                        embeddings = outputs.embeddings[:, :-1]
                        labels_embeddings = rearrange(inputs['z'], 'b s t o -> b (s t) o')[:, 1:]
                        loss_embedding = F.mse_loss(embeddings, labels_embeddings)

                next_input = outputs.output_sequence
                next_input[:, self.head_observations.compute_slice(num_steps=next_input.shape[1], prev_steps=i)] = self.embedder['z'](Categorical(logits=outputs.logits_observations).sample())
                inputs = {'z': next_input , 'a': actions}
                mems = outputs.mems
                
           
                if self.regularization_tokens:
                    labels_observations_reg = labels_observations[:, i:].reshape(-1)
                    logits = outputs.logits_observations[:, :-1] if i < 16 else outputs.logits_observations[:, :-2]
                    logits_observations_reg = rearrange(logits, 'b t o -> (b t) o')
                    self.loss_iter.append(F.cross_entropy(logits_observations_reg, labels_observations_reg).reshape(-1))

                if self.regularization_embeddings:
                    labels_embeddings_reg = rearrange(labels_embeddings[:, i:], 'b t o -> (b t) o')
                    embeds = outputs.embeddings[:, :-1] if i < 16 else outputs.embeddings[:, :-2]
                    embeds_observations_reg = rearrange(embeds, 'b t o -> (b t) o') 
                    self.loss_iter_embeds.append(F.mse_loss(embeds_observations_reg, labels_embeddings_reg).reshape(-1))

            loss = LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

            if self.embedding_input:
                loss.loss_total += loss_embedding
                loss.intermediate_losses['loss_embeddings'] = loss_embedding    

            if self.regularization_post_quant:
                tokens = Categorical(logits=outputs.logits_observations[:,:-1]).sample()
                post_quant_z = tokenizer.post_quant_conv(rearrange(tokenizer.embedding(tokens), 'b (l t) e -> (b l) e t', l = 19).reshape(*post_quant_z_gt.shape).contiguous())
                loss_post_quant = F.mse_loss(post_quant_z, post_quant_z_gt)
                loss.loss_total += loss_post_quant 
                loss.intermediate_losses['loss_reg_post_quant'] = loss_post_quant 
                
            if self.regularization_embeddings:
                loss_reg_embeds = torch.cat(self.loss_iter_embeds, dim=0).mean()
                loss.loss_total += loss_reg_embeds
                loss.intermediate_losses['loss_reg_embeds'] = loss_reg_embeds
                self.loss_iter_embeds = []
                
            if self.regularization_tokens:
                loss_reg_tokens = torch.cat(self.loss_iter, dim=0).mean()
                loss.loss_total += loss_reg_tokens
                loss.intermediate_losses['loss_reg_tokens'] = loss_reg_tokens
                self.loss_iter = []
            
        return loss

            
    

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations, labels_rewards.reshape(-1), labels_ends.reshape(-1)

    def compute_labels_continuos_world_model(self, embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        end_positions = mask_fill.repeat_interleave(embeddings.shape[2], dim=1) if embeddings.dim() == 4 else mask_fill.repeat_interleave(embeddings.shape[1], dim=1) 
        end_positions = end_positions.unsqueeze(2).repeat(1,1,embeddings.shape[-1])[:,1:]
        return end_positions, labels_rewards.reshape(-1), labels_ends.reshape(-1)

    def compute_labels_OC_world_model(self, embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        end_positions = mask_fill.repeat_interleave(embeddings.shape[2], dim=1) if embeddings.dim() == 4 else mask_fill.repeat_interleave(embeddings.shape[1], dim=1) 
        end_positions = end_positions.unsqueeze(2).repeat(1,1,embeddings.shape[-1])[:,self.config.tokens_per_block:]
        return end_positions, labels_rewards.reshape(-1), labels_ends.reshape(-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.unsqueeze(0)
        return self.pe[:, :x.size(1)].squeeze(0)


class OCWorldModel(WorldModel):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__(obs_vocab_size, act_vocab_size, config)

        self.tokens_per_block = config.tokens_per_block
        self.max_blocks = config.max_blocks
        self.spatial_pos_emb = PositionalEmbedding(config.tokens_per_block, config.embed_dim)
        # self.temporal_pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.ref_count = torch.zeros(obs_vocab_size)
        self.ref_count_log = []
        self.save_after = 25
        self.save_count_every = 25

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        spatial_emb = self.spatial_pos_emb(torch.arange(self.tokens_per_block, device=tokens.device)).repeat(self.max_blocks, 1)[:num_steps]
        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device)) #+ spatial_emb

        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        if x.requires_grad:
            logits = rearrange(logits_observations[:, :-1], 'b t o -> (b t) o')
            tokens = logits.argmax(dim=-1)
            tokens_onehot = F.one_hot(tokens, num_classes=self.obs_vocab_size).float()
            self.ref_count += tokens_onehot.sum(dim=0).detach().cpu()

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)
    
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)

        if self.act_vocab_size == 0:
            tokens = rearrange(obs_tokens, 'b l k -> b (l k)')
        else:
            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
            tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        bs = tokens.size(0)

        outputs = self(tokens)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        # additional reconstruction loss
        # next_observations = batch['observations'][:, 1:]
        # next_logits_observations = rearrange(outputs.logits_observations[:, tokenizer.num_slots*tokenizer.tokens_per_slot:], 'b t o -> (b t) o')
        # z_q = tokenizer.decode_logits(next_logits_observations)
        # z_q = rearrange(z_q, '(b l k t) e -> b l e k t', b=bs, k=tokenizer.num_slots, t=tokenizer.tokens_per_slot)
        # reconstructions = tokenizer.decode(z_q)
        # reconstruction_loss = torch.pow(next_observations - reconstructions, 2).mean()

        if self.act_vocab_size == 0:
            return LossWithIntermediateLosses(loss_obs=loss_obs)
        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)
        # return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends, reconstruction_loss=reconstruction_loss)

    def _reset_count(self):
        self.ref_count = torch.zeros(self.obs_vocab_size)
        self.ref_count_log = []
    
    def plot_count(self, epoch, save_dir):
        self.ref_count_log.append(self.ref_count.numpy())
        if epoch >= self.save_after and epoch % self.save_count_every == 0:
            plt.figure(figsize=(10,1))
            plt.imshow(self.ref_count_log)
            plt.tight_layout()
            plt.savefig(save_dir / f"ref_count_wm_{epoch}.png")
            plt.close()
            self._reset_count()

