import random
from typing import List, Optional, Union
import sys
import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision
from collections import deque
from tqdm import tqdm

class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None, model: Optional[str] = None, actor: Optional[torch.nn.Module] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.model = model
        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens, self.embedded_tokens = None, None, None, None
        self.parallel_imagination = True
        self.env = env
        self.first_obs_token = None
        if actor is not None:
            self.actor = actor.to(self.device).eval()
        self.slot_based = tokenizer.slot_based

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:

        encodings = self.tokenizer.encode(observations, should_preprocess=True)   # (B, C, H, W) -> (B, K)
        self.obs_tokens = encodings.z if self.slot_based else encodings.tokens 
        init_variables = self.obs_tokens if self.model != 'irisXL-continuos' else rearrange(encodings.z_quantized, 't e o p -> t (o p) e')
        if self.model == 'OC-irisXL':
            init_variables = self.obs_tokens
            num_observations_tokens = self.obs_tokens.shape[1] 
        else:
            _, num_observations_tokens = self.obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens
        if self.model == 'iris':
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)
        #self.obs_logits = self.tokenizer.encode_logits(outputs.z) # need to use logits instead of tokens when using dVAE, from nakano_dev
        output = self.decode_obs_tokens(observations) if self.slot_based else self.decode_obs_tokens()
        return output, init_variables
    
    

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)
    
    #@torch.no_grad()
    #def refresh_embeddings_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor, stop_mask: Optional[bool] = None) -> torch.FloatTensor:
    #    embeddings = self.world_model.transformer.initialize_embeddings({'z': self.world_model.embedder['z'](obs_tokens.unsqueeze(1))}, tgt_length=16, stop_mask=stop_mask)
    #    return embeddings   # (B, K, E)
    
    @torch.no_grad()
    def parallel_step(self, initial_observations: Optional[torch.Tensor], horizon: Union[int, np.ndarray, torch.LongTensor], mems: Optional[List] = None , embedder: Optional[torch.nn.Module] = None, show_pbar: bool = False) -> None:
       
        all_observations = []
        all_tokens_observations = []
        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        output_sequence, tokens_sequences, lengths = deque(), deque(), deque()
        self.embedded_tokens = None
        obs, obs_tokens = self.reset_from_initial_observations(initial_observations[:, -1])
        stop_mask = torch.zeros((obs_tokens.shape[0], 1), device=obs_tokens.device) # Stop Mask initialization
        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1
        for k in tqdm(range(horizon*num_passes), disable=not show_pbar, desc='Imagination', file=sys.stdout):

                all_observations.append(obs)
                all_tokens_observations.append(obs_tokens)

                outputs_ac =  self.actor(embedder(obs_tokens)) if self.actor.latent_actor else self.actor(obs) 
                action_token = Categorical(logits=outputs_ac.logits_actions).sample()
                
                action_token = action_token.clone().detach() if isinstance(action_token, torch.Tensor) else torch.tensor(action_token, dtype=torch.long)
                action_token = token.reshape(-1, 1).to(self.device)  # (B, 1)
                h_a = self.world_model.embedder['a'](token)
                '---------------------------------------------------------------------------------------------------'
                
                #obs, reward, done, mems, obs_tokens = step(action_token, mems, should_predict_next_obs=(k < horizon - 1), stop_mask=stop_mask)
                
                should_predict_next_obs=(k < horizon - 1)
                num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1
                if k == 0:
                    outputs_wm = self.world_model({'z': action_token, 'h': h_a}, mems=mems, stop_mask=stop_mask, tgt_length=1, embedding_input=self.world_model.embedding_input) if self.world_model.embedding_input else self.world_model({'z': action_token}, mems=mems, stop_mask=stop_mask, tgt_length=1) 
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)
                    first_token = Categorical(logits=outputs_wm.logits_observations).sample()
                    output_sequence.appendleft(outputs_wm.output_sequence)
                    tokens_sequences.appendleft(first_token)
                    tokens_cat = self.first_obs_token = first_token       # (B, K)
                    output_sequence_cat = self.first_sequence = outputs_wm.output_sequence
                elif k%num_passes == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)
                    
                else:

                    outputs_wm = self.world_model({'z': tokens_cat, 'h': output_sequence_cat}, mems=mems, stop_mask=stop_mask, tgt_length=1, embedding_input=self.world_model.embedding_input) if self.world_model.embedding_input else self.world_model({'z': tokens_cat}, mems=mems, stop_mask=stop_mask, tgt_length=1) 
                    #output_sequence.append(outputs_wm.output_sequence)
                    mems = outputs_wm.mems
                    token = Categorical(logits=outputs_wm.logits_observations).sample()
                    output_sequence_cat = torch.cat([self.first_sequence, outputs_wm.output_sequence], dim=1)   # (B, 1 + K, E)
                    tokens_cat = torch.cat([self.first_obs_token, token], dim=1)        # (B, K)
                    #embeddings_sequence.appendleft(output_sequence_cat)
                    tokens_sequences.appendleft(tokens_cat)
                    lengths.appendleft(tokens_cat.shape[1])
                
                
                '------------------------------------------------------------------------------------------------------'
                all_actions.append(action_token)
                all_logits_actions.append(outputs_ac.logits_actions)
                all_values.append(outputs_ac.means_values)
                all_rewards.append(torch.tensor(reward).reshape(-1, 1))
                all_ends.append(torch.tensor(done).reshape(-1, 1))
                stop_mask = torch.stack(all_ends, dim=1).squeeze(2) if len(all_ends) > 1 else torch.tensor(done).reshape(-1, 1)
                
                
                
            

    @torch.no_grad()
    def step(self, tokens: Union[int, np.ndarray, torch.LongTensor], mems: Optional[List] = None , should_predict_next_obs: bool = True, stop_mask: Optional[bool] = None, should_return_slots: Optional[bool] = False, obs: Optional[torch.LongTensor] = None) -> None:
       
        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1
        output_sequence, obs_tokens= [], []
        if isinstance(tokens, dict):
            tokens['a'] = tokens['a'].clone().detach().reshape(-1, 1)
        tokens = tokens.clone().detach().reshape(-1, 1).to(self.device) if isinstance(tokens, torch.Tensor) else {name: mod(tokens[name]).unsqueeze(1).to(self.device) for name, mod in self.world_model.embedder.items()} if isinstance(tokens, dict) else torch.tensor(tokens, dtype=torch.long).reshape(-1, 1).to(self.device)
        
        if self.model == 'iris':
            if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
                _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

            for k in range(num_passes):  # assumption that there is only one action token.

                outputs_wm = self.world_model(tokens, past_keys_values=self.keys_values_wm, mems=mems, stop_mask=stop_mask, tgt_length=1)
                output_sequence.append(outputs_wm.output_sequence)
                
                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

                if k < self.num_observations_tokens:
                    tokens = Categorical(logits=outputs_wm.logits_observations).sample()
                    obs_tokens.append(tokens)

            self.obs_tokens = torch.cat(obs_tokens, dim=1)  
            obs = self.decode_obs_tokens() if should_predict_next_obs else None
            
            return obs, reward, done, mems, self.obs_tokens

        elif self.model == 'irisXL-discrete':
            h = self.world_model.embedder['a'](tokens)
            for k in range(num_passes):  # assumption that there is only one action tokens.

                outputs_wm = self.world_model({'z': tokens, 'h': h}, mems=mems, stop_mask=stop_mask, tgt_length=1, embedding_input=self.world_model.embedding_input, generation=True) 
                if mems is not None:
                    mems = outputs_wm.mems

                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

                if k < self.num_observations_tokens:
                    tokens = Categorical(logits=outputs_wm.logits_observations).sample()
                    h = outputs_wm.output_sequence
                    obs_tokens.append(tokens)

            self.obs_tokens = torch.cat(obs_tokens, dim=1)     # (B, 1 + K, E)
            obs = self.decode_obs_tokens() if should_predict_next_obs else  None
            return obs, reward, done, mems, self.obs_tokens

        elif self.model == 'irisXL-continuos':
            
            h = self.world_model.embedder['a'](tokens)
            for k in range(num_passes):  # assumption that there is only one action token.
                outputs_wm = self.world_model({'z': h}, mems=mems, stop_mask=stop_mask, tgt_length=1, generation=True) 
                
                mems = outputs_wm.mems
                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)
                if k < self.num_observations_tokens:
                    h = outputs_wm.embeddings
                    output_sequence.append(outputs_wm.embeddings)

            self.embedded_tokens = self.obs_tokens = torch.stack(output_sequence, dim=1) # (B, 1 + K, E)
      
            obs = self.decode_obs_tokens() if should_predict_next_obs else None
            
            return obs, reward, done, mems, self.embedded_tokens
            
        elif self.model == 'OC-irisXL':
            num_passes = self.num_observations_tokens if should_predict_next_obs else 1
            
            outputs_wm = self.world_model(tokens, mems=mems, stop_mask=stop_mask, tgt_length=1, embedding_input=True, generation=True) 
            
            mems = outputs_wm.mems
            reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
            done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)
            output_sequence = outputs_wm.embeddings

            self.embedded_tokens = self.obs_tokens = output_sequence # (B, 1 + K, E)
      
            output = self.decode_obs_tokens(obs) if obs is not None else self.decode_obs_tokens() if should_predict_next_obs else None
            if should_return_slots:
                obs, color, mask = output 
                return obs, reward, done, mems, color, mask, self.embedded_tokens 
            else:
                obs, _, _ = output
                return obs, reward, done, mems, self.embedded_tokens
      
      

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self, observations=None) -> List[Image.Image]:
        if self.slot_based:
            rec, color, mask = self.tokenizer.decode_slots(self.obs_tokens, observations)         # (B, C, H, W)
            return torch.clamp(rec, 0, 1), color, mask
        else:
            embedded_tokens = self.tokenizer.embedding(self.obs_tokens) if self.embedded_tokens == None else self.embedded_tokens.squeeze(2)    # (B, K, E)
            z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
            rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
            return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
