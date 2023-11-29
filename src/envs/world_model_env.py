import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None, model: Optional[str] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.model = model
        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

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
        self.obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = self.obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens
        if self.model == 'iris':
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)
        return self.decode_obs_tokens(), self.obs_tokens
    

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
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], mems: Optional[List] = None , should_predict_next_obs: bool = True, stop_mask: Optional[bool] = None) -> None:
        #assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        if self.model == 'iris':
            if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
                _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

            for k in range(num_passes):  # assumption that there is only one action token.

                outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm, mems=mems, stop_mask=stop_mask, tgt_length=1)
                output_sequence.append(outputs_wm.output_sequence)
                if mems is not None:
                    mems = outputs_wm.mems

                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

                if k < self.num_observations_tokens:
                    token = Categorical(logits=outputs_wm.logits_observations).sample()
                    obs_tokens.append(token)

        elif self.model == 'irisXL-discrete':
            
            
            #h = self.refresh_embeddings_with_initial_obs_tokens(obs_tokens=self.obs_tokens, stop_mask=stop_mask)
            h = self.world_model.embedder['a'](token)
            
            for k in range(num_passes):  # assumption that there is only one action token.

                outputs_wm = self.world_model({'z': token, 'h': h}, mems=mems, stop_mask=stop_mask, tgt_length=1, embedding_input=self.world_model.embedding_input) if self.world_model.embedding_input else self.world_model({'z': token}, mems=mems, stop_mask=stop_mask, tgt_length=1) 
                #output_sequence.append(outputs_wm.output_sequence)
                if mems is not None:
                    mems = outputs_wm.mems

                if k == 0:
                    reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                    done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

                if k < self.num_observations_tokens:
                    token = Categorical(logits=outputs_wm.logits_observations).sample()
                    h = outputs_wm.output_sequence
                    obs_tokens.append(token)

        #output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, mems, self.obs_tokens

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
