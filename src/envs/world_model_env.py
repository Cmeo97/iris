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

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_logits, self.obs_tokens, self._num_observations_tokens = None, None, None, None

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
        outputs = self.tokenizer.encode(observations, should_preprocess=True)
        self.obs_tokens = outputs.z
        #obs_tokens = outputs.tokens    # (B, C, H, W) -> (B, K)
        num_observations_tokens = self.obs_tokens.shape[1]
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)
        
        out = self.tokenizer.encode_logits(outputs.z)

        return out

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True, should_return_slots: bool = False) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_logits, obs_tokens = [], [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.

            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens:
                logits = outputs_wm.logits_observations
                if self.tokenizer.slot_based:
                    # token = torch.argmax(outputs_wm.logits_observations, dim=-1)
                    token = outputs_wm.logits_observations #Categorical(logits=outputs_wm.logits_observations).sample()
                else:
                    token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_logits.append(logits)
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_logits = torch.cat(obs_logits, dim=1)        # (B, K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        if self.tokenizer.slot_based:
            obs, color, mask = self.decode_obs_tokens() if should_predict_next_obs else None
            if should_return_slots:
                return self.obs_tokens, color, mask, reward, done, None
            else:
                return self.obs_tokens, reward, done, None
        else:
            obs = self.decode_obs_tokens() if should_predict_next_obs else None
            return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        if self.tokenizer.slot_based:
            embedded_tokens = self.tokenizer.decode_logits(self.obs_logits)     # (B, K, E)
            z = rearrange(embedded_tokens, 'b (k t) e -> b e k t', k=self.tokenizer.num_slots, t=self.tokenizer.tokens_per_slot)
            rec, color, mask = self.tokenizer.decode_slots(z, should_postprocess=True)         # (B, C, H, W)
            return torch.clamp(rec, 0, 1), color, mask
        else:
            embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
            z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
            rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
            return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
