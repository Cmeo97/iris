from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .slicer import Embedder
from models.tokenizer import Tokenizer
from dataset import Batch
from envs.world_model_env import WorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses
from einops import rearrange


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    tokens_observations: torch.FloatTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, use_original_obs: bool = False, model: str = 'iris', latent_actor: bool = False) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.latent_actor = latent_actor
        self.model = model
        self.lstm_dim = 512

        if not latent_actor:
            self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
            self.maxp1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.maxp2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
            self.maxp3 = nn.MaxPool2d(2, 2)
            self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.maxp4 = nn.MaxPool2d(2, 2)
            self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        else:
            self.lstm = nn.LSTMCell(256, self.lstm_dim)

        self.hx, self.cx = None, None
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_vocab_size)

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.critic_linear.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            #assert burnin_observations.ndim == 5 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])


    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None, collecting: bool = False, tokenizer: Tokenizer = None, wm: WorldModelEnv = None) -> ActorCriticOutput:
        if not self.latent_actor:
            assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
            assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
            assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
            x = inputs[mask_padding] if mask_padding is not None else inputs

            x = x.mul(2).sub(1)
            x = F.relu(self.maxp1(self.conv1(x)))
            x = F.relu(self.maxp2(self.conv2(x)))
            x = F.relu(self.maxp3(self.conv3(x)))
            x = F.relu(self.maxp4(self.conv4(x)))
            x = torch.flatten(x, start_dim=1)
        elif collecting:
            x = wm.embedder.embedding_tables[1](tokenizer.encode(inputs).tokens)
        else:
            x = inputs[mask_padding] if mask_padding is not None else inputs


        if x.ndim > 2: # when using latent rapresentations for actor
            for i in range(x.shape[1]):
                if mask_padding is None:
                    self.hx, self.cx = self.lstm(x[:, i], (self.hx, self.cx))
                else:
                    self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x[:, i], (self.hx[mask_padding], self.cx[mask_padding]))
        else:
            if mask_padding is None:
                self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
            else:
                self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))


        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        assert not self.use_original_obs
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device, model=self.model)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []
        all_tokens_observations = []

        
        if not self.latent_actor:
            burnin_observations = torch.clamp(tokenizer.encode_decode(initial_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0, 1) if initial_observations.size(1) > 1 else None
            self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])
        else:
            with torch.no_grad():
                burnin_tokens_obs = tokenizer.encode(initial_observations[:, :-1], should_preprocess=True).tokens if initial_observations.size(1) > 1 else None
                if isinstance(wm_env.world_model.embedder, Embedder): # vanilla iris is used 
                    b, l, t = burnin_tokens_obs.shape
                    burnin_tokens_obs = rearrange(burnin_tokens_obs, 'b l t -> b (l t)') 
                    burnin_encodings = wm_env.world_model.embedder.embedding_tables[1](burnin_tokens_obs).reshape(b, l, t, -1)
                else:
                    b, l, t = burnin_tokens_obs.shape
                    burnin_encodings = wm_env.world_model.embedder['z'](burnin_tokens_obs)
                self.reset(n=initial_observations.size(0), burnin_observations=burnin_encodings, mask_padding=mask_padding[:, :-1])

            self.reset(n=initial_observations.size(0), burnin_observations=burnin_encodings, mask_padding=mask_padding[:, :-1])
        if isinstance(wm_env.world_model.embedder, nn.ModuleDict):
            embedder = wm_env.world_model.embedder['z'].eval()
        else:
            embedder = wm_env.world_model.embedder.embedding_tables[1].eval()
        obs, obs_tokens = wm_env.reset_from_initial_observations(initial_observations[:, -1])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):
        
            all_observations.append(obs)
            all_tokens_observations.append(obs_tokens)

            outputs_ac =  self(embedder(obs_tokens)) if self.latent_actor else self(obs) 
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            obs, reward, done, _, obs_tokens = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()
        return ImagineOutput(
        observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
        tokens_observations=torch.stack(all_tokens_observations, dim=1),   
        actions=torch.cat(all_actions, dim=1),                                  # (B, T)
        logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
        values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
        rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
        ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )

        
        
    
  