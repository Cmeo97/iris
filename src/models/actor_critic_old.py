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

from dataset import Batch
from envs.world_model_env import WorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel, OCWorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
}


    
class MLP(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_list, 
        out_dim, 
        std=np.sqrt(2),
        bias_const=0.0,
        activation='tanh'):

        #ctor: {layers: 5, units: 1024, act: silu, norm: layer, minstd: 0.1, maxstd: 1.0, outscale: 1.0, outnorm: False, unimix: 0.01, inputs: [deter, stoch], winit: normal, fan: avg, symlog_inputs: False}
        #critic: {layers: 5, units: 1024, act: silu, norm: layer, dist: symlog_disc, outscale: 0.0, outnorm: False, inputs: [deter, stoch], winit: normal, fan: avg, bins: 255, symlog_inputs: False}
  
        super().__init__()
        assert activation in ['relu', 'tanh', 'gelu', 'silu']
        self.layer_norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.layers.append(layer_init(nn.Linear(in_dim, hidden_list[0])))
        self.layers.append(nn.LayerNorm(512))
        self.layers.append(activations[activation])
        
        for i in range(len(hidden_list)-1):
            self.layers.append(layer_init(nn.Linear(hidden_list[i], hidden_list[i+1])))
            self.layers.append(nn.LayerNorm(512))
            self.layers.append(activations[activation])
        self.layers.append(layer_init(nn.Linear(hidden_list[-1],out_dim), std=std, bias_const=bias_const))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
    

@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, use_original_obs: bool = False) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.mha = nn.MultiheadAttention(512, 4, 0.05, batch_first=True)
        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None
    
        self.critic = MLP(512, [512], 1, activation='silu')
        
        self.actor = MLP(512, [512], act_vocab_size, activation='silu')
        self.device = self.actor.layers[0].weight.device
       

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.critic.layers[0].weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 4 and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    @property
    def device(self):
        return self.critic.layers[0].weight.device

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        #assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        #assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs
        
        x = self.mha(self.hx, x, x)
        
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
        wm_env = WorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        burnin_observations = tokenizer.encode_logits(tokenizer.encode(batch['observations']).z).reshape(-1, 20, 4, 512) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])

        obs = wm_env.reset_from_initial_observations(initial_observations[:, -1])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )

    @torch.no_grad()
    def rollout(self, batch: Batch, tokenizer: Tokenizer, world_model: OCWorldModel, burn_in: int, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert observations.ndim == 5 and observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_rewards = []
        all_ends = []
        all_observations = []
        all_colors = []
        all_masks = []

        obs = wm_env.reset_from_initial_observations(observations[:, 0])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Rollout', file=sys.stdout):

            action_token = rearrange(batch['actions'][:, k], 'b -> b 1 1')
            obs, color, mask, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=True, should_return_slots=True)

            all_actions.append(action_token)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))
            all_observations.append(obs)
            all_colors.append(color)
            all_masks.append(mask)

        all_observations = torch.stack(all_observations, dim=1) # (B, T, C, H, W)
        all_colors = torch.stack(all_colors, dim=1)
        all_masks = torch.stack(all_masks, dim=1)
        print(all_observations.shape, all_colors.shape, all_masks.shape)

        return all_observations, all_colors, all_masks
