from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import ObsTokenizer, ActTokenizer
from models.world_model import WorldModel
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(
        self, obs_tokenizer: ObsTokenizer, act_tokenizer: ActTokenizer, world_model: WorldModel, actor_critic: ActorCritic
    ):
        super().__init__()
        self.obs_tokenizer = obs_tokenizer
        self.act_tokenizer = act_tokenizer        
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(
        self,
        path_to_checkpoint: Path,
        device: torch.device,
        load_tokenizer: bool = True,
        load_world_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.obs_tokenizer.load_state_dict(
                extract_state_dict(agent_state_dict, "obs_tokenizer")
            )
            self.act_tokenizer.load_state_dict(
                extract_state_dict(agent_state_dict, "act_tokenizer")
            )
        if load_world_model:
            self.world_model.load_state_dict(
                extract_state_dict(agent_state_dict, "world_model")
            )
        if load_actor_critic:
            self.actor_critic.load_state_dict(
                extract_state_dict(agent_state_dict, "actor_critic")
            )

    def act(
        self,
        obs: torch.FloatTensor,
        should_sample: bool = True,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        input_ac = (
            obs
            if self.actor_critic.use_original_obs
            else torch.clamp(
                self.obs_tokenizer.encode_decode(
                    obs, should_preprocess=True, should_postprocess=True
                ),
                0,
                1,
            )
        )

        dist_actions = self.actor_critic(input_ac).dist_actions  # [:, -1] / temperature
        act = dist_actions.sample()
        
        # if should_sample else
        # else dist_actions.argmax(dim=-1)

        return act
