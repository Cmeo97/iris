from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
# from collector import VIPERCollector as Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch, make_reconstructions_with_slots_from_batch, save_image_with_slots
from models.actor_critic import ActorCritic

from models.world_model import WorldModel, OCWorldModel
from utils import configure_optimizer, EpisodeDirManager, set_seed, linear_warmup_exp_decay
from torchvision.utils import save_image



class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        ### Slot-based model ###
        try:
            self.slot_based = cfg.common.slot_based
            print("The model is slot-based")
        except:
            self.slot_based = False

        ### Upscaling ###
        if cfg.tokenizer.encoder.config.resolution > 64:
            self.upscale = True
            print("The model is upscaling")
        else:
            self.upscale = False

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            if cfg.common.use_pretrained:
                shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "saved_models"), dst="./saved_models")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_dataset._check_upscale(self.upscale)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_dataset._check_upscale(self.upscale)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions, model=cfg.world_model.model, tokenizer=tokenizer)
        image_decoder = instantiate(cfg.image_decoder)

        self.agent = Agent(tokenizer, world_model, actor_critic, image_decoder).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        tokenizer_lr = cfg.training.learning_rate if "learning_rate" not in cfg.training.tokenizer else cfg.training.tokenizer.learning_rate
        world_model_lr = cfg.training.learning_rate if "learning_rate" not in cfg.training.world_model else cfg.training.world_model.learning_rate
        actor_critic_lr = cfg.training.learning_rate if "learning_rate" not in cfg.training.actor_critic else cfg.training.actor_critic.learning_rate
        image_decoder_lr = cfg.training.learning_rate if "learning_rate" not in cfg.training.image_decoder else cfg.training.image_decoder.learning_rate
        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=tokenizer_lr)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, world_model_lr, cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=actor_critic_lr)
        self.optimizer_image_decoder = torch.optim.Adam(self.agent.image_decoder.parameters(), lr=image_decoder_lr)

        # self.scheduler_tokenizer = LambdaLR(self.optimizer_tokenizer, lr_lambda=linear_warmup_exp_decay(1000, 0.5, 10000))
        self.scheduler_tokenizer = LambdaLR(self.optimizer_tokenizer, lr_lambda=linear_warmup_exp_decay(10000, 0.5, 100000))
        self.scheduler_world_model = None
        self.scheduler_actor_critic = None
        self.scheduler_image_decoder = None

        self.use_amp = cfg.common.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.common.use_amp)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self) -> None:
        # for name, param in self.agent.tokenizer.quantizer.named_parameters():
        #     print(name)
        #     param.requires_grad = False

        if self.cfg.common.use_pretrained:
            self.agent.load(Path(hydra.utils.get_original_cwd()) / 'saved_models' / 'dinosaur_sam.pt', device=self.device, strict=False)

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):
            # if epoch > 200:
            #     self.agent.tokenizer.slot_attn.prior_class = "gru"
            #     for name, param in self.agent.tokenizer.quantizer.named_parameters():
            #         param.requires_grad = True
            #     self.agent.tokenizer.tau = 1.0

            # if epoch == 100:
            #     episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=self.cfg.collection.train.num_episodes_to_save)

            #     def create_env(cfg_env, num_envs):
            #         env_fn = partial(instantiate, config=cfg_env)
            #         return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

            #     train_env = create_env(self.cfg.env.train, self.cfg.collection.train.num_envs)
            #     self.train_dataset = instantiate(self.cfg.datasets.train)
            #     self.train_dataset._check_upscale(self.upscale)
            #     self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)
            #     for _ in tqdm(range(25), desc="Recollecting dataset"):
            #         self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                self.test_dataset.clear()
                to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
                to_log += self.eval_agent(epoch)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

            if self.slot_based:
                self.agent.tokenizer.quantizer.plot_count(epoch, self.reconstructions_dir) # for debugging
                # self.agent.tokenizer.quantizer.plot_slot_dist(epoch, self.reconstructions_dir) # for debugging
                # self.agent.tokenizer.quantizer.plot_codebook(epoch, self.reconstructions_dir) # for debugging
                self.agent.tokenizer.set_tau()
                #self.agent.world_model.plot_count(epoch, self.reconstructions_dir) # for debugging

        self.finish()

    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic, metrics_image_decoder = {}, {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic
        cfg_image_decoder = self.cfg.training.image_decoder
        
        if self.cfg.world_model.regularization_post_quant:
            cfg_world_model.batch_num_samples = 16
            

        w = self.cfg.training.sampling_weights

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer, self.scheduler_tokenizer, sequence_length=4, sample_from_start=True, sampling_weights=w, **cfg_tokenizer)

        self.agent.tokenizer.eval()

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.train_component(self.agent.world_model, self.optimizer_world_model, self.scheduler_world_model, sequence_length=self.cfg.common.sequence_length, sample_from_start=True, sampling_weights=w, tokenizer=self.agent.tokenizer, **cfg_world_model)
        self.agent.world_model.eval()

        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic, self.scheduler_actor_critic, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, sampling_weights=w, tokenizer=self.agent.tokenizer, world_model=self.agent.world_model, **cfg_actor_critic)
        self.agent.actor_critic.eval()

        if epoch > cfg_image_decoder.start_after_epochs:
            metrics_image_decoder = self.train_component(self.agent.image_decoder, self.optimizer_image_decoder, self.scheduler_image_decoder, sequence_length=4, sample_from_start=True, sampling_weights=w, **cfg_image_decoder)

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic, **metrics_image_decoder}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, None]], steps_per_epoch: int, batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float], sequence_length: int, sampling_weights: Optional[Tuple[float]], sample_from_start: bool, **kwargs_loss: Any) -> Dict[str, float]:
        if not isinstance(max_grad_norm, float):
            max_grad_norm = None
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sampling_weights, sample_from_start)
                batch = self._to_device(batch)

                with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                    if str(component) == 'image_decoder':
                        with torch.no_grad():
                            z, inits, z_vit, feat_recon = self.agent.tokenizer(batch['observations'], should_preprocess=True)
                        losses = component.compute_loss(batch, z, **kwargs_loss) / grad_acc_steps
                    else:
                        losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                    loss_total_step = losses.loss_total
                    loss_total_epoch += loss_total_step.item() / steps_per_epoch

                # loss_total_step.backward()
                self.scaler.scale(loss_total_step).backward()

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            # optimizer.step()
            self.scaler.step(optimizer)
            self.scaler.update()
            if scheduler is not None:
                scheduler.step()

        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic
        
        if self.cfg.world_model.regularization_post_quant:
            cfg_world_model.batch_num_samples = 16

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=self.cfg.common.sequence_length)


        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(self.agent.world_model, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer, **cfg_world_model)
            self.inspect_world_model(epoch)
        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=2, sequence_length=self.cfg.common.sequence_length))
            tr_batch = self._to_device(self.train_dataset.sample_batch(batch_num_samples=2, sequence_length=self.cfg.common.sequence_length))
            if self.slot_based:
                batch['observations'] = torch.cat([batch['observations'], tr_batch['observations']], dim=0)
                make_reconstructions_with_slots_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer, image_decoder=self.agent.image_decoder)
                self.inspect_world_model(epoch)
            else:
                make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                if str(component) == 'image_decoder':
                    z, inits, z_vit, feat_recon = self.agent.tokenizer(batch['observations'], should_preprocess=True)
                    losses = component.compute_loss(batch, z, **kwargs_loss)
                else:
                    losses = component.compute_loss(batch, **kwargs_loss)
                loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    #@torch.no_grad()
    #def inspect_world_model(self, epoch: int) -> None:
    #    batch = self.train_dataset.sample_batch(batch_num_samples=1, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
    #    recons, colors, masks = self.agent.actor_critic.rollout(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, burn_in=5, horizon=self.cfg.evaluation.actor_critic.horizon-5, show_pbar=True)
    #    save_image_with_slots(batch['observations'][:, 1:1+recons.shape[1]], recons, colors, masks, save_dir=self.reconstructions_dir, epoch=epoch, suffix='rollout')

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
            outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log
    
    
    @torch.no_grad()
    def inspect_world_model(self, epoch: int) -> None:
        batch = self.train_dataset.sample_batch(batch_num_samples=5, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
            recons = self.agent.actor_critic.rollout(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon-5, show_pbar=True)
        if self.agent.tokenizer.slot_based:
            recons, colors, masks = recons
        def save_sequence_image(observations, recons, save_dir, epoch, suffix='sample'):
            b, t, _, _, dim = observations.size()
            
            for i in range(b):
                obs = observations[i].cpu() # (t c h w)
                recon = recons[i].cpu() # (t c h w)

                full_plot = torch.cat([obs.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
                full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
                full_plot = full_plot.view(-1, 3, dim, dim)  # (H*W, 3, D, D)

                save_image(full_plot, save_dir / f'epoch_{epoch:03d}_{suffix}_{i:03d}.png', nrow=t)

        def save_image_with_slots(observations, recons, colors, masks, save_dir, epoch, suffix='sample'):
            b, t, _, h, w = observations.size()

            for i in range(b):
                obs = observations[i].cpu() # (t c h w)
                recon = recons[i].cpu() # (t c h w)

                full_plot = torch.cat([obs.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
                color = colors[i].cpu()
                mask = masks[i].repeat(1,1,3,1,1).cpu()
                subimage = color * mask
                full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
                full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
                full_plot = full_plot.view(-1, 3, h, w)  # (H*W, 3, D, D)

                save_image(full_plot, save_dir / f'epoch_{epoch:03d}_{suffix}_{i:03d}.png', nrow=t)

        if self.agent.tokenizer.slot_based:       
            save_image_with_slots(batch['observations'][:, 1:1+recons.shape[1]], recons, colors, masks, save_dir=self.reconstructions_dir, epoch=epoch, suffix='rollout')
        else:
            save_sequence_image(batch['observations'][:, 1:1+recons.shape[1]], recons, save_dir=self.reconstructions_dir, epoch=epoch, suffix='rollout')

    
      


    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()

    @torch.no_grad()
    def run_vis(self, try_slots: bool = False):
        from einops import rearrange
        from PIL import Image
        from torchvision.utils import save_image

        if try_slots:
            from make_reconstructions import reconstruct_through_tokenizer_with_slots as reconstruct_through_tokenizer
        else:
            from make_reconstructions import reconstruct_through_tokenizer

        num_samples = 10

        vis_dir = self.reconstructions_dir
        if self.cfg.common.resume:
            vis_dir = self.media_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True, parents=True)

        start_time = time.time()
        to_log = []

        to_log += self.train_collector.collect(self.agent, epoch=0, **self.cfg.collection.train.config)

        self.test_dataset.clear()
        to_log += self.test_collector.collect(self.agent, epoch=0, **self.cfg.collection.test.config)
        # to_log += self.eval_agent(epoch=0)

        batch = self.train_dataset.sample_batch(batch_num_samples=num_samples, sequence_length=20)
        for burn_in in [1, 5]:
            recons, _, _ = self.agent.actor_critic.rollout(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, burn_in=burn_in, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

            b, t, _, h, w = batch['observations'].size()
            for i in range(b):
                obs = batch['observations'][i].cpu() # (t c h w)
                recon = recons[i].cpu() # (t c h w)
                full_plot = torch.cat([obs.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
                full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
                full_plot = full_plot.view(-1, 3, h, w)  # (H*W, 3, D, D)

                save_image(full_plot, vis_dir / f'train_burn{burn_in}_sample_{i:03d}.png', nrow=t)

        #########
        # train_batch = self._to_device(self.train_dataset.sample_batch(batch_num_samples=num_samples, sequence_length=self.cfg.common.sequence_length))
        # test_batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=num_samples, sequence_length=self.cfg.common.sequence_length))

        # train_batch['observations'] = torch.cat([train_batch['observations'], train_batch['observations']], dim=0)
        # make_reconstructions_with_slots_from_batch(train_batch, save_dir=vis_dir, epoch=0, tokenizer=self.agent.tokenizer)

        #########

        # inputs = rearrange(train_batch['observations'], 'b t c h w -> (b t) c h w')
        # outputs = reconstruct_through_tokenizer(inputs, self.agent.tokenizer)
        # b, t, _, _, _ = train_batch['observations'].size()
        # if try_slots:
        #     recons, colors, masks = outputs
        #     recons = rearrange(recons, '(b t) c h w -> b t c h w', b=b, t=t)
        #     colors = rearrange(colors, '(b t) k c h w -> b t k c h w', b=b, t=t)
        #     masks = rearrange(masks, '(b t) k c h w -> b t k c h w', b=b, t=t)
        # else:
        #     recons = outputs
        #     recons = rearrange(recons, '(b t) c h w -> b t c h w', b=b, t=t)

        # for i in tqdm(range(num_samples)):
        #     train_obs = train_batch['observations'][i].cpu() # (t c h w)
        #     recon_obs = recons[i].cpu() # (t c h w)

        #     full_plot = torch.cat([train_obs.unsqueeze(1), recon_obs.unsqueeze(1)], dim=1) # (t 2 c h w)
        #     if try_slots:
        #         color = colors[i].cpu()
        #         mask = masks[i].cpu()
        #         subimage = color * mask
        #         mask = mask.repeat(1,1,3,1,1)
        #         full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
        #     full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
        #     full_plot = full_plot.view(-1, 3, 64, 64)  # (H*W, 3, D, D)

        #     save_image(full_plot, vis_dir / f'a_train_{i}.png', nrow=t)

        # inputs = rearrange(test_batch['observations'], 'b t c h w -> (b t) c h w')
        # outputs = reconstruct_through_tokenizer(inputs, self.agent.tokenizer)
        # b, t, _, _, _ = test_batch['observations'].size()
        # for i in tqdm(range(num_samples)):
        #     test_obs = test_batch['observations'][i].cpu()
        #     recon_obs = recons[i].cpu() # (t c h w)

        #     full_plot = torch.cat([test_obs.unsqueeze(1), recon_obs.unsqueeze(1)], dim=1) # (t 2 c h w)
        #     if try_slots:
        #         color = colors[i].cpu()
        #         mask = masks[i].cpu()
        #         subimage = color * mask
        #         mask = mask.repeat(1,1,3,1,1)
        #         full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
        #     full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
        #     full_plot = full_plot.view(-1, 3, 64, 64)  # (H*W, 3, D, D)

        #     save_image(full_plot, vis_dir / f'a_test_{i}.png', nrow=t)
