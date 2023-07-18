import hydra
from omegaconf import DictConfig

from trainer import Trainer
# from trainer_vp import VPTrainer as Trainer


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run_vis(try_slots=False)


if __name__ == "__main__":
    main()
