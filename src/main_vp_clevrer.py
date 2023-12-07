import hydra
from omegaconf import DictConfig

from trainer_clevrer import VPTrainer


@hydra.main(config_path="../config", config_name="trainer_clevrer")
def main(cfg: DictConfig):
    trainer = VPTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
