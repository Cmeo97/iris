import hydra
from omegaconf import DictConfig

from trainer import Trainer
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
