import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config: DictConfig):

    print(OmegaConf.to_yaml(config))

    pl.seed_everything(config.seed)

    # Model
    model = instantiate(config.model)

    # Data
    datamodule = instantiate(config.datamodule)

    # Trainer
    trainer = pl.Trainer(**config.trainer)

    # Train
    trainer.fit(model, datamodule)

    # Test
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()