import os
import pyrootutils
import logging
import numpy as np
import hydra
from datetime import datetime
from omegaconf import DictConfig,OmegaConf

import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.data.dataloader

pyrootutils.setup_root(
    search_from = __file__,
    indicator = ["environment.yaml"],
    pythonpath= True
)

from src.data import my_data_module
from src.model import my_predictor_module

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None , config_path="../config/" , config_name="train.yaml")
def main(cfg):
    print("CONFIG")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    log.info("Here we go")
    task = cfg.task_name
    task_cfg = cfg.experiment.gfp
    log.info(f"The current task is {task}")
    data_module = my_data_module(task_cfg = task_cfg, **cfg.data)
    # data_module.setup()
    
    # print(type(data_module._dataset))   

    # test_loader = data_module.train_dataloader()

    # for batches, (feature, target) in enumerate(test_loader):
    #     print(f"batches {batches}")
    #     print(f"features {feature.shape}")
    #     print(f"target {target.shape}")
    #     break

    predictor_module = my_predictor_module(cfg.model)

    output_dir = datetime.now().strftime("%m_%d_%Y_%H_%M") 


    ckpt_dir = os.path.join(
        cfg.callbacks.dirpath,
        f"mutant_{task_cfg.min_mutant_dist}",
        f"percentile_{'_'.join([str(x) for x in task_cfg.filter_percentile])}",   
        task_cfg.smoothing_params,
        output_dir  
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    cfg.callbacks.dirpath = ckpt_dir
    cfg_save_path = os.path.join(ckpt_dir, "config.yaml")

    # os.path.join(cfg.gfp.task_dir, f"mutant{cfg.gfp.min_mutant_dist}_percentile_{self._task_cfg.filter_per[0]}_{self._task_cfg.filter_per[1]}")


    log.info(f"The config file will be saved to {cfg_save_path}")

    OmegaConf.save(cfg, cfg_save_path)

    callbacks_module = hydra.utils.instantiate(cfg.callbacks)

    trainer = Trainer(**cfg.trainer, callbacks = callbacks_module, devices=[torch.cuda.current_device()])

    log.info(f"Now start training!")

    trainer.fit(model = predictor_module, datamodule = data_module)



if __name__ == "__main__":
    main()