import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import anndata as ad
from utils.dataset import get_loader
from dataclasses import dataclass
from utils.models.Regi import StateReg
import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from geomloss import SamplesLoss
from omegaconf import OmegaConf


#TODO:
# Train with the original metrics included in the website
# Scoring notebook check

class MyLoss():
    def __init__(self):
        # KL divergence loss
        #self.criterion = F.kl_div
        self.criterion = SamplesLoss(loss="sinkhorn", p=2, blur=0.05) # Wasserstein loss
    
    def __call__(self, outputs, targets):
        #log_probs = F.log_softmax(outputs, dim=-1)
        #targets_probs = F.softmax(targets, dim=-1)
        #self.criterion(log_probs, targets_probs, reduction='batchmean')
        return self.criterion(outputs, targets)
    
    def get_name(self):
        return "MSE"
    


@dataclass
class Config:
    log_dir = "logs"
    name = "RegiState"
    batch_size = 1000 # Genes processed at once
    version = 1
    epochs = 30
    lr = 1e-3

@dataclass
class ModelConfig:
    embed_dim = 128
    num_transformer_layers = 2
    num_mlp_layers = 2
    mlp_hidden_dims = [256, 128]

cfg = Config()
model_cfg = ModelConfig()

from lightning.pytorch.callbacks import ModelCheckpoint


def main():
    dataset,gene_dim,pert_dim,batch_dim = get_loader()
    trainDataset,valDataset = torch.utils.data.random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])
    #implement k-fold later
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valDataset, batch_size=cfg.batch_size, shuffle=False)    

    kwargs = {
        'gene_dim': gene_dim,
        'pert_dim': pert_dim,
        'batch_dim': batch_dim,
        'embed_dim': model_cfg.embed_dim,
        'num_transformer_layers': model_cfg.num_transformer_layers,
        'num_mlp_layers': model_cfg.num_mlp_layers,
        'mlp_hidden_dims': model_cfg.mlp_hidden_dims}
    
    criterion = MyLoss()
    model = StateReg(
        criterion=criterion,
        kwargs=kwargs)
    


    logger = TensorBoardLogger(cfg.log_dir, name=cfg.name)
    cfg.version = logger.version
    dirPath = f"{cfg.log_dir}/{cfg.name}/version_{cfg.version}"
    lossName = criterion.get_name()
    best_model = ModelCheckpoint(dirpath=f"{dirPath}/best/",filename="best-{epoch:02d}-{loss:.3f}",monitor=f"val/loss",mode="min",save_top_k=3,)
    periodic = ModelCheckpoint(dirpath=f"{dirPath}/periodic/",filename="epoch{epoch:02d}-step{step:02d}",every_n_train_steps=100,)
    ckpt_path_to_use = best_model.best_model_path if best_model.best_model_path else (periodic.best_model_path if periodic.best_model_path else None)
    cfg.loss = lossName
    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator='gpu', devices=1,callbacks=[best_model,periodic],logger=logger,default_root_dir=dirPath)
    trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path_to_use)
    with open(f"{dirPath}/config.yaml","w") as f:
        OmegaConf.save(config=OmegaConf.structured(cfg),f=f)

if __name__ == "__main__":
    main()