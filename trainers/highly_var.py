import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import pandas as pd
import anndata as ad
from dataclasses import dataclass
from utils.dataset_highvars import get_loader
from utils.models.Bence import StateBence
from utils.metrics import MyMetrics
import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from geomloss import SamplesLoss
from omegaconf import OmegaConf
from create_submissiondata import prep_submission

#TODO:
#Proxy train batch embedding

class MyLoss():
    def __init__(self,name:str = "Wasserstein"):
        # KL divergence loss
        #self.criterion = F.kl_div
        crits = {"Wasserstein":SamplesLoss(loss="sinkhorn", p=2, blur=0.05), "MSE": nn.MSELoss()}
        self.criterion = crits[name]
        self.name = name

    
    def __call__(self, outputs, targets):
        #log_probs = F.log_softmax(outputs, dim=-1)
        #targets_probs = F.softmax(targets, dim=-1)
        #self.criterion(log_probs, targets_probs, reduction='batchmean')
        return self.criterion(outputs, targets)
    
    def get_name(self):
        return self.name
    




@dataclass
class Config:
    log_dir = "logs"
    name = "Highly_Var"
    batch_size = 64 # Genes processed at once
    version = 1
    epochs = 30
    lr = 1e-3
    num_workers = 15
    num_samples = 700
    target_gene_dim = 128

@dataclass
class ModelConfig:
    embed_dim = 128
    num_heads = 4
    mlp_hidden_dims = [256, 128]

cfg = Config()
model_cfg = ModelConfig()

from lightning.pytorch.callbacks import ModelCheckpoint


def main():
    dataset,gene_dim,maskidx = get_loader(cfg.num_samples,cfg.target_gene_dim)
    splitratio = .95
    trainDataset,valDataset = torch.utils.data.random_split(dataset,[int(splitratio*len(dataset)),len(dataset)-int(splitratio*len(dataset))])
    #implement k-fold later
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(valDataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    kwargs = {
        'gene_dim': gene_dim,
        'pert_dim': cfg.target_gene_dim,
        'embed_dim': model_cfg.embed_dim,
        'num_heads': model_cfg.num_heads,
        'mlp_hidden_dims': model_cfg.mlp_hidden_dims,
        'sample_count' : cfg.num_samples}
    
    criterion = MyLoss(name = "MSE")
    MyMetrics_instance = MyMetrics()
    model = StateBence(criterion=criterion,metrics=MyMetrics_instance,kwargs=kwargs)
    
    logger = TensorBoardLogger(cfg.log_dir, name=cfg.name)
    cfg.version = logger.version
    dirPath = f"{cfg.log_dir}/{cfg.name}/version_{cfg.version}"
    lossName = criterion.get_name()
    cfg.loss = lossName

    best_model = ModelCheckpoint(dirpath=f"{dirPath}/best/",filename="best-{epoch:02d}-{loss:.3f}",monitor=f"val/loss",mode="min",save_top_k=3,)
    periodic = ModelCheckpoint(dirpath=f"{dirPath}/periodic/",filename="{epoch:02d}-{step:02d}",every_n_train_steps=100,)

    ckpt_path_to_use = best_model.best_model_path if best_model.best_model_path else (periodic.best_model_path if periodic.best_model_path else None)
    
    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator='gpu', devices=1,callbacks=[best_model,periodic],logger=logger,default_root_dir=dirPath)
    trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path_to_use)
    
    with open(f"{dirPath}/config.yaml","w") as f:
        OmegaConf.save(config=OmegaConf.structured(cfg),f=f)
    pth = dirPath + "/periodic/" + os.listdir(f"{dirPath}/periodic")[0]
    # Prep submission is a bit harder here
    #prep_submission(pth,dirPath+ "/")
    
if __name__ == "__main__":
    main()