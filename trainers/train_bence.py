import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from dataclasses import dataclass
from utils.Bence.dataset import get_loader
from utils.Bence.model import StateBence
from utils.metrics import MyMetrics, MyLoss

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

@dataclass
class Config:
    log_dir = "logs"
    name = "PosPerturb"
    batch_size = 128 # Genes processed at once
    version = 1
    epochs = 30
    lr = 1e-3
    num_workers = 15
    num_samples = 100
    target_gene_dim = 128

@dataclass
class ModelConfig:
    embed_dim = 128*2
    num_heads = 8
    mlp_hidden_dims = [256, 128]

cfg = Config()
model_cfg = ModelConfig()

from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    dataset,gene_dim,_ = get_loader(cfg.num_samples,cfg.target_gene_dim)
    trainDataset,valDataset = torch.utils.data.random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])
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
    
    criterion = MyLoss()
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


if __name__ == "__main__":
    main()