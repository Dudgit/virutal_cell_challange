import os
from pathlib import Path
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
print("Current Working Directory ", os.getcwd())


import torch
from dataclasses import dataclass
import anndata as ad
import scanpy as sc
import pytorch_lightning as pl

from utils.datasets.diffusion_set import GeneExpressionDataset
from utils.models.AttentionUnet import AttentionDiffusion

@dataclass
class Config:
    log_dir = "logs"
    name = "AttentionDiffuse"
    dims = [256,128,64]
    batch_size = 64 # Genes processed at once
    epochs = 50
    lr = 1e-3
    num_workers = 15
    diffusion_steps = 10_000
    weights = 10
cfg = Config()

class WeighedMSELoss(torch.nn.Module):
    def __init__(self,weights:float = 10.0):
        super().__init__()
        self.weights = weights
        
    def forward(self,input,target):
        loss = (input - target)**2
        mask = target > 0
        loss = loss * mask
        loss = loss * self.weights
        loss = loss + (1 - mask*1) * (input - target)**2
        return loss.mean()


def get_loaders():
    dataRoot = "data/vcc_data"
    tr_adata_path = f"{dataRoot}/adata_Training.h5ad"
    adata = ad.read_h5ad(tr_adata_path)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    genemap = {b:i for i,b in enumerate(adata.obs.target_gene.unique())}
    dataset = GeneExpressionDataset(adata,genemap)
    trainDataset,valDataset = torch.utils.data.random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])
    #implement k-fold later
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(valDataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader,val_loader

def main():
    train_loader,val_loader = get_loaders()
    MetricDict = {"mse":torch.nn.MSELoss(), "mae":torch.nn.L1Loss(), "weighed_mse":WeighedMSELoss(),"KLDiv":torch.nn.KLDivLoss()}
    model = AttentionDiffusion(dims=cfg.dims,criterion=WeighedMSELoss(weights=cfg.weights),lr=cfg.lr,metricdict=MetricDict,num_steps=cfg.diffusion_steps)
    dirPath = f"{cfg.log_dir}/{cfg.name}"
    trainer = pl.Trainer(max_epochs=cfg.epochs,default_root_dir=dirPath,accelerator="auto",devices="auto")
    trainer.fit(model,train_loader,val_loader)

if __name__ == "__main__":
    main()