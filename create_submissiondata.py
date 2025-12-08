import torch
import numpy as np
import anndata as ad
from dataclasses import dataclass
from utils.datasetBence import get_loader, get_positional_encoding_vector
from utils.models.Bence import StateBence
from utils.metrics import MyMetrics
from geomloss import SamplesLoss
import pandas as pd
from tqdm import tqdm

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
        return "Wasserstein"
    

@dataclass
class Config:
    batch_size = 64 # Genes processed at once
    num_workers = 15
    num_samples = 100
    target_gene_dim = 128

@dataclass
class ModelConfig:
    embed_dim = 128*2
    num_heads = 8
    mlp_hidden_dims = [256, 128]


def prep_submission(cpkt_path,vnum):
    cfg = Config()
    model_cfg = ModelConfig()

    dataset,gene_dim,_ = get_loader(cfg.num_samples,cfg.target_gene_dim)
    testLoader =  torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)


    kwargs = {'gene_dim': gene_dim,'pert_dim': cfg.target_gene_dim,'embed_dim': model_cfg.embed_dim, 'num_heads': model_cfg.num_heads,'mlp_hidden_dims': model_cfg.mlp_hidden_dims,'sample_count' : cfg.num_samples}
    criterion = MyLoss()
    MyMetrics_instance = MyMetrics()
    model = StateBence(criterion=criterion,metrics=MyMetrics_instance,kwargs=kwargs)


    cpkt = torch.load(cpkt_path,weights_only=False)
    model.load_state_dict(cpkt['state_dict'])

    gene_names = pd.read_csv("data/vcc_data/gene_names.csv", header = None).to_numpy().flatten()
    pert_counts = pd.read_csv('data/vcc_data/pert_counts_Validation.csv')

    model.eval()
    matrix = []
    dl_iter = iter(testLoader)
    batch_info, _, _ = next(dl_iter)
    for target_gene in pert_counts['target_gene'].unique():
        print(f'Gene:{target_gene}')
        repeats = pert_counts.loc[pert_counts.target_gene  == target_gene,"n_cells"].item()
        curr_length = 0
        for it in tqdm(range(0,repeats,cfg.batch_size)):
            pert_info = get_positional_encoding_vector(dataset.target_gene_mapping[target_gene],cfg.target_gene_dim)
            pert = torch.stack([torch.tensor(pert_info,requires_grad=False)]*batch_info.shape[0])
            pred_levels = model(batch_info,pert)
            curr_length += cfg.batch_size
            if curr_length > repeats:
                remainings = cfg.batch_size - curr_length + repeats
                matrix.extend(pred_levels.detach().numpy()[:remainings])
            else:
                matrix.extend(pred_levels.detach().numpy())
    np.save(str(vnum) + 'submissionmx.npy',matrix)
    np.save(str(vnum) + 'submissionbatch.npy',batch_info.detach().numpy())

if __name__ == "__main__":
    
    trainingID = "PosPerturb"
    vnum = 17
    cp = "periodic"
    currentSave = "epoch=29-step=34300"
    cpkt_path = f'logs/{trainingID}/version_{vnum}/{cp}/{currentSave}.ckpt'
    prep_submission(cpkt_path, vnum="v4_")