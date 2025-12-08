

from cell_load.data_modules import PerturbationDataModule
from cell_load.utils.modules import get_datamodule

from omegaconf import OmegaConf


from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
import torch

import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc





def prep_data_module(cfg,sentence_len):
    data_module: PerturbationDataModule = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )
    data_module.setup(stage="fit")
    return data_module

def get_module_config(cfg,data_module):
    data_config =   cfg["data"]["kwargs"]
    var_dims = data_module.get_var_dims()
    gene_dim = var_dims["gene_dim"]
    gene_dim = var_dims.get("gene_dim", 2000)

    module_config = {**cfg["model"]["kwargs"], **cfg["training"]}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]
    module_config["control_pert"] = data_config.get("control_pert", "non-targeting")
    module_config["batch_size"] = cfg["training"]["batch_size"]


    return dict(input_dim=var_dims["input_dim"],gene_dim=gene_dim,hvg_dim=var_dims["hvg_dim"],
                output_dim=var_dims["output_dim"],pert_dim=var_dims["pert_dim"],batch_dim=var_dims["batch_dim"],
                basal_mapping_strategy=data_config["basal_mapping_strategy"],
                 **module_config)



class GeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for loading gene expression, perturbation, and batch data.
    
    This class handles on-the-fly conversion of sparse data and one-hot encoding
    of categorical observation data.
    """
    def __init__(self, adata, target_gene_mapping, batch_mapping):
        """
        Args:
            adata (anndata.AnnData): An AnnData object containing the data. 
                                     Can be a slice (e.g., adata_train).
            target_gene_mapping (dict): A pre-computed dictionary mapping 
                                        target gene names to integer indices.
            batch_mapping (dict): A pre-computed dictionary mapping 
                                  batch names to integer indices.
        """
        # Store data sources
        self.adata_X = adata.X
        self.adata_obs = adata.obs

        # Store the pre-computed mappings
        self.target_gene_mapping = target_gene_mapping
        self.batch_mapping = batch_mapping

        # Store dimensions for one-hot vectors
        self.num_unique_target_genes = len(target_gene_mapping)
        self.num_unique_batches = len(batch_mapping)

    def __len__(self):
        """Returns the total number of cells in the dataset."""
        return self.adata_X.shape[0]

    def __getitem__(self, idx):
        """
        Fetches and processes a single cell's data at the given index.
        """
        
        # 1. Process Gene Expression Data
        gene_expression = self.adata_X[idx]
        
        # Handle sparsity: convert to dense numpy array if sparse
        if issparse(gene_expression):
            gene_expression_np = gene_expression.toarray().astype(np.float32)
        else:
            gene_expression_np = gene_expression.astype(np.float32)
            
        # Remove the extra dimension (e.g., (1, num_genes) -> (num_genes))
        gene_expression_np = gene_expression_np.squeeze()
        
        # Convert to PyTorch tensor
        gene_expression_tensor = torch.from_numpy(gene_expression_np)

        # 2. Process Perturbation and Batch Information
        # Get the observation data for the single cell
        obs_row = self.adata_obs.iloc[idx]
        target_gene = obs_row['target_gene']
        batch = obs_row['batch']

        # 3. Create One-Hot Tensors
        perturbation_tensor = torch.zeros(self.num_unique_target_genes, dtype=torch.float32)
        if target_gene in self.target_gene_mapping:
            pert_idx = self.target_gene_mapping[target_gene]
            perturbation_tensor[pert_idx] = 1.0

        batch_tensor = torch.zeros(self.num_unique_batches, dtype=torch.float32)
        if batch in self.batch_mapping:
            batch_idx = self.batch_mapping[batch]
            batch_tensor[batch_idx] = 1.0

        return gene_expression_tensor, perturbation_tensor, batch_tensor


def load_data():
    dataRoot = "data/vcc_data"
    tr_adata_path = f"{dataRoot}/adata_Training.h5ad"
    pertPath = f"{dataRoot}/pert_counts_Validation.csv"
    pcounts = pd.read_csv(pertPath)
    adata = ad.read_h5ad(tr_adata_path)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata

def get_mappings(adata):
    gene_dim = adata.shape[1] # Number of genes
# Number of unique target genes for one-hot encoding
    pert_dim = adata.obs['target_gene'].nunique()
    # Number of unique batches for one-hot encoding
    batch_dim = adata.obs['batch'].nunique()
    all_target_genes = adata.obs['target_gene'].unique()
    all_batches = adata.obs['batch'].unique()

    target_gene_mapping = {gene: i for i, gene in enumerate(all_target_genes)}
    batch_mapping = {batch: i for i, batch in enumerate(all_batches)}

    return gene_dim,pert_dim,batch_dim, target_gene_mapping, batch_mapping

def get_loader():
    adata = load_data()
    gene_dim,pert_dim,batch_dim,target_gene_mapping,batch_mapping = get_mappings(adata)
    dataset = GeneExpressionDataset(adata, target_gene_mapping, batch_mapping)
    return dataset,gene_dim,pert_dim,batch_dim
