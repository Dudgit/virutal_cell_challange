from torch.utils.data import Dataset
from scipy.sparse import issparse
import torch

import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc



def get_positional_encoding_vector(pos, d_model):
    """
    Calculates the positional encoding vector for a single position.
    
    Args:
        pos (int): The position index (e.g., 0, 1, 2...).
        d_model (int): The dimension of the embedding (e.g., 512).
                       Must be an even number for this implementation.

    Returns:
        np.ndarray: A 1D array of shape (d_model,)
    """
    
    # Ensure d_model is even for simplicity in pairing sin/cos
    if d_model % 2 != 0:
        raise ValueError("d_model must be an even number.")

    # 1. Create an array for all dimension indices: [0, 1, 2, ..., d_model-1]
    d_indices = np.arange(d_model)

    # 2. Calculate the 'i' term for the denominator: [0, 0, 1, 1, 2, 2, ...]
    # This is the 'i' from the formula
    i = d_indices // 2

    # 3. Calculate the denominator (the "timescale" term)
    # 10000^(2i / d_model)
    denominator = np.power(10000, (2 * i) / d_model)

    # 4. Calculate the angle for every dimension: pos / denominator
    angles = pos / denominator
    
    # 5. Create the final vector
    pe_vector = np.zeros(d_model)

    # 6. Apply sin to all even indices
    pe_vector[0::2] = np.sin(angles[0::2])

    # 7. Apply cos to all odd indices
    pe_vector[1::2] = np.cos(angles[1::2])

    return torch.tensor(pe_vector).float()


class GeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for loading gene expression, perturbation, and batch data.
    
    This class handles on-the-fly conversion of sparse data and one-hot encoding
    of categorical observation data.
    """
    def __init__(self, adata, target_gene_mapping, batch_mapping,batch_var_dict,sampleCount:int = 700,target_gene_dim:int = 128):
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
        mask = (adata.obs.target_gene != "non-targeting")
        mask2 = (adata.obs.target_gene == "non-targeting")
        
        self.adata_X = adata[mask].X
        self.adata_obs = adata[mask].obs
        self.batch_vars = batch_var_dict
        #mask2 = (adata.obs.target_gene == "non-targeting").to_numpy()
        self.adata_batchX = adata[mask2].X
        self.adata_obs_batch = adata[mask2].obs



        # Store the pre-computed mappings
        self.target_gene_mapping = target_gene_mapping
        self.batch_mapping = batch_mapping

        # Store dimensions for one-hot vectors
        self.num_unique_target_genes = len(target_gene_mapping)
        self.num_unique_batches = len(batch_mapping)
        self.sampleCount = sampleCount
        self.target_gene_dim = target_gene_dim

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

        # 3. Create One-Hot Tensors
        perturbation_tensor = get_positional_encoding_vector(self.target_gene_mapping[target_gene],self.target_gene_dim)

        batch = obs_row['batch']

        inputX = self.adata_batchX[self.adata_obs_batch['batch'].values == batch ].toarray()
        inputX = inputX[np.random.choice(inputX.shape[0],size = self.sampleCount,replace=False)]
        return inputX, gene_expression_tensor, perturbation_tensor


def load_data():
    dataRoot = "data/vcc_data"
    tr_adata_path = f"{dataRoot}/adata_Training.h5ad"
    adata = ad.read_h5ad(tr_adata_path)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    batch_var_dict = {}
    for batch in adata.obs['batch'].unique():
        batch_var_dict[batch] = adata[(adata.obs['batch']==batch) & (adata.obs.target_gene == "non-targeting")].X.toarray().std(axis=0)
    return adata,batch_var_dict

def get_mappings(adata):
    gene_dim = adata.shape[1] # Number of genes
    # Number of unique target genes for one-hot encoding
    # Number of unique batches for one-hot encoding
    batch_dim = adata.obs['batch'].nunique()
    all_batches = adata.obs['batch'].unique()
    gene_names = pd.read_csv("data/vcc_data/gene_names.csv", header = None).to_numpy().flatten()
    target_gene_mapping = {gene: i for i, gene in enumerate(["non-targeting",*gene_names])}
    batch_mapping = {batch: i for i, batch in enumerate(all_batches)}

    return gene_dim,batch_dim, target_gene_mapping, batch_mapping

def get_loader(num_samples,target_gene_dims = 128):
    adata,batch_var_dict = load_data()
    gene_dim,batch_dim,target_gene_mapping,batch_mapping = get_mappings(adata)
    dataset = GeneExpressionDataset(adata, target_gene_mapping, batch_mapping,batch_var_dict,sampleCount = num_samples,target_gene_dim=target_gene_dims)
    return dataset,gene_dim,batch_dim
