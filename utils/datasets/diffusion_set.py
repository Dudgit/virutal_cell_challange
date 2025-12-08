from torch.utils.data import Dataset
import numpy as np
import torch

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
    def __init__(self, adata,geneMapping,seqLength = 32):
        mask = (adata.obs.target_gene != "non-targeting")
        mask2 = (adata.obs.target_gene == "non-targeting")
        
        self.adata_X = adata[mask].X
        self.adata_obs = adata[mask].obs
        
        self.adata_batchX = adata[mask2].X
        self.adata_obs_batch = adata[mask2].obs
        self.seqLength = seqLength
        self.geneMapping = geneMapping

    def __len__(self):
        return self.adata_X.shape[0]

    def __getitem__(self, idx):
        # Get labels to convert to numbers
        currentBatch = self.adata_obs.iloc[idx].batch
        gene = self.adata_obs.iloc[idx].target_gene


        gene_expression = self.adata_X[idx].toarray().squeeze()
        gene_expressionseq = gene_expression.reshape(self.seqLength,-1)
        
        mask = self.adata_obs_batch["batch"] == currentBatch
        valid_indices = mask.to_numpy()
        random_index = np.random.choice(valid_indices)
        clean_expression = self.adata_batchX[random_index].toarray().squeeze()
        
        geneidx = self.geneMapping[gene]
        
        return gene_expressionseq, clean_expression.reshape(self.seqLength,-1), np.expand_dims(get_positional_encoding_vector(geneidx,self.seqLength),axis = -1)