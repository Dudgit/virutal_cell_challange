import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F
import scanpy as sc


class log1pact(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,X):
        return torch.log1p(X)

# Define helper classes based on the structure observed in state_model.py
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        #layers.append(nn.ReLU())
        #layers.append(log1pact())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class BatchEmbedding(nn.Module):
    """Builds expression-aware cell embeddings."""
    def __init__(self, gene_dim, embed_dim,sample_count = 700):
        super().__init__()
        ##Maybe Modify to embed batch values one by one
        self.backbone = nn.Sequential(
            nn.Linear(gene_dim,embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2,embed_dim))
        self.combiner = nn.Linear(sample_count,1)
    def forward(self,gene_expression):
        x = self.backbone(gene_expression)
        return self.combiner(x.permute(0,2,1)).squeeze()


class BenceTransitionModel(nn.Module):
    def __init__(self, gene_dim, pert_dim, embed_dim, num_heads, mlp_hidden_dims,sample_count = 700):
        super().__init__()
        self.state_embedding = BatchEmbedding(gene_dim, embed_dim,sample_count=sample_count)  # Creating Embedding for the whole batch, so we can use it as a batch-state vector
                                                                    # From BatchSize x NumSample x GeneDim -> BatchSize x Embed_dim
        self.pert_embed = nn.Linear(pert_dim, embed_dim)            # Create Perturbation embedding from BatchSize x GeneNames -> BatchSize x Embed_dim
        self.MultiHeadAttention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.decoder = MLP(embed_dim, mlp_hidden_dims, gene_dim)

    def forward(self, gene_expr, pert_info):
        batch_state = self.state_embedding(gene_expr)
        pert_embedding = self.pert_embed(pert_info)
        x = self.MultiHeadAttention(pert_embedding.unsqueeze(1), batch_state.unsqueeze(1), batch_state.unsqueeze(1)) # Transforming the batch_state by the perturbation embedding state
        return self.decoder(x[0].squeeze(1))

    
class StateBence(pl.LightningModule):
    def __init__(self,criterion, metrics, kwargs):
        super().__init__()
        self.model = BenceTransitionModel(**kwargs)
        self.criterion = criterion
        self.lossName = criterion.get_name()
        self.metrics = metrics
        self.save_hyperparameters()

    def forward(self, gene_expr, pert_info):
        return self.model(gene_expr, pert_info)

    def calculate_metrics(self, outputs, targets):
        return self.metrics.compute(outputs, targets)


    def shared_step(self, batch,mode):
        input_genes, gene_expr, pert_info = batch
        pred_gene_expr = self(input_genes, pert_info)
        loss = self.criterion(pred_gene_expr, gene_expr)
        self.log(f'{mode}/loss', loss)
        with torch.no_grad():
            metric_values = self.calculate_metrics(pred_gene_expr, gene_expr)
            for name, value in metric_values.items():
                self.log(f'{mode}/{name}_metric', value)
        return loss
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = self.shared_step(batch,mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = self.shared_step(batch,mode="val")
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

