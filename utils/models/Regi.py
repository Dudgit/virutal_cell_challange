import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F




# Define helper classes based on the structure observed in state_model.py
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class StateEmbeddingModel(nn.Module):
    """Builds expression-aware cell embeddings."""
    def __init__(self, gene_dim, pert_dim, batch_dim, embed_dim):
        super().__init__()
        self.gene_fc = nn.Linear(gene_dim, embed_dim)
        self.pert_fc = nn.Linear(pert_dim, embed_dim)
        self.batch_fc = nn.Linear(batch_dim, embed_dim)
        # Assuming a simple additive or concatenative approach for now
        self.combiner = nn.Linear(embed_dim * 3, embed_dim) # Example combiner

    def forward(self, gene_expr, pert_info, batch_info):
        gene_embed = self.gene_fc(gene_expr)
        pert_embed = self.pert_fc(pert_info)
        batch_embed = self.batch_fc(batch_info)

        # Simple concatenation and linear combination
        combined_embed = torch.cat([gene_embed, pert_embed, batch_embed], dim=-1)
        cell_embed = self.combiner(combined_embed) # Example: project back to embed_dim

        # In a real scenario, this might involve more complex interactions
        # potentially including CLS tokens or other mechanisms as hinted in the original code
        # For this subtask, a basic combination is sufficient to demonstrate integration.

        return cell_embed # Returning just the main cell embedding for simplicity


class StateTransitionModel(nn.Module):
    """Predicts how perturbations will change gene expressions and overall cell states."""
    def __init__(self, gene_dim, pert_dim, batch_dim, embed_dim, num_transformer_layers, num_mlp_layers, mlp_hidden_dims):
        super().__init__()
        self.state_embedding = StateEmbeddingModel(gene_dim, pert_dim, batch_dim, embed_dim)

        # Simple Transformer Encoder setup
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Reconstruction head (MLP to predict gene expression changes)
        self.reconstruction_head = MLP(embed_dim, mlp_hidden_dims, gene_dim)

    def forward(self, gene_expr, pert_info, batch_info):
        # Get initial cell embeddings
        cell_embed = self.state_embedding(gene_expr, pert_info, batch_info)

        # Pass embeddings through transformer
        # Transformer expects (sequence_length, batch_size, features)
        # Our current tensor is (batch_size, features) - treat each cell as a sequence item of length 1
        # Or, if we were processing sequences of cells, we'd need to reshape accordingly.
        # For now, let's assume transformer processes independent cell embeddings in parallel.
        # A more complex model might use attention across cells within a batch or sequence.
        # Let's treat each cell as a single item in a sequence for the transformer.
        cell_embed = cell_embed.unsqueeze(0) # Add sequence length dimension (1) -> (1, batch_size, embed_dim)


        transformer_output = self.transformer_encoder(cell_embed)

        # Remove the sequence length dimension
        transformer_output = transformer_output.squeeze(0) # -> (batch_size, embed_dim)

        # Predict the output gene expression (or change in gene expression)
        predicted_gene_expr = self.reconstruction_head(transformer_output)

        return predicted_gene_expr
    
class StateReg(pl.LightningModule):
    def __init__(self,criterion, metrics, kwargs):
        super().__init__()
        self.model = StateTransitionModel(**kwargs)
        self.criterion = criterion
        self.lossName = criterion.get_name()
        self.metrics = metrics
        self.save_hyperparameters()

    def forward(self, gene_expr, pert_info, batch_info):
        return self.model(gene_expr, pert_info, batch_info)

    def calculate_metrics(self, outputs, targets):
        return self.metrics.compute(outputs, targets)


    def shared_step(self, batch,mode):
        gene_expr, pert_info, batch_info = batch
        pred_gene_expr = self(gene_expr, pert_info, batch_info)
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

