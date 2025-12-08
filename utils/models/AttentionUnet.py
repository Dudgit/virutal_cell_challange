import torch
from diffusers import DDPMScheduler
import torch.nn as nn
import pytorch_lightning as pl

import math
def timestep_embedding(t, dim):
    """
    t: (batch,)
    dim: embedding dimension
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32) * -(math.log(10000) / half)
    ).to(t.device)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb 

class Uattention(nn.Module):
    def __init__(self,inDims,outDims,condDIm = 1):
        super().__init__()
        self.qlayer = nn.Linear(inDims,outDims)
        self.klayer = nn.Linear(condDIm,outDims)
        self.vlayer = nn.Linear(condDIm,outDims)
        self.mha = nn.MultiheadAttention(embed_dim = outDims,num_heads=8,batch_first=True) # Might be scaled
        self.norm = nn.LayerNorm(outDims)
        self.time_proj = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, inDims)
        )
    
    def forward(self,x,conditioning,timesteps):
        time_emb = timestep_embedding(timesteps,32)
        x = x + self.time_proj(time_emb).unsqueeze(1)
        q = self.qlayer(x)
        k = self.klayer(conditioning)
        v = self.vlayer(conditioning)
        attn,_ = self.mha(q,k,v)
        attn = self.norm(attn)
        return attn
    
class AttentionEncoder(nn.Module):
    def __init__(self,dims=[256,128,64]):
        super().__init__()
        self.down1 = Uattention(inDims = 565,outDims=dims[0])
        self.down2 = Uattention(inDims = dims[0],outDims=dims[1])
        self.down3 = Uattention(inDims = dims[1],outDims=dims[2])
    
    def forward(self,x, conditioning,timesteps):
        
        state1 = self.down1(x,conditioning,timesteps)
        state2 = self.down2(state1,conditioning,timesteps)
        state3 = self.down3(state2,conditioning,timesteps)
        return  state1,state2, state3
    
class AttentionDecoder(nn.Module):
    def __init__(self,dims=[64,128,256]):
        super().__init__()
        self.up3 = Uattention(inDims=dims[0]*2, outDims=dims[1])
        self.up2 = Uattention(inDims=dims[1]*2, outDims=dims[2])
        self.up1 = Uattention(inDims=dims[2]*2, outDims=512) # Final dimension matches input
        self.final_layer = nn.Linear(512,565)
        
    def forward(self, x, state1,stat2,state3,conditioning,timesteps):
        x = torch.concat([x,state3],dim=-1)
        x = self.up3(x,conditioning,timesteps)
        x = torch.concat([x,stat2],dim=-1)
        x = self.up2(x,conditioning,timesteps)
        x = torch.concat([x,state1],dim=-1)
        x = self.up1(x,conditioning,timesteps)
        x = self.final_layer(x)
        return x
    
class AttentionDiffusion(pl.LightningModule):
    def __init__(self,dims =[256,128,64],num_steps = 5000,criterion = nn.MSELoss(),lr=1e-3, metricdict = None):
        super().__init__()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_steps)
        self.encoder = AttentionEncoder(dims=dims)
        self.latentBlock = nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64))
        self.decoder = AttentionDecoder(dims=dims[::-1])
        self.criterion = criterion
        self.lr = lr
        self.metricdict = metricdict
        self.weight = criterion.weights if hasattr(criterion,'weights') else 1.0
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
    
    def shared_step(self,batch):
        
        target, inputdata,condition = batch
        device = inputdata.device
        noise = torch.randn(inputdata.shape,device=device)
        
        bs = inputdata.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), dtype=torch.int64,device=device)
        noisy_input = self.noise_scheduler.add_noise(inputdata, noise, timesteps)
        
        stat1,stat2,state3 = self.encoder(noisy_input,conditioning=condition,timesteps=timesteps)
        x = self.latentBlock(state3)
        genes = self.decoder(x,stat1,stat2,state3,conditioning=condition,timesteps=timesteps)
        return genes, target   
    
    def training_step(self, batch,batch_idx):
        genes, target = self.shared_step(batch)
        loss = self.criterion(genes, target)
        self.log("train/loss", loss,prog_bar=True)
        if self.metricdict is not None:
            for name,metric in self.metricdict.items():
                metric_val = metric(genes,target)
                self.log(f"train/{name}", metric_val,prog_bar=False)
            
        return loss
    
    def validation_step(self, batch,batch_idx):
        genes,target = self.shared_step(batch)
        loss = self.criterion(genes, target)
        self.log("val/loss", loss,prog_bar=True)
        if self.metricdict is not None:
            for name,metric in self.metricdict.items():
                metric_val = metric(genes,target)
                self.log(f"val/{name}", metric_val,prog_bar=False)
        return loss