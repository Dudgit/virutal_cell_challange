import torch
import torch.nn as nn
from geomloss import SamplesLoss
import torch.nn.functional as F


class MyMetrics():
    def __init__(self,metric_names=["MSE","KLDiv"]):
        self.names = metric_names
        self.ws = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    
    def compute(self, outputs, targets):
        # Compute some metrics here
        mse = F.mse_loss(outputs, targets).item()
        log_probs = F.log_softmax(outputs, dim=-1)
        targets_probs = F.softmax(targets, dim=-1)
        kl_div = F.kl_div(log_probs, targets_probs, reduction='batchmean').item()
        ws_loss = self.ws(outputs, targets).item()
        return {"MSE": mse, "KLDiv": kl_div,"Wasserstein": ws_loss}
    
    def get_name(self):
        return self.names
    


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