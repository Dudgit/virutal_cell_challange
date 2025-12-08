
import torch
import lightning as pl
import torch.nn as nn

from lightning.pytorch.loggers import TensorBoardLogger
from utils.metrics import sampleMetric


criterion = nn.BCELoss()

class Basemodel():
    def __init__(self,*args,**kwargs):
        self.name = "Placeholder"
        self.backbone = 0 # loadBackbone
        self.extrablock = nn.Linear(*args)

    def __call__(self):
        return 0
    
    


class ModelTrainer(pl.LightningModule):
    def __init__(self ,*args, **kwargs):
        super(ModelTrainer, self).__init__()
        model = Basemodel(**kwargs)
        self.model = model
        self.save_hyperparameters()
        self.m1 = sampleMetric()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        X_input, y_target = batch
        preds = self.model(X_input)
        loss = criterion(preds, y_target)
        metricValue = self.m1(preds=preds,targets=y_target)
        self.log('train/loss', loss)
        self.log(f'train/{self.m1.name}',metricValue)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X_input, y_target = batch
        preds = self.model(X_input)
        loss = criterion(preds, y_target)
        metricValue = sampleMetric(preds=preds,targets=y_target)
        self.log('validation/loss', loss)
        self.log(f'validation/{self.m1.name}',metricValue)
        return loss
