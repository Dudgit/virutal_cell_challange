from dataclasses import dataclass
import glob
from utils.dataset import getLoader
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as pl 
from omegaconf import OmegaConf
from utils.model import ModelTrainer


@dataclass
class ModelParams:
    epochs: int = 10
    devices: int = 1 # Number of gpus
    init_features: int = 32 # Number of initial parameters
    #param_list = [16,32,64,128]
    learning_rate: float = 1e-4 # Default ADAM probabbly will not change

@dataclass
class Config:
    #? Dataset related hyperparameters
    batch_size: int = 32 
    num_workers: int = 4 
    persistent_workers:bool = False
    ratio:float = 0.8

    #? Training loop related hyperparameters
    modelParams = ModelParams()
    
    #? Logging related hyperparameters
    train_data_path: str = 'data/competition_support_set'
    log_dir: str = 'logs/'
    name = "EDLCT" # Elte Deep learning competition team
    version = "v1"


def main():
    conf = Config()
    paths = glob.glob(f'{conf.train_data_path}/*')
    trainLoader, valLoader = getLoader(paths=paths,conf=conf)
    tblogger = TensorBoardLogger(save_dir=conf.log_dir,name = conf.name, version=conf.version)
    model = ModelTrainer(**conf.modelParams)
    
    trainer = pl.Trainer(max_epochs=conf.epochs, logger=tblogger,default_root_dir=tblogger.log_dir,devices=conf.devices, accelerator="auto")
    
    with open(f'{conf.log_dir}/{conf.trainType}/config_{conf.version}.yaml',"w+") as f:
        OmegaConf.save(conf,f)
    trainer.fit(model,train_dataloaders = trainLoader, val_dataloaders = valLoader)

if __name__ == "__main__":
    main()
