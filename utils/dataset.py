#TODO: create dataset and data loaders
from torch.utils.data import DataLoader


class CellDataset:
    def __init__(self,paths,transforms = lambda x:x):
        self.paths = paths
        self.transform = transforms
    
    def len(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        path = self.paths[idx]
        data = 0 #TODO: Implement load methods
        label = 0
        return data,label

def getLoader(paths,conf):
    trainPaths,ValPaths = paths[:int(len(paths)*conf.ratio)], paths[int(len(paths)*conf.ratio):] # add other method if needed.
    trainDS = CellDataset(trainPaths)
    valDS = CellDataset(ValPaths)
    trainLoader = DataLoader(trainDS,batch_size = conf.batch_size,shuffle = True, num_workers= conf.num_workers, persistent_workers = conf.persistent_workers)
    valLoader = DataLoader(valDS,batch_size = conf.batch_size,shuffle = False, num_workers= conf.num_workers, persistent_workers = conf.persistent_workers)
    return trainLoader, valLoader