import torch
import torch.nn as nn

class Mymetric():
    def __init__(self):
        self.name = "metric"

    def __call__(preds,targets):
        return torch.mean(preds-targets)