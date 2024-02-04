import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class F1Score(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes = 4):
        super(F1Score, self).__init__()
        self.n_classes = n_classes

    def forward(self, output, labels):
        tp, fp, fn, tn = smp.metrics.get_stats(output.argmax(axis=1), labels.argmax(axis=1), mode='multiclass', num_classes=self.n_classes)
        f1_score, f1_score_mean = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")

        return f1_score_mean
