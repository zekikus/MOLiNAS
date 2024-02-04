import io
import os
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.save_best_model import BestModelCheckPoint
from utils.cell_multi_class_dataset import CellDataset


import warnings
warnings.filterwarnings("ignore")

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda')
        else:
            return super().find_class(module, name)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 480, 287, 55 - moode_v1
# 1682, 98, 78 - moode_v2
seed = 143
modelNo = 78
n_classes = 4
batch_size = 128
path = "results/moode_versions/moode_v2/nds_v2_42"
data_path = "Datasets/WRTC/images_train_patches_pickle"


seed_torch(seed)

checkpoint = BestModelCheckPoint(modelNo)
device = torch.device('cuda')

# Dataloaders
train_dataset = CellDataset(data_path, mode="training", split=0.9)
val_dataset = CellDataset(data_path, mode="training", split=0.9, is_val=True)

val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=False)
train_loader = DataLoader(train_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=True)

print(train_dataset.__len__())

log = ""

# Loss Function
loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 10, 10, 1]).to(device))

# Load Model
model = None
with open(f"{path}/model_{modelNo}.pkl", "rb") as f:
    model = GPU_Unpickler(f).load()

print("Model No:", model.solNo, "Seed:", seed)
summary(model, input_size=(1, 3, 128, 128))

model.reset()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 1e-3

for epoch in range(200): 
  train_loss = []
  train_iou = []
  train_f1 = []
  
  # Train Phase
  model.train()
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

    with torch.set_grad_enabled(True):
      output = model(inputs)
      error = loss(output, labels)
      train_loss.append(error.item())

      # Calculate score
      tp, fp, fn, tn = smp.metrics.get_stats(output.argmax(axis=1), labels.argmax(axis=1), mode='multiclass', num_classes=n_classes)
      iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
      f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
      
      train_iou.append(iou_score[1].item())
      train_f1.append(f1_score[1].item())
      optimizer.zero_grad()
      error.backward()
      optimizer.step()
      del output
      
      del error
    del inputs
    del labels

  torch.cuda.empty_cache()

  # Validation Phase
  val_loss = []
  val_iou = []
  val_f1 = []
  model.eval()
  for inputs, labels in val_loader:

    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
    output = model(inputs)
      
    error = loss(output, labels)
    train_loss.append(error.item())

    # Calculate score
    tp, fp, fn, tn = smp.metrics.get_stats(output.argmax(axis=1), labels.argmax(axis=1), mode='multiclass', num_classes=n_classes)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
    val_loss.append(error.item())

    val_iou.append(iou_score[1].item())
    val_f1.append(f1_score[1].item())

    del output
    del error
    del inputs
    del labels
    
  avg_tr_loss = sum(train_loss) / len(train_loss)
  avg_tr_score = sum(train_iou) / len(train_iou)
  avg_val_loss = sum(val_loss) / len(val_loss)
  avg_val_score = sum(val_iou) / len(val_iou)
  txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_iou_score: {avg_tr_score}, val_loss: {avg_val_loss}, val_iou: {avg_val_score}"
  log += txt
  print(txt)
  checkpoint.check(avg_val_score, model, seed)
  torch.cuda.empty_cache()

# Write Log
with open(f"log_{modelNo}_seed_{seed}.txt", "w") as f:
    f.write(log)
