import io
import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.save_best_model import BestModelCheckPoint
from utils.cell_multi_class_dataset import CellDataset

import warnings
warnings.filterwarnings("ignore")

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda:0')
        else:
            return super().find_class(module, name)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # 480, 287, 55 - v1
    # 78, 98, 1682 - v2
    modelNo = 78 
    n_classes = 4
    
    pickle_path = "nds_v2_42"
    path = "results/moode_versions/moode_v2"
    data_path = "Datasets/WRTC/images_test_patches_pickle"
    

    checkpoint = BestModelCheckPoint(modelNo)
    device = torch.device('cuda')

    # Dataloaders
    test_dataset = CellDataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False,  num_workers=0, pin_memory=True)

    print(test_dataset.__len__())

    result_df = pd.DataFrame(columns=["seed", "metric", "c1", "c2", "c3", "c4"])

    # 0: background
    # 1: platelets
    # 2: wbc
    # 3: rbc

    for seed in [10, 15, 143, 1234, 3074]:
        
        print("Seed:", seed)
        metrics = {"iou_micro": (smp.metrics.iou_score, []), 
            "iou_macro": (smp.metrics.iou_score, []), 
            "f1_micro": (smp.metrics.f1_score, []), 
            "f1_macro": (smp.metrics.f1_score, []), 
            "acc_micro": (smp.metrics.accuracy, []), 
            "precision_micro": (smp.metrics.precision, []),
            "precision_macro": (smp.metrics.precision, []),
            "recall_micro": (smp.metrics.recall, []),
            "recall_macro": (smp.metrics.recall, []),
            "accuracy_micro": (smp.metrics.accuracy, []),
            "accuracy_macro": (smp.metrics.accuracy, [])}

        # Load Model
        model = None
        with open(f"{path}/{pickle_path}/model_{modelNo}.pkl", "rb") as f:
            model = GPU_Unpickler(f).load()

        # Load pre-trained weights
        print("Model No:", model.solNo, "Seed:", seed)
        print("Load Model...")
        model.load_state_dict(torch.load(f"{path}/long_runs/model_{modelNo}_seed_{seed}.pt", map_location='cuda:0'))
        model.to(device)

        tp_values = torch.from_numpy(np.zeros((1,4)))
        fp_values = torch.from_numpy(np.zeros((1,4)))
        fn_values = torch.from_numpy(np.zeros((1,4)))
        tn_values = torch.from_numpy(np.zeros((1,4)))

        # Calculate metrics for each test image
        counter = 0
        model.eval()
        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
                output = model(inputs)

                tp, fp, fn, tn = smp.metrics.get_stats(output.argmax(axis=1), labels.argmax(axis=1), mode='multiclass', num_classes=n_classes)
                
                tp_values += tp
                fp_values += fp
                fn_values += fn
                tn_values += tn

                counter += 1

                
                # Image-wise metric calculation
                if counter % 336 == 0:
                    for key, value in metrics.items():
                        if 'micro' in key: continue

                        func, result_list = value
                        metric_score = func(tp_values, fp_values, fn_values, tn_values, reduction="macro")
                        result_list.append(metric_score[0].numpy())

                    tp_values = torch.from_numpy(np.zeros((1,4)))
                    fp_values = torch.from_numpy(np.zeros((1,4)))
                    fn_values = torch.from_numpy(np.zeros((1,4)))
                    tn_values = torch.from_numpy(np.zeros((1,4)))

                    counter = 0
                    #print("Reset")
        
        # Calculate average values for each metrics and classes
        for key, value in metrics.items():
            if 'micro' in key: continue
            func, result_list = value
            average_metrics = np.nanmean(result_list, axis=0)
            result_df.loc[len(result_df.index)]= ([f"seed:{seed}", key, average_metrics[0], average_metrics[1], average_metrics[2], average_metrics[3]])

        print()
    
    print()
    result_df.set_index(["seed", "metric"], inplace=True)
    result_df.to_excel(f"model_{modelNo}_results.xlsx")
