import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

class CellDataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None, de_train=False):

        self.mode = mode
        self.is_val = is_val
        self.de_train = de_train
        self.data_path = path
        self.label_path = path.replace("images", "labels")
        self.data_file = os.listdir(self.data_path)
        if mode == "test":
            self.data_file = [f"{idx}.pkl" for idx in range(len(self.data_file))]
        self.img_file = self._select_img(self.data_file)

        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).permute(2, 1, 0).float()
        
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.label_path, img_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).permute(2, 1, 0).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            #img = self.transforms(img)
            torch.manual_seed(seed)
            #gt = self.transforms(gt)

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            img_list.append(file)

        return img_list

    def __len__(self):
        if self.de_train == False:
            return len(self.img_file)
        else:
            return len(self.img_file) // 8
