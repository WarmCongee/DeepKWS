from torch.utils.data import Dataset, DataLoader
import os
import torch
from set_creation import *

class KWSDataSet(Dataset):
    def __init__(self, dataset_path):
        self.dataset_list = []
        files = os.listdir(dataset_path)

        for index, file in enumerate(files):
            if index % 2000 == 1999:
                print(str(index+1) +" datas had load")
            fbank_tensor = torch.load(os.path.join(dataset_path, file))
            self.dataset_list.append(fbank_tensor)
 
    def __getitem__(self, index):
        return self.dataset_list[index]

    def __len__(self):
        return len(self.dataset_list)

