import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, data_path, transform=None,detection_algorithm=None):
        self.data_path = data_path
        self.transform = transform
        self.algo=detection_algorithm

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        # Load data
        sample=np.load(self.data_path[idx])
        data=sample['data']
        label=sample['mask']

        # Converting from numpy to pytorch
        data=torch.from_numpy(data).to(torch.float32)
        label=torch.from_numpy(label).to(torch.float32)


        # Matching dimmension for the transformations
        data=data.unsqueeze(0)
        label=label.unsqueeze(0)

        # Apply transformation to the data and mask
        data,label=self.transform(data,label)
        
        # Calculating DMD from the 
        data,_=self.algo.DMD_torch(data.squeeze(0))
        data=torch.cat((data.real,data.imag),dim=0)

        # Add channel dim and permute the input: (1, H, W, D)
        data=data.unsqueeze(0).permute(0,2,3,1)
        label=label.unsqueeze(-1)

        return data, label

    
    






