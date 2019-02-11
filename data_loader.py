import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def tiff_to_tensor(path):
    image = Image.open(path)
    return ToTensor()(image)

class BPDataSet(Dataset):
    def __init__(self, transform=None):
        self.root_dir = os.path.join(os.getcwd(), 'data', 'train')
        self.transform = transform
        self.samples = [get_file_name(path) for path in glob.glob(os.path.join(self.root_dir, '*[!mask].tif'))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        return tiff_to_tensor(sample_path + '.tif'), tiff_to_tensor(sample_path + '_mask.tif')
