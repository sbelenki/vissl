import torch
from torch.utils.data import Dataset
from vissl.config import AttrDict
from fvcore.common.file_io import PathManager
import h5py
import os
import numpy as np
from PIL import Image

class FastMRIDataSet(Dataset):
    """
    Adapter for loading MRI images, in k-space, from h5 files in the fastMRI data set
    https://fastmri.org/dataset/
    
    Args:
        cfg (AttrDict): configuration defined by user
        path (string): path to the dataset
        split (string): specify split for the dataset (either "train" or "val").
    """
    def __init__(self, cfg: AttrDict, path: str, split: str, dataset_name="fastmri_dataset", data_source="fastmri"):
        super(FastMRIDataSet, self).__init__()
        
        assert PathManager.isdir(path), f"Directory {path} does not exist"
        self.dataset_name = "singlecoil"
        self.data_source = "fastmri"
        self.path = path
        
        data = cfg.get("DATA", AttrDict({}))
        self.key = data.get("KEY", "reconstruction_esc")
        self.index = data.get("INDEX", 12)
        self.split = split.lower()
        self.dataset = self._load_data()
    
    def _load_data(self):
        splittype = "train" if self.split == "train" else "val"
        directory = os.path.join(self.path, f"{self.dataset_name}_{splittype}")
        files = [os.path.join(directory,x) for x in os.listdir(directory) if x.endswith(".h5")]
        return files
    
    def num_samples(self):
        """
        Size of the dataset
        """
        return len(self.dataset)

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx:int):
        """
        Simply return the mean dummy image of the specified size and mark
        it as a success.
        """       
        file_name = self.dataset[idx]
        #print(file_name)
        hf = h5py.File(file_name, "r")
        #print('Keys:', list(hf.keys()))
        #print('Attrs:', dict(hf.attrs))
        data  = hf[self.key][()]
        if np.iscomplexobj(data):
            data = np.stack((data.real, data.imag), axis=-1)
        
        tmp = data[self.index]
        
        #tmp = np.stack( (tmp,)*3, axis=-1)
        #print(tmp.shape)
        
        
        
        
        tmp = ( tmp * 255 / np.max(tmp)).astype('uint8')
        
        onedimImage = Image.fromarray(tmp)
        
        # need to make sure the image has 3 channels as that is what unet and other models expect
        img = Image.new(mode="L", size=onedimImage.size)
        img.paste(onedimImage,box=[0,0])
        
        #img = torch.from_numpy(tmp).unsqueeze(0)
        
        return img, True