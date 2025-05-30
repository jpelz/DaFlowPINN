import torch
from torch.utils.data import Dataset
import numpy as np
import glob

def min_max_normalize(tensor, return_denormalization=False):
    min_vals = tensor.amin(dim=(-3,-2,-1), keepdim=True)  # Min per channel
    max_vals = tensor.amax(dim=(-3,-2,-1), keepdim=True)  # Max per channel
    
    # Apply Min-Max normalization
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero

    if not return_denormalization:
        return normalized_tensor
    else:
        # Create a denormalization function
        def denormalize(tensor):
            return tensor * (max_vals - min_vals) + min_vals
        
        return normalized_tensor, denormalize

class PatchDataset_MultiFile(Dataset):
    def __init__(self, data_directory, device, batch_size, shuffle=True):
        """
        PatchDataset class for loading and managing low-resolution (LR) and high-resolution (HR) data.
        This class loads the full data of one file into RAM, but only the current batch is loaded onto the GPU or specified device during training.
        
        Attributes:
            data_directory (str): Directory containing the LR and HR image data files.
            device (torch.device): Device on which the data will be loaded (e.g., 'cpu' or 'cuda').
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
            lr_files (list): List of file paths for the LR image data.
            hr_files (list): List of file paths for the HR image data.
            lr_buffer (numpy.ndarray): Buffer containing the loaded LR image data.
            hr_buffer (numpy.ndarray): Buffer containing the loaded HR image data.
            file_idx_list (numpy.ndarray): Array of indices for the LR and HR files.
            patch_idx_list (numpy.ndarray): Array of indices for the patches within the LR and HR buffers.
            n_files (int): Number of LR and HR files.
            n_patches (int): Number of patches in the LR and HR buffers.
            batches_per_file (int): Number of batches per file based on the batch size.
        """

        self.data_directory = data_directory
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.lr_files = glob.glob(data_directory + "/*_LR.npy")
        self.hr_files = glob.glob(data_directory + "/*_HR.npy")

        self.lr_files.sort()
        self.hr_files.sort()

        self.lr_buffer = np.load(self.lr_files[0])
        self.hr_buffer = np.load(self.hr_files[0])

        self.file_idx_list = np.arange(len(self.lr_files))
        self.patch_idx_list = np.arange(self.lr_buffer.shape[0])

        self.n_files = len(self.lr_files)
        self.n_patches = self.lr_buffer.shape[0]
        self.batches_per_file = self.n_patches // self.batch_size

        if self.shuffle:
            np.random.shuffle(self.file_idx_list)
            np.random.shuffle(self.patch_idx_list)


    def __len__(self):
        return self.n_files * self.batches_per_file
    
    def __getitem__(self, idx):

        if idx == 0 and self.shuffle:
            np.random.shuffle(self.file_idx_list)

        if idx % self.batches_per_file == 0:
            file_idx = self.file_idx_list[idx // self.batches_per_file]
            self.lr_buffer = np.load(self.lr_files[file_idx])
            self.hr_buffer = np.load(self.hr_files[file_idx])

            if self.shuffle:
                np.random.shuffle(self.patch_idx_list)

        patch_idx0 = idx % self.batches_per_file * self.batch_size
        patch_idx1 = patch_idx0 + self.batch_size

        lr_patches = self.lr_buffer[self.patch_idx_list[patch_idx0:patch_idx1]]
        hr_patches = self.hr_buffer[self.patch_idx_list[patch_idx0:patch_idx1]]

        lr_patches = torch.from_numpy(lr_patches).float().to(self.device)
        hr_patches = torch.from_numpy(hr_patches).float().to(self.device)


        return min_max_normalize(lr_patches), min_max_normalize(hr_patches)
    

class PatchDataset_SingleFile(Dataset):
    def __init__(self, hr_file, lr_file, device):
        """
        """
        self.device=device
        
        self.lr = min_max_normalize(torch.from_numpy(np.load(lr_file)).float().to(self.device))
        self.hr = min_max_normalize(torch.from_numpy(np.load(hr_file)).float().to(self.device))


    def __len__(self):
        return self.lr.shape[0]
    
    def __getitem__(self, idx):

        lr_patches = self.lr[idx]
        hr_patches = self.hr[idx]

        return lr_patches, hr_patches
    

class PatchDataset_SingleFile2(Dataset):
    def __init__(self, hr_file, lr_file, device, batch_size, shuffle=True, condition_vector=None):
        """
        """
        self.device=device
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.lr = min_max_normalize(torch.from_numpy(np.load(lr_file)).float())
        self.hr = min_max_normalize(torch.from_numpy(np.load(hr_file)).float())

        self.condition_vector = condition_vector

        if self.condition_vector is not None:
            self.condition_vector = np.load(self.condition_vector)
            if self.condition_vector.shape[0] != self.lr.shape[0]:
                raise ValueError("Condition vector length must match the number of patches in the LR data.")
            self.condition_vector = torch.from_numpy(self.condition_vector).float()
            self.lr = self.lr + self.condition_vector[:, None, None, None, None]
        

        self.patch_idx_list = np.arange(self.lr.shape[0])
        self.n_patches = self.lr.shape[0]
        self.n_batches = self.n_patches // self.batch_size

        if self.shuffle:
            np.random.shuffle(self.patch_idx_list)


    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        if idx == 0 and self.shuffle:
            np.random.shuffle(self.patch_idx_list)

        patch_idx0 = idx * self.batch_size
        patch_idx1 = patch_idx0 + self.batch_size

        lr_patches = self.lr[self.patch_idx_list[patch_idx0:patch_idx1]].to(self.device)
        hr_patches = self.hr[self.patch_idx_list[patch_idx0:patch_idx1]].to(self.device)

        if self.condition_vector is not None:
            cond_vec = self.condition_vector[self.patch_idx_list[patch_idx0:patch_idx1]].to(self.device)

            return lr_patches, hr_patches, cond_vec
        else:
            return lr_patches, hr_patches