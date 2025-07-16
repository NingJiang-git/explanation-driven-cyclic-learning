import nibabel as nib
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import cv2


class PairedSmriDataset(Dataset):
    """ sliced sMRI datasets."""    
        
    def __init__(self, opt):
        super(PairedSmriDataset, self).__init__()
        """
        Args:
            clean_dir (string): Directory of the clean sMRIs.
            corrupted_dir (string): Directory of the corrupted sMRIs.
            data_file (string): File name of the train/val/test split file.
        """
        self.opt = opt
        self.clean_dir, self.corrupted_dir = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_file = opt['data_file']

    def __len__(self):
        return sum(1 for line in open(self.data_file)) - 2
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        df.close()
        
        lst_0 = lines[idx].split()
        clean_img_name_0 = lst_0[0]
        corrupted_img_name_0 = lst_0[1]
        sub_name_0 = clean_img_name_0[0:9]
        corrupted_image_path_0 = self.corrupted_dir + '/o' + sub_name_0 + '-T1_corrupted.nii' + '/axial/' + corrupted_img_name_0

        lst_1 = lines[idx+1].split()
        clean_img_name_1 = lst_1[0]
        corrupted_img_name_1 = lst_1[1]
        sub_name_1 = clean_img_name_1[0:9]
        clean_image_path_1 = self.clean_dir + '/' + sub_name_1 + '_T1w.nii' + '/axial/' + clean_img_name_1
        corrupted_image_path_1 = self.corrupted_dir + '/o' + sub_name_1 + '-T1_corrupted.nii' + '/axial/' + corrupted_img_name_1

        lst_2 = lines[idx+2].split()
        clean_img_name_2 = lst_2[0]
        corrupted_img_name_2 = lst_2[1]
        sub_name_2 = clean_img_name_2[0:9]
        corrupted_image_path_2 = self.corrupted_dir + '/o' + sub_name_2 + '-T1_corrupted.nii' + '/axial/' + corrupted_img_name_2

        if sub_name_0 != sub_name_1 or sub_name_1 != sub_name_2:
            return self.__getitem__(idx+1)
        else:
            corrupted_slice_0 = stand_for_brain(np.load(corrupted_image_path_0)[np.newaxis,:])
            corrupted_slice_1 = stand_for_brain(np.load(corrupted_image_path_1)[np.newaxis,:])
            corrupted_slice_2 = stand_for_brain(np.load(corrupted_image_path_2)[np.newaxis,:])
            clean_slice_1 = stand_for_brain(np.load(clean_image_path_1)[np.newaxis,:])
            corrupted_input = np.concatenate((corrupted_slice_0,corrupted_slice_1,corrupted_slice_2),axis=0)
            clean_slice_1 = np.concatenate((clean_slice_1,clean_slice_1,clean_slice_1),axis=0)
            clean_slice_1 = torch.from_numpy(clean_slice_1).float()
            corrupted_input = torch.from_numpy(corrupted_input).float()

            if lst_1[2] == 0:
                label = np.array([1,0,0,0])   
            elif lst_1[2] == 1:
                label = np.array([0,1,0,0]) 
            elif lst_1[2] == 2:
                label = np.array([0,0,1,0]) 
            elif lst_1[2] == 3:
                label = np.array([0,0,1,1]) 
            elif lst_1[2] == 4:
                label = np.array([0,1,0,1]) 
            else:
                label = np.array([0,1,1,0]) 
            label = torch.from_numpy(label).long()

            sample = {'lq': corrupted_input, 'gt': clean_slice_1, 'gt_name': corrupted_img_name_1, 'label': label}  
            return sample




class PairedSmriDataset_class(Dataset):
    """ sliced sMRI datasets."""    
        
    def __init__(self, opt):
        super(PairedSmriDataset_class, self).__init__()
        """
        Args:
            corrupted_dir (string): Directory of the corrupted sMRIs.
            data_file (string): File name of the train/val/test split file.
        """
        self.opt = opt
        self.clean_dir = opt['dataroot_gt']
        self.corrupted_dir = opt['dataroot_lq']
        self.data_file = opt['data_file']

    def __len__(self):
        return sum(1 for line in open(self.data_file)) - 2
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        df.close()
        # print(lines)
        
        lst_0 = lines[idx].split()
        clean_img_name_0 = lst_0[0]
        corrupted_img_name_0 = lst_0[1]
        sub_name_0 = clean_img_name_0[0:9]

        if int(lst_0[2]) != 0:
            corrupted_image_path_0 = self.corrupted_dir + '/o' + sub_name_0 + '-T1_corrupted.nii' + '/axial/' + corrupted_img_name_0

        else:
            corrupted_image_path_0 = self.clean_dir + '/' + sub_name_0 + '_T1w.nii' + '/axial/' + clean_img_name_0

        if lst_0[2] == 0:
            label = np.array([1,0,0,0])   
        elif lst_0[2] == 1:
            label = np.array([0,1,0,0]) 
        elif lst_0[2] == 2:
            label = np.array([0,0,1,0]) 
        elif lst_0[2] == 3:
            label = np.array([0,0,1,1]) 
        elif lst_0[2] == 4:
            label = np.array([0,1,0,1]) 
        else:
            label = np.array([0,1,1,0]) 
        label = torch.from_numpy(label).long()

        if True :
            corrupted_slice_0 = stand_for_brain(np.load(corrupted_image_path_0)[np.newaxis,:])
            corrupted_input = np.concatenate((corrupted_slice_0,corrupted_slice_0,corrupted_slice_0),axis=0)
            sample = {'lq': corrupted_input, 'label': label}  
            return sample


def stand(x, mean, std):
    x = (x - mean) / std
    return x


def stand_for_brain(x):
    y = np.nanmin(x)
    z = np.nanmax(x)
    x = (x - y) / (z - y)
    x_stand = stand(x, 0.5, 0.5)
    return x_stand
