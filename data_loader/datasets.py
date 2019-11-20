import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class AbdomenDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.transform = transform
        if train:
            self.split_dir = os.path.join(root_dir, 'train')
        else:
            self.split_dir = os.path.join(root_dir, 'val')

        ct_list = os.listdir(os.path.join(self.split_dir, 'CT'))
        gt_list = os.listdir(os.path.join(self.split_dir, 'GT'))
        ct_list.sort(key=lambda x: int(x[3:7]))
        gt_list.sort(key=lambda x: int(x[5:9]))

        cts = []
        gts = []
        labels = []
        for ct_name, gt_name in zip(ct_list, gt_list):
            ct_org = sitk.ReadImage(os.path.join(self.split_dir, 'CT', ct_name), sitk.sitkInt16)

            ct_array = sitk.GetArrayFromImage(ct_org)  #(96, 256, 256)
            cts.append(ct_array)
            gt_org = sitk.ReadImage(os.path.join(self.split_dir, 'GT', gt_name), sitk.sitkInt8)
            gt_array = sitk.GetArrayFromImage(gt_org)  # (96, 256, 256)
            gts.append(gt_array)
            label = np.zeros((gt_array.shape[0], 13))
            for idx in range(gt_array.shape[0]):
                for cls in np.unique(gt_array[idx]):
                    if cls != 0:
                        label[idx, cls-1] = 1
            labels.append(label)
        self.cts = np.vstack(cts).astype('float32')
        self.cts = self.cts / self.cts.max()
        self.gts = np.vstack(gts)
        self.labels = np.vstack(labels).astype('float32')

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.transform(self.cts[idx]), self.gts[idx], torch.tensor(self.labels[idx]))
        return sample

class KidneyDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.transform = transform
        if train:
            self.split_dir = os.path.join(root_dir, 'train')
        else:
            self.split_dir = os.path.join(root_dir, 'val')

        ct_list = os.listdir(os.path.join(self.split_dir, 'CT'))
        gt_list = os.listdir(os.path.join(self.split_dir, 'GT'))
        ct_list.sort(key=lambda x: int(x[3:7]))
        gt_list.sort(key=lambda x: int(x[5:9]))

        cts = []
        gts = []
        labels = []
        for ct_name, gt_name in zip(ct_list, gt_list):
            ct_org = sitk.ReadImage(os.path.join(self.split_dir, 'CT', ct_name), sitk.sitkInt16)

            ct_array = sitk.GetArrayFromImage(ct_org)  #(96, 256, 256)
            cts.append(ct_array)
            gt_org = sitk.ReadImage(os.path.join(self.split_dir, 'GT', gt_name), sitk.sitkInt8)
            gt_array = sitk.GetArrayFromImage(gt_org)  # (96, 256, 256)
            gts.append(gt_array)
            label = np.zeros((gt_array.shape[0], 2))
            for idx in range(gt_array.shape[0]):
                for cls in np.unique(gt_array[idx]):
                    if cls != 0:
                        label[idx, cls-1] = 1
            labels.append(label)
        self.cts = np.vstack(cts).astype('float32')
        self.cts = self.cts / self.cts.max()
        self.gts = np.vstack(gts)
        self.labels = np.vstack(labels).astype('float32')

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.transform(self.cts[idx]), self.gts[idx], torch.tensor(self.labels[idx]))
        return sample