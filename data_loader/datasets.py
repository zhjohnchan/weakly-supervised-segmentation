import os
import numpy as np
import scipy.io
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class KidneyDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.transform = transform
        if train:
            self.split_dir = os.path.join(root_dir, 'train')
        else:
            self.split_dir = os.path.join(root_dir, 'val')

        file_list = os.listdir(self.split_dir)
        file_list.sort(key=lambda x: int(x[-5:]))

        cts = []
        gts = []
        labels = []
        intensity_range = (-80, 200)

        for idx, file_path in enumerate(file_list):
            print('[{}/{}] Loading data.'.format(idx, len(file_list)))
            ct_org = sitk.ReadImage(os.path.join(self.split_dir, file_path, 'imaging.nii.gz'), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct_org).astype(np.int32)
            ct_array[ct_array > intensity_range[1]] = intensity_range[1]
            ct_array[ct_array < intensity_range[0]] = intensity_range[0]

            ct_tensor = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
            # slice = ct_tensor.size()[-1]
            ct_tensor = F.interpolate(ct_tensor, size=(256, 256, ct_tensor.size()[-1]), mode='trilinear')
            ct_array = ct_tensor.squeeze().squeeze().numpy().astype(np.int32).transpose((2, 0, 1))
            cts.append(ct_array)

            gt_org = sitk.ReadImage(os.path.join(self.split_dir, file_path, 'segmentation.nii.gz'), sitk.sitkInt8)
            gt_array = sitk.GetArrayFromImage(gt_org).astype(np.int8)
            gt_tensor = torch.FloatTensor(gt_array).unsqueeze(0).unsqueeze(0)
            gt_tensor = F.interpolate(gt_tensor, size=(256, 256, gt_tensor.size()[-1]), mode='trilinear')
            gt_array = gt_tensor.squeeze().squeeze().numpy().astype(np.int8).transpose((2, 0, 1))
            gts.append(gt_array)

            label = np.zeros((gt_array.shape[0], 2))  # slice数x2（有无kidney/tumor）
            for idx in range(gt_array.shape[0]):
                for cls in np.unique(gt_array[idx, :, :]):
                    if cls != 0:
                        label[idx, cls - 1] = 1  # idx for slice number, cls 0 for kidney, 1 for tumor.
            labels.append(label)

        self.cts = np.vstack(cts).astype('float32')
        self.cts = self.cts[:, :, :, np.newaxis].repeat(3, 3)  # (23634,550,550,3)
        self.cts = self.cts / self.cts.max()
        self.gts = np.vstack(gts)
        self.labels = np.vstack(labels).astype('float32')  # float32

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.transform(self.cts[idx]), self.gts[idx], torch.tensor(self.labels[idx]))
        return sample


class KidneyBCDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.transform = transform
        if train:
            self.split_dir = os.path.join(root_dir, 'train')
        else:
            self.split_dir = os.path.join(root_dir, 'val')

        file_list = os.listdir(self.split_dir)
        file_list.sort(key=lambda x: int(x[-5:]))
        # print(file_list)

        cts = []
        masks = []
        gts = []
        labels = []
        intensity_range = (-80, 200)
        for idx, file_path in enumerate(file_list):
            print('[{}/{}] Loading data.'.format(idx, len(file_list)))
            ct_org = sitk.ReadImage(os.path.join(self.split_dir, file_path, 'imaging.nii.gz'), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct_org).astype(np.int32)
            ct_array[ct_array > intensity_range[1]] = intensity_range[1]
            ct_array[ct_array < intensity_range[0]] = intensity_range[0]
            ct_tensor = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)

            ds_ct_tensor = F.interpolate(ct_tensor, size=(32, 32, ct_tensor.size()[-1]), mode='trilinear')
            ds_ct_array = ds_ct_tensor.squeeze().squeeze().numpy().astype(np.int32).transpose((2, 0, 1))
            ct_tensor = F.interpolate(ct_tensor, size=(256, 256, ct_tensor.size()[-1]), mode='trilinear')
            ct_array = ct_tensor.squeeze().squeeze().numpy().astype(np.int32).transpose((2, 0, 1))
            mask = ds_ct_array
            mask[mask > -75] = 0
            mask[mask <= -75] = 0.5
            cts.append(ct_array)
            masks.append(mask)

            gt_org = sitk.ReadImage(os.path.join(self.split_dir, file_path, 'segmentation.nii.gz'), sitk.sitkInt8)
            gt_array = sitk.GetArrayFromImage(gt_org).astype(np.int8)
            gt_tensor = torch.FloatTensor(gt_array).unsqueeze(0).unsqueeze(0)
            gt_tensor = F.interpolate(gt_tensor, size=(256, 256, gt_tensor.size()[-1]), mode='trilinear')
            gt_array = gt_tensor.squeeze().squeeze().numpy().astype(np.int8).transpose((2, 0, 1))
            gts.append(gt_array)

            label = np.zeros((gt_array.shape[0], 2))  # slice数x2（有无kidney/tumor）
            for idx in range(gt_array.shape[0]):
                for cls in np.unique(gt_array[idx, :, :]):
                    if cls != 0:
                        label[idx, cls - 1] = 1  # idx for slice number, cls 0 for kidney, 1 for tumor.
            labels.append(label)

        self.cts = np.vstack(cts).astype('float32')
        self.cts = self.cts[:, :, :, np.newaxis].repeat(3, 3)  # (23634,550,550,3)
        self.cts = self.cts / self.cts.max()
        self.masks = np.vstack(masks).astype('float32')
        self.gts = np.vstack(gts)
        self.labels = np.vstack(labels).astype('float32')  # float32

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (
        self.transform(self.cts[idx]), self.gts[idx], torch.tensor(self.masks[idx]), torch.tensor(self.labels[idx]))
        return sample

