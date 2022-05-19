import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.debug = config.DEBUG
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()  #eg:'02828884-3d2ee152db78b312e5a8eba5f6050bab.npy'
        if self.debug:
            lines=lines[0:72]

        self.file_list = []
        for line in lines:
            line = line.strip()##eg:'02828884-3d2ee152db78b312e5a8eba5f6050bab.npy'
            taxonomy_id = line.split('-')[0]#eg:'02828884'
            model_id = line.split('-')[1].split('.')[0] #'3d2ee152db78b312e5a8eba5f6050bab.npy'
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')#这根本就没loaded，只是读取了样本标签和文件编号
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        n, c=data.shape
        assert c == 3

        if n!=self.npoints:
            choice = np.random.choice(n, self.npoints, replace=True)
            data = data[choice, :]

        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)