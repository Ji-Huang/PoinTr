import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *
import torch
import torch_points_kernels as tpk


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class ShapeNet_Car_Seq(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            # if config.CARS:
            #     self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        # self.n_trajs = 15 if self.subset == 'train' else 1
        self.n_trajs = config.N_TRAJS
        self.file_list = self._get_file_list(self.subset, self.n_trajs)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 256  #2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',  #RandomSamplePoints
                'parameters': {
                    'n_points': 256  #2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_trajs):
        """Prepare file list for the dataset"""
        file_list = []

        # print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
        samples = self.dataset_categories[subset]
        # if subset == 'train':
        # for s in samples:
        #     partial_paths = []
        #     for i in range(n_trajs):
        #         partial_path = self.partial_points_path % (subset, s, f'{i:02}')
        #         partial_paths.append(partial_path)
        #
        #     file_list.append({
        #         'taxonomy_id':
        #         s,
        #         'partial_path':
        #         partial_paths
        #         ,
        #         'gt_path':
        #         self.complete_points_path % (subset, s),
        #     })
        # else:
        for s in samples:
            for i in range(n_trajs):
                partial_path = self.partial_points_path % (subset, s, f'{i:02}')

                file_list.append({
                    'taxonomy_id':
                    f'{s}_{i:02}',
                    'partial_path':
                    partial_path,
                    'gt_path':
                    self.complete_points_path % (subset, s),
                })

        print_log('Complete collecting files of the dataset. Total files in %s split: %d' % (subset, len(file_list)), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # rand_idx = random.randint(0, 40) if self.subset == 'train' else 0
        gt_path = sample['gt_path']
        gt = IO.get(gt_path).astype(np.float16)
        gt_data = {'gt': gt}
        if self.transforms is not None:
            gt_data = self.transforms(gt_data)['gt']

        # if self.subset == 'train':
        # partial_paths = sample['partial_path']
        # partials_data = []
        # for partial_path in partial_paths:
        #     partial_dir = os.path.dirname(partial_path)
        #     # List all files in the directory to count available partial files
        #     print(partial_dir)
        #     available_files = sorted([f for f in os.listdir(partial_dir) if f.endswith('.pcd')])
        #     rand_idx = random.randint(0, len(available_files) - 1)
    
        #     # print('loading', os.path.join(partial_path, f'{rand_idx:03}.pcd'))
        #     partial = IO.get(os.path.join(partial_path, f'{rand_idx:03}.pcd')).astype(np.float32)
        #     partial_data = {'partial': partial}
        #     if self.transforms is not None:
        #         partial_data = self.transforms(partial_data)
        #     partials_data.append(partial_data['partial'])
        if self.subset == 'train':
            partial_path = sample['partial_path']
            partials_data = []

            partial_dir = os.path.dirname(partial_path)
            # List all files in the directory to count available partial files
            available_files = sorted([f for f in os.listdir(partial_dir) if f.endswith('.pcd')])

            window_size = 9
            half_window = window_size // 2

            # # Iterate through each file
            # for i in range(len(available_files)):
            # rand_idx = random.randint(0, len(available_files))
            rand_idx = random.randint(half_window+1, len(available_files)-half_window-1)

            # Loop through the window size to gather partials from (i - half_window) to (i + half_window)
            for offset in range(-half_window, half_window + 1):
                idx = rand_idx + offset

                # Ensure the index is within valid bounds
                if 0 <= idx < len(available_files):
                    partial = IO.get(os.path.join(partial_dir, f'{idx:03}.pcd')).astype(np.float32)
                    partial_data = {'partial': partial}
                    if self.transforms is not None:
                        partial_data = self.transforms(partial_data)
                    # Append the concatenated result to the list
                    partials_data.append(partial_data['partial'])

        else:
            partial_path = sample['partial_path']
            partials_data = []

            partial_dir = os.path.dirname(partial_path)
            # List all files in the directory to count available partial files
            available_files = sorted([f for f in os.listdir(partial_dir) if f.endswith('.pcd')])

            for idx in range(len(available_files)):
                partial = IO.get(os.path.join(partial_dir, f'{idx:03}.pcd')).astype(np.float32)
                partial_data = {'partial': partial}
                if self.transforms is not None:
                    partial_data = self.transforms(partial_data)
                # Append the concatenated result to the list
                partials_data.append(partial_data['partial'])

            # window_size = 9
            # half_window = window_size // 2
            #
            # # # Iterate through each file
            # # for i in range(len(available_files)):
            # rand_idx = 6
            #
            # # Loop through the window size to gather partials from (i - half_window) to (i + half_window)
            # for offset in range(-half_window, half_window + 1):
            #     idx = rand_idx + offset
            #
            #     # Ensure the index is within valid bounds
            #     if 0 <= idx < len(available_files):
            #         partial = IO.get(os.path.join(partial_dir, f'{idx:03}.pcd')).astype(np.float32)
            #         partial_data = {'partial': partial}
            #         if self.transforms is not None:
            #             partial_data = self.transforms(partial_data)
            #         # Append the concatenated result to the list
            #         partials_data.append(partial_data['partial'])


        # window_size = random.choice([1, 3, 5, 7, 9, 11, 13])
        # for i in range(len(available_files)):
        #     partial = IO.get(os.path.join(partial_dir, f'{i:03}.pcd')).astype(np.float16)
        #     if len(aggregated_pcds) < window_size:
        #         aggregated_pcds.append(partial)
        #     else:
        #         aggregated_pcds.pop(0)
        #         aggregated_pcds.append(partial)

        #     combined_points = np.vstack(aggregated_pcds)
        #     partial_data = {'partial': combined_points}
        #     if self.transforms is not None:
        #         partial_data = self.transforms(partial_data)
        #     partials_data.append(partial_data['partial'])
        #     del combined_points
        #     torch.cuda.empty_cache()

        # else:
        # partial_dir = os.path.dirname(sample['partial_path'])
        # available_files = sorted([f for f in os.listdir(partial_dir) if f.endswith('.pcd')])
        # available_files = available_files[:50]
        # partials = []
        # window_size = random.choice([1, 3, 5, 7, 9, 11, 13])
        # aggregated_pcds = []
        # partials_data = []

        # for i in range(len(available_files)):
        #     partial = IO.get(os.path.join(partial_dir, f'{i:03}.pcd')).astype(np.float16)
        #     partials.append(partial)
        #     # Check if we can start aggregating with the sliding window
        #     if len(partials) >= window_size:
        #         # Get the windowed point clouds (last `window_size` point clouds)
        #         window_pcds = partials[-window_size:]
        #         combined_points = np.vstack(window_pcds)  # Shape: [window_size * N, 3]
        #         aggregated_pcds.append(combined_points)

        # for partial in aggregated_pcds:
        #     partial_data = {'partial': partial}
        #     if self.transforms is not None:
        #         partial_data = self.transforms(partial_data)
        #     partials_data.append(partial_data['partial'])

        # for i in range(len(available_files)):
        #     partial = IO.get(os.path.join(partial_dir, f'{i:03}.pcd')).astype(np.float16)
        #     if len(aggregated_pcds) < window_size:
        #         aggregated_pcds.append(partial)
        #     else:
        #         aggregated_pcds.pop(0)
        #         aggregated_pcds.append(partial)

        #     combined_points = np.vstack(aggregated_pcds)
        #     partial_data = {'partial': combined_points}
        #     if self.transforms is not None:
        #         partial_data = self.transforms(partial_data)
        #     partials_data.append(partial_data['partial'])
        #     del combined_points
        #     torch.cuda.empty_cache()

        data = {'partial': partials_data, 'gt': gt_data}
        # aggregated_pcds.clear()

        if data['gt'].shape[0] != self.npoints:
            print(f"Error at index {idx}: Expected {self.npoints}, but got {data['gt'].shape[0]}")
        assert data['gt'].shape[0] == self.npoints

        return sample['taxonomy_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)