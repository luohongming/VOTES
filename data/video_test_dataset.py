import glob
import os.path

import torch
from os import path as osp
from torch.utils import data as data

from data.data_util import generate_frame_indices, read_img_seq
from utils import get_root_logger, scandir
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial, VISTAWeChat.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.

    :returns (several lq frames or single lq frame) and (single gt center frames)
            if num_frame = 1, single lq frame is selected.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt   # collect dataset options.
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': []}   # lq_path: img_name, folder: video name, idx: index of img in folder
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt, self.imgs_name = {}, {}, {}

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial', 'vistawechat', 'vistasm', 'zte']:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))  # scan all images in subfolder_lq
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))  # scan all images in subfolder_gt

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                ####  the length of three following list should be the same
                self.data_info['lq_path'].extend(img_paths_lq)     # append all images in subfolder into a long list
                self.data_info['gt_path'].extend(img_paths_gt)     # append all images in subfolder into a long list
                self.data_info['folder'].extend([subfolder_name] * max_idx)  # append all subfolders into a long list

                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')

                # border_l = [0] * max_idx
                # for i in range(self.opt['num_frame'] // 2):
                #     border_l[i] = 1
                #     border_l[max_idx - i - 1] = 1
                # self.data_info['border'].extend(border_l)        # num_frame = 7   border_l = 1110000000...000000111

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                    self.imgs_name[subfolder_name] = img_paths_lq
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')


    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        # border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            # all frames have been read when calling __init__()
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))   # 0: select from rows
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt =read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            # 'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])



@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        Same as VideoTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
            imgs_name = self.imgs_name[folder]
        else:
            raise NotImplementedError('Without cache_data is not implemented.')

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'lq_path': imgs_name,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class VideoUVSSMTestDataset(data.Dataset):

    def __init__(self, opt):
        super(VideoUVSSMTestDataset, self).__init__()
        self.opt = opt   # collect dataset options.
        self.cache_data = opt['cache_data']

        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [],
                              'idx': []}  # lq_path: img_name, folder: video name, idx: index of img in folder
        # file client (io backend)
        self.center_gt = opt['center_gt']
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt, self.imgs_name = {}, {}, {}

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))


        if opt['name'].lower() in ['uvssmwechat', 'reds4', 'redsofficial', 'uvssmsm']:

            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(
                    list(scandir(subfolder_lq, full_path=True)))  # scan all images in subfolder_lq
                img_paths_gt = sorted(
                    list(scandir(subfolder_gt, full_path=True)))  # scan all images in subfolder_gt

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (
                    f'Different number of images in lq ({max_idx})'
                    f' and gt folders ({len(img_paths_gt)})')

                ####  the length of three following list should be the same
                self.data_info['lq_path'].extend(
                    img_paths_lq)  # append all images in subfolder into a long list
                self.data_info['gt_path'].extend(
                    img_paths_gt)  # append all images in subfolder into a long list
                self.data_info['folder'].extend(
                    [subfolder_name] * max_idx)  # append all subfolders into a long list

                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')

                # border_l = [0] * max_idx
                # for i in range(self.opt['num_frame'] // 2):
                #     border_l[i] = 1
                #     border_l[max_idx - i - 1] = 1
                # self.data_info['border'].extend(border_l)        # num_frame = 7   border_l = 1110000000...000000111

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                    self.imgs_name[subfolder_name] = img_paths_lq
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt

        else:
            raise ValueError(f'Non-supported video test dataset: {opt["name"]}')

        self.num_frame = self.opt['num_frame']
        self.img_len = len(self.data_info['gt_path']) // self.opt['num_frame']
        # print(self.img_len)
        # self.total_img = self.opt['num_frame'] * self.img_len


    def __getitem__(self, index):
        if not self.center_gt:
            index = index * self.num_frame + (self.num_frame // 2)

        folder = self.data_info['folder'][index]

        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        # border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.num_frame, padding=self.opt['padding'])

        lq_path_name = [osp.basename(self.imgs_lq[folder][i]) for i in select_idx]

        if self.cache_data:
            # all frames have been read when calling __init__()
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))   # 0: select from rows
            if self.mask_root is not None:
                imgs_mask = self.imgs_mask[folder].index_select(0, torch.LongTensor(select_idx))   # 0: select from rows
            if self.center_gt:
                img_gt = self.imgs_gt[folder][idx]
            else:
                img_gt = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            img_paths_gt = [self.imgs_gt[folder][i] for i in select_idx]

            imgs_lq = read_img_seq(img_paths_lq)
            if self.center_gt:
                img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            else:
                img_gt = read_img_seq(img_paths_gt)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (1, c, h, w) or (t, c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'lq_path': lq_path,
            'lq_path_name': lq_path_name,
            'select_idx': select_idx,
        }


    def __len__(self):
        if self.center_gt:
            return len(self.data_info['gt_path'])
        else:
            return self.img_len