import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
from utils import FileClient, get_root_logger, imfrombytes, img2tensor
from utils.registry import DATASET_REGISTRY
from data.transforms import augment, paired_random_crop
import cv2


@DATASET_REGISTRY.register()
class UVSSMDataset(data.Dataset):
    """VISTA dataset for training.

     The keys are generated from a meta info txt file.
     basicsr/data/meta_info/meta_info_VISTAWeChat_GT.txt

     Each line contains:
     1. subfolder (clip) name; 2. frame number; seperated by a white space.
     Examples:
     000 100
     001 100
     ...

     Key examples: "000/00000000"
     GT (gt): Ground-Truth;
     LQ (lq): Low-Quality, e.g., "000/00000000.png".

     Args:
         opt (dict): Config for train dataset. It contains the following keys:
             dataroot_gt (str): Data root path for gt.
             dataroot_lq (str): Data root path for lq.
             meta_info_file (str): Path for meta information file.
             val_partition (list): Validation partition.
             io_backend (dict): IO backend type and other kwarg.

             num_frame (int): Window size for input frames.
             center_gt (bool) : Only read center frame if it is True, else read num_frame frames
             gt_size (int): Cropped patched size for gt patches.
             interval_list (list): Interval list for temporal augmentation.
             random_reverse (bool): Random reverse input frames.
             use_flip (bool): Use horizontal flips.
             use_rot (bool): Use rotation (use vertical flip and transposing h
                 and w for implementation).

             scale (bool): Scale, which will be added automatically.
     """

    # This datareader can access serveral frames or single frame per time,
    def __init__(self, opt):
        super(UVSSMDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        # self.mask_root = Path(opt['dataroot_mask']) if opt['dataroot_mask'] is not None else None

        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.center_frame_idx = opt['num_frame'] // 2
        self.center_gt = opt['center_gt']
        if self.center_gt:
            self.frame_interval = 100 // 1
        else:
            self.frame_interval = 100 // self.num_frame

        # Read all videos clips in meta_info_file e.g., 000, 025, 066, etc.
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, _ = line.split(' ')
                self.keys.append(folder)
                # self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation set and testing set
        val_partition = []
        if opt['meta_val_file'] is not None:
            with open(opt['meta_val_file'], 'r') as fin:
                for line in fin:
                    folder, _ = line.split(' ')
                    val_partition.append(folder)

        if opt['meta_test_file'] is not None:
            with open(opt['meta_test_file'], 'r') as fin:
                for line in fin:
                    folder, _ = line.split(' ')
                    val_partition.append(folder)

        self.keys = [v for v in self.keys if v not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':  # only lmdb need db_paths and client_keys
            self.is_lmdb = True
            if self.mask_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.mask_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'mask']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']


        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, item):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        item = item // self.frame_interval

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        clip_name = self.keys[item]

        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        keys = []

        start_frame_idx = random.randint(0, 99 - (self.num_frame - 1) * interval)
        end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

        # each clip has 100 frames starting from 0 to 99
        # while (start_frame_idx < 0) or (end_frame_idx > 99):
        #     start_frame_idx = random.randint(0, 99 - (self.num_frame - 1) * interval)
        #     end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frames and LQ frames
        img_gts = []
        img_lqs = []
        # img_masks = []
        for i, neighbor in enumerate(neighbor_list):
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                # if self.mask_root is not None:
                #     img_mask_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                # if self.mask_root is not None:
                #     img_mask_path = self.mask_root / clip_name / f'{neighbor:08d}.png'

            if self.center_gt:
                if i == self.center_frame_idx:
                    img_bytes = self.file_client.get(img_gt_path, 'gt')
                    img_gt = imfrombytes(img_bytes, float32=True)
                    img_gts.append(img_gt)
            else:
                img_bytes = self.file_client.get(img_gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                img_gts.append(img_gt)

            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # if self.mask_root is not None:
            #     img_bytes = self.file_client.get(img_mask_path, 'mask')
            #     img_mask = imfrombytes(img_bytes, flag='grayscale', float32=True)
            #     img_mask = img_mask[:, :, np.newaxis]
                # img_masks.append(img_mask)

            keys.append(f'{clip_name}/{neighbor:08d}')

        img_len = len(img_lqs)

        # if self.mask_root is not None:
        #     img_lqs.extend(img_masks)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)

        img_lqs = torch.stack(img_results[:img_len], dim=0)
        img_gts = torch.stack(img_results[img_len:], dim=0)


        if self.num_frame == 1:
            img_lqs = img_lqs[0]
            img_gts = img_gts[0]
            # if self.mask_root is not None:
            #     img_masks = img_masks[0]

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)  or (1, c, h ,w)
        # key: str

        return {'lq': img_lqs, 'gt': img_gts, 'key': keys}

    def __len__(self):
        return len(self.keys) * self.frame_interval




# if __name__ == '__main__':
#     opt = {'name': 'VISTAWeChat', 'type': 'VISTAWeChatTestDataset', 'dataroot_gt': '/home/luo/PycharmProjects/Mybasicsr/datasets/VISTAWeChat',
#     'dataroot_lq': '/home/luo/PycharmProjects/Mybasicsr/datasets/VISTAWeChat', 'meta_info_file': '/home/luo/PycharmProjects/Mybasicsr/data/meta_info/meta_info_VISTAWeChat_val.txt',
#     'num_frame': 100, 'io_backend': {'type': 'disk'}}
#     from torch.utils.data import dataloader
#     dataset = VISTAWeChatTestDataset(opt)
#     data = dataloader.DataLoader(dataset)
#     k = iter(data)
#     k.__next__()
