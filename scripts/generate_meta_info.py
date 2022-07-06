from os import path as osp
from PIL import Image
import os

from utils import scandir
import glob
import random

def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = 'datasets/DIV2k/DIV2k_sub'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DIV2k800sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


def generate_meta_info_VISTAWeChat():
    """Generate meta info for DIV2K dataset.
    """

    import random

    # gt_folder = '/home/luo/PycharmProjects/BasicSR-master/datasets/VISTAWeChat'
    gt_folder = '../datasets/VISTAWeChat/val/HQ_frames'
    meta_info_txt = '../data/meta_info/meta_info_VISTAWeChat_val.txt'
    # meta_val_txt = '../../data/meta_info/meta_info_VISTAWeChat_val.txt'
    # meta_test_txt = '../../data/meta_info/meta_info_VISTAWeChat_test.txt'

    img_list = sorted(os.listdir(gt_folder))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            full_path = os.path.join(gt_folder, img_path)
            if not os.path.isdir(full_path):
                continue
            lr_imgs = os.path.join(full_path, '*.png')
            imgs = glob.glob(lr_imgs)
            info = f'{img_path} {len(imgs)}'
            print(idx + 1, info)
            f.write(f'{info}\n')


    # random.shuffle(img_list)
    # print(img_list)
    # img_val_list = img_list[:4]
    # img_test_list = img_list[4:14]
    # print(len(img_val_list), len(img_test_list))

    # with open(meta_val_txt, 'w') as f:
    #     for idx, img_path in enumerate(img_val_list):
    #         full_path = os.path.join(gt_folder, img_path)
    #         if not os.path.isdir(full_path):
    #             continue
    #         lr_imgs = os.path.join(full_path, '*.png')
    #         imgs = glob.glob(lr_imgs)
    #         info = f'{img_path} {len(imgs)}'
    #         print(idx + 1, info)
    #         f.write(f'{info}\n')
    #
    # with open(meta_test_txt, 'w') as f:
    #     for idx, img_path in enumerate(img_test_list):
    #         full_path = os.path.join(gt_folder, img_path)
    #         if not os.path.isdir(full_path):
    #             continue
    #         lr_imgs = os.path.join(full_path, '*.png')
    #         imgs = glob.glob(lr_imgs)
    #         info = f'{img_path} {len(imgs)}'
    #         print(idx + 1, info)
    #         f.write(f'{info}\n')



def generate_meta_info_REDS():
    """Generate meta info for REDS dataset.
    """

    # gt_folder = '/home/luo/PycharmProjects/BasicSR-master/datasets/VISTAWeChat'
    gt_folder = '../../datasets/REDS/train_sharp'
    meta_info_txt = '../../data/meta_info/meta_info_REDS.txt'
    # meta_val_txt = '../../data/meta_info/meta_info_VISTAWeChat_val.txt'
    # meta_test_txt = '../../data/meta_info/meta_info_VISTAWeChat_test.txt'

    img_list = sorted(os.listdir(gt_folder))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            full_path = os.path.join(gt_folder, img_path)
            if not os.path.isdir(full_path):
                continue
            lr_imgs = os.path.join(full_path, '*.png')
            imgs = glob.glob(lr_imgs)
            info = f'{img_path} {len(imgs)}'
            print(idx + 1, info)
            f.write(f'{info}\n')




if __name__ == '__main__':
    # generate_meta_info_div2k()
    # generate_meta_info_REDS()
    generate_meta_info_VISTAWeChat()
