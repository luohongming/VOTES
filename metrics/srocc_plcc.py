
# import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr

# from metrics.metric_util import reorder_image
from utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_srocc(img1, img2, crop_border=0):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert len(img1.shape) == 2, (f'Image must be gray image.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img1 = img1.astype(np.float64).reshape(-1)
    img2 = img2.astype(np.float64).reshape(-1)

    srocc, pval = spearmanr(img1, img2)

    return srocc


@METRIC_REGISTRY.register()
def calculate_plcc(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert len(img1.shape) == 2, (f'Image must be gray image.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img1 = img1.astype(np.float64).reshape(-1)
    img2 = img2.astype(np.float64).reshape(-1)

    plcc, pval = pearsonr(img1, img2)

    return plcc


# if __name__ == '__main__':
#     a_path = '/home/lhm/data/data/VISTAWeChat/LQ_frames/041/00000002.png'
#     b_path = '/home/lhm/data/data/VISTAWeChat/Mask_frames/041/00000002.png'
#     a = cv2.imread(a_path)
#     a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
#     b = cv2.imread(b_path)
#     b_gray = b[:, :, 0]
#
#     srocc1 = calculate_srocc(a_gray, b_gray)
#     plcc1 = calculate_plcc(a_gray, b_gray)
#
#     srocc2 = calculate_srocc(a_gray, a_gray/255.)
#     plcc2 = calculate_plcc(a_gray, a_gray/255.)
#
#     print(srocc1, plcc1)
#     print(srocc2, plcc2)


