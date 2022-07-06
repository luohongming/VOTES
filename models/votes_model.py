
import logging
from torch.nn.parallel import DistributedDataParallel
from metrics import calculate_metric
from utils import imwrite, tensor2img
from utils.dist_util import get_dist_info
from collections import OrderedDict

from torch import distributed as dist
from tqdm import tqdm
from collections import Counter
from os import path as osp

from utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
import torch

logger = logging.getLogger('basicsr')

@MODEL_REGISTRY.register()
class VOTESModel(VideoBaseModel):
    def __init__(self, opt):
        super(VOTESModel, self).__init__(opt)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            if len(self.gt.size()) == 5 and self.gt.size(1) == 1:
                self.gt = self.gt[:, 0, ...]

        # if 'mask' in data:
        #     self.mask = data['mask'].to(self.device)

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            # l_pix.backward()
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics and not hasattr(self, 'metrics_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        # record all frames (border and center frames)
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='frame')

        for idx in range(rank, len(dataset), world_size):
            val_data = dataset[idx]
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            # val_data['mask'].unsqueeze_(0)

            folder = val_data['folder']
            frame_idx, max_idx = val_data['idx'].split('/')
            lq_path = val_data['lq_path']

            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            result_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            # del self.mask

            torch.cuda.empty_cache()

            if save_img:
                # if self.opt['is_train']:
                #     raise NotImplementedError('saving image is not supported during training.')
                # else:
                img_name = osp.splitext(osp.basename(lq_path))[0]
                # print(folder, img_name)
                if self.opt['val'].get('suffix', False):
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                             f'{img_name}_{self.opt["val"]["suffix"]}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                             f'{img_name}.png')

                imwrite(result_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                    metric_data = dict(img1=result_img, img2=gt_img)
                    result = calculate_metric(metric_data, opt_)
                    self.metric_results[folder][int(frame_idx), metric_idx] += result
            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {folder}:' f'{int(frame_idx) + world_size}/{max_idx}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()
            else:
                pass  # assume use one gpu in non-dist testing

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

