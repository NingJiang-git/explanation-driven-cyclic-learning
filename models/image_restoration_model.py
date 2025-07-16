# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import nibabel as nib
from torch import autograd
import csv
import pandas as pd
from scipy.io import savemat

from torch.nn.parallel import DataParallel, DistributedDataParallel
from define import define_network
from models.base_model import BaseModel
from utils import get_root_logger
import ssl

metric_module = importlib.import_module('metrics')
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

class ImageRestorationModel2(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel2, self).__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.L1 = torch.nn.L1Loss()
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scaler_g = torch.cuda.amp.GradScaler()


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                elif k.startswith('module.generator'): # params in classifier needs gradient to interpret but will be not optimized
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio, 'eps': 1e-3}],
                                                **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.label = data['label'].to(self.device)
        self.name = data['gt_name']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
    

    def compute_generator_loss(self):
        G_losses = OrderedDict() #{}
        clean_label = np.array([1,0,0,0])
        clean_label = torch.from_numpy(clean_label).long().to(self.device)
        pred_label, pred_img = self.net_g(self.lq, self.label)
        G_losses['pixel'] = 10 * self.L1(pred_img.repeat(1,3,1,1), self.gt)
        G_losses['class'] = 0.1* self.criterion(pred_label, clean_label)

        return pred_img, G_losses


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast():
            preds, G_losses = self.compute_generator_loss()
        self.output = preds
        l_total = G_losses['pixel'] + G_losses['class']
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        if torch.any(torch.isnan(l_total)):
            raise ValueError('loss ia nan!!!!!!!')
        self.scaler_g.scale(l_total).backward()

        use_grad_clip = True
        if use_grad_clip:
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)


        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        
        self.log_g_dict = G_losses

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            _, pred = self.net_g(self.lq, self.label)
        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred
        self.net_g.train()

    # for visualization    
    def get_latest_images(self):
        return [self.lq[0], self.output[0], self.name[0], self.gt[0]]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_1 = {
                metric: []
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = val_data['gt_name'][0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            img_restored = visuals['result'].numpy()
            img_corrupted = visuals['lq'].numpy()
            if 'gt' in visuals:
                img_gt = visuals['gt'].numpy()
                del self.gt


            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()


            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                    self.metric_results_1[name].append(getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_))                    


            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
        pbar.close()

        current_metric = 0.
        if with_metrics:
            df = pd.DataFrame(self.metric_results_1)
            df.to_csv('./result.csv', index=False, header=True)

            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter,
                                      tb_logger):
        log_str = f'Validation,\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:,1,:,:].unsqueeze(dim=1).detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[:,1,:,:].unsqueeze(dim=1).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class ImageClassificationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageClassificationModel, self).__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # for multi-label classification
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scaler_g = torch.cuda.amp.GradScaler()


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio, 'eps': 1e-3}],
                                                **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.name = data['gt_name'].to(self.device)
        self.label = data['label'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
    def compute_generator_loss(self):
        G_losses = OrderedDict() #{}
        pred = self.net_g(self.lq)
        label_one_hot = self.label
        G_losses['loss'] = 10 * self.criterion(pred, label_one_hot)
        return pred, G_losses


    def optimize_parameters(self, current_iter):
        # scaler = GradScaler()
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast():
            preds, G_losses = self.compute_generator_loss()

        self.output = preds
        l_total = G_losses['loss']

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        if torch.any(torch.isnan(l_total)):
            raise ValueError('loss ia nan!!!!!!!')
        self.scaler_g.scale(l_total).backward()

        use_grad_clip = True
        if use_grad_clip:
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()
        
        self.log_g_dict = G_losses

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred= self.net_g(self.lq[i:j, :, :, :])

                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    # for visualization    
    def get_latest_images(self):
        return [self.lq[0], self.output[0], self.output1[0], self.name[0], self.gt[0]]


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        # dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_1 = {
                metric: []
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):

            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['label'], **opt_)
                    self.metric_results_1[name].append(getattr(
                        metric_module, metric_type)(visuals['result'], visuals['label'], **opt_))                    


            pbar.update(1)
            pbar.set_description(f'Test {idx}')
            cnt += 1
        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter,
                                      tb_logger):
        log_str = f'Validation,\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['label'] = self.label.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)