import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from utils.utils import AverageMeter

import pickle
from utils.torch import to_cpu

from switching.dataloader import getTrainingTestingData, getTestData

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, tb_logger, config, modes):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.tb_logger = tb_logger
        self.cfg = config
        self.m = self.cfg.BYOL_m

        self.dataloader_train, self.dataloader_val = getTrainingTestingData(mode=modes, batch_size=self.cfg.bs,
                                                                  is_shuffle=self.cfg.shuffle, seed=self.cfg.seed,
                                                                  base_dir=self.cfg.data_dir, imsize=self.cfg.imsize,
                                                                  takes=self.cfg.takes, is_augment=self.cfg.augment,
                                                                  augment_type=self.cfg.augment_type,
                                                                  num_worker=self.cfg.num_worker,
                                                                  augment_type_neg=self.cfg.augment_type_neg,
                                                                  contrastive_mode=self.cfg.contrastive_mode)


        # self.m = params['m']
        # self.checkpoint_interval = params['checkpoint_interval']


    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train_epoch(self, dataloader, mode='train', epoch=0):

        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(dataloader)
        end = time.time()
        for i, sample_batched in enumerate(dataloader):
            batch_view_1 = sample_batched["tobii"].to(self.device)
            batch_view_2 = sample_batched["augmented_tobii"].to(self.device)
            # compute query feature
            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
            # compute key features
            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)
            loss_ = self.regression_loss(predictions_from_view_1, targets_to_view_1)
            loss_ += self.regression_loss(predictions_from_view_2, targets_to_view_2)
            loss = loss_.mean()
            losses.update(loss.data.item(), batch_view_1.size(0))

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_target_network_parameters()  # update the key encoder

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))
            # Log progress
            niter = epoch * N + i
            if i % 50 == 0:
                # Print to console
                print('{mode} Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta, mode=mode))
            if i % self.cfg.tb_log_interval == 0:
                self.tb_logger.scalar_summary('BYOL_RegressionLoss_' + mode, losses.val, niter)

            """clean up gpu memory"""
            torch.cuda.empty_cache()
            del  batch_view_1,  batch_view_2, predictions_from_view_1, predictions_from_view_2, targets_to_view_2, targets_to_view_1
        if mode == 'train':
            #with to_cpu(self.online_network) and to_cpu(self.target_network):
            if epoch % self.cfg.save_model_interval == 0:
                model_path = self.cfg.model_dir + "ckpt_{}_{}.pkl".format(mode, int(epoch + 1))
                model_cp = {'online_network_state_dict': self.online_network.state_dict(),
                            'target_network_state_dict': self.target_network.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}
                pickle.dump(model_cp, open(model_path, 'wb'))
        print('finihed. epoch={}, mode={}'.format(epoch, mode))

    def train(self):
        self.initializes_target_network()
        for epoch in range(self.cfg.epochs):
            self.train_epoch(self.dataloader_train, mode='train', epoch=epoch)
            #self.train_epoch(self.dataloader_val, mode='val', epoch=epoch)


