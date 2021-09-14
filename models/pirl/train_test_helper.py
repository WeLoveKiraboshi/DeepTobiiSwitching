import torch
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_

from models.pirl.pirl_loss import loss_pirl, get_img_pair_probs

import time
import datetime
from utils.utils import AverageMeter

import pickle
from utils.torch import to_cpu

from switching.dataloader import getTrainingTestingData, getTestData


def get_count_correct_preds(network_output, target):

    score, predicted = torch.max(network_output, 1)  # Returns max score and the index where max score was recorded
    count_correct = (target == predicted).sum().float()  # So that when accuracy is computed, it is not rounded to int

    return count_correct


def get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr):
    """
    Get count of correct predictions for pre-text task
    :param img_pair_probs_arr: Prob vector of batch of images I and I_t to belong to same data distribution.
    :param img_mem_rep_probs_arr: Prob vector of batch of I and mem_bank_rep of I to belong to same data distribution
    """

    avg_probs_arr = (1/2) * (img_pair_probs_arr + img_mem_rep_probs_arr)
    count_correct = (avg_probs_arr >= 0.5).sum().float()  # So that when accuracy is computed, it is not rounded to int

    return count_correct.item()


class PIRLModelTrainTest():

    def __init__(self, network, device, cfg, threshold=1e-4, modes='train',optimizer=None,only_train=False,lrscheduler=None,tb_logger=None):
                 #train_image_indices,
                 #val_image_indices, count_negatives, temp_parameter, beta, only_train=False, threshold=1e-4):
        super(PIRLModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.cfg = cfg
        self.mode = modes
        self.optimizer = optimizer
        self.lrscheduler=lrscheduler
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9
        self.params_max_norm = 4
        self.tb_logger = tb_logger
        self.train_data_loader, self.val_data_loader, self.train_image_indices, self.val_image_indices = getTrainingTestingData(mode=self.mode, batch_size=cfg.bs,
                                                                  is_shuffle=cfg.shuffle, seed=cfg.seed,
                                                                  base_dir=cfg.data_dir, imsize=cfg.imsize,
                                                                  takes=cfg.takes, is_augment=cfg.augment,
                                                                  augment_type=cfg.augment_type,
                                                                  separate_augment=cfg.separate_augment,
                                                                  num_worker=cfg.num_worker,
                                                                  cfg = cfg)
        len_train_val_set=len(self.train_image_indices) + len(self.val_image_indices)
        all_images_mem = np.random.randn(len_train_val_set, 64)
        self.all_images_mem = torch.tensor(all_images_mem, dtype=torch.float).to(device)
        self.count_negatives = cfg.count_negatives
        self.temp_parameter = cfg.temp_parameter
        self.beta = cfg.beta
        self.only_train = only_train

    def train_epoch(self, epoch):
        self.network.train()
        correct = 0
        no_train_samples = len(self.train_image_indices)
        no_val_samples = len(self.val_image_indices)

        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(self.train_data_loader)
        end = time.time()

        for batch_idx, (data_batch, batch_img_indices) in enumerate(self.train_data_loader):
            # Separate input image I batch and transformed image I_t batch (jigsaw patches) from data_batch
            i_batch, i_t_patches_batch = data_batch[0], data_batch[1]
            # import matplotlib.pyplot as plt
            # from torchvision.utils import save_image
            # for idx in range(9):
            #     img = i_t_patches_batch[0][idx] # 9, 3, 60, 60
            #     # img = img.permute(1, 2, 0)
            #     # img = img[:, :, (2, 1, 0)]
            #     print(img.shape)
            #     save_image(img.squeeze()[[2, 1, 0]], 'tobii_' + str(idx) + '.png')

            # Set device for i_batch, i_t_patches_batch and batch_img_indices
            i_batch, i_t_patches_batch = i_batch.to(self.device), i_t_patches_batch.to(self.device)
            batch_img_indices = batch_img_indices.to(self.device)

            # Forward pass through the network
            self.optimizer.zero_grad()
            vi_batch, vi_t_batch = self.network(i_batch, i_t_patches_batch)

            # Prepare memory bank of negatives for current batch
            np.random.shuffle(self.train_image_indices)
            mn_indices_all = np.array(list(set(self.train_image_indices) - set(batch_img_indices)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:self.count_negatives]
            mn_arr = self.all_images_mem[mn_indices]

            # Get memory bank representation for current batch images
            mem_rep_of_batch_imgs = self.all_images_mem[batch_img_indices]

            # Get prob for I, I_t to belong to same data distribution.
            img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, self.temp_parameter)

            # Get prob for I and mem_bank_rep of I to belong to same data distribution
            img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, self.temp_parameter)

            # Compute loss => back-prop gradients => Update weights
            loss = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)
            loss.backward()

            clip_grad_norm_(self.network.parameters(), self.params_max_norm)
            self.optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - batch_idx))))

            # Log progress
            losses.update(loss.data.item(), i_batch.size(0))
            niter = epoch * N + batch_idx
            if batch_idx % 50 == 0:
                # Print to console
                print('{mode} Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, batch_idx, N, batch_time=batch_time, loss=losses, eta=eta, mode=self.mode))
            if batch_idx % self.cfg.tb_log_interval == 0:
                self.tb_logger.scalar_summary('PIRL_Loss_' + self.mode, losses.val, niter)

            # Update running loss and no of pseudo correct predictions for epoch
            correct += get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr)

            # Update memory bank representation for images from current batch
            all_images_mem_new = self.all_images_mem.clone().detach()
            all_images_mem_new[batch_img_indices] = (self.beta * all_images_mem_new[batch_img_indices]) + \
                                                    ((1 - self.beta) * vi_batch)
            self.all_images_mem = all_images_mem_new.clone().detach()

            del i_batch, i_t_patches_batch, vi_batch, vi_t_batch, mn_arr, mem_rep_of_batch_imgs
            del img_mem_rep_probs_arr, img_pair_probs_arr

        if epoch % self.cfg.save_model_interval == 0:
            model_path = self.cfg.model_dir + "ckpt_{}_{}.pkl".format(self.mode, int(epoch + 1))
            model_cp = {'VAEresnet': self.network.state_dict()}
            pickle.dump(model_cp, open(model_path, 'wb'))

        if self.only_train is False:
            self.test(epoch, self.val_data_loader, no_val_samples)

        train_acc = correct / no_train_samples

        # Log progress
        self.tb_logger.scalar_summary('epoch_PIRL_Acc_' + self.mode, train_acc, epoch)
        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
             epoch, losses.val, correct, no_train_samples, 100. * correct / no_train_samples))



    def train(self):
        for epoch in range(self.cfg.epochs):
            self.train_epoch(epoch)
            self.lrscheduler.step()

    def test(self, epoch, test_data_loader, no_test_samples):

        self.network.eval()
        correct = 0
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(test_data_loader)
        end = time.time()

        for batch_idx, (data_batch, batch_img_indices) in enumerate(test_data_loader):

            # Separate input image I batch and transformed image I_t batch (jigsaw patches) from data_batch
            i_batch, i_t_patches_batch = data_batch[0], data_batch[1]

            # Set device for i_batch, i_t_patches_batch and batch_img_indices
            i_batch, i_t_patches_batch = i_batch.to(self.device), i_t_patches_batch.to(self.device)
            batch_img_indices = batch_img_indices.to(self.device)

            # Forward pass through the network
            vi_batch, vi_t_batch = self.network(i_batch, i_t_patches_batch)

            # Prepare memory bank of negatives for current batch
            np.random.shuffle(self.val_image_indices)

            mn_indices_all = np.array(list(set(self.val_image_indices) - set(batch_img_indices)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:self.count_negatives]
            mn_arr = self.all_images_mem[mn_indices]

            # Get memory bank representation for current batch images
            mem_rep_of_batch_imgs = self.all_images_mem[batch_img_indices]

            # Get prob for I, I_t to belong to same data distribution.
            img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, self.temp_parameter)

            # Get prob for I and mem_bank_rep of I to belong to same data distribution
            img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, self.temp_parameter)

            # Compute loss
            loss = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - batch_idx))))

            # Log progress
            losses.update(loss.data.item(), i_batch.size(0))
            niter = epoch * N + batch_idx
            if batch_idx % 50 == 0:
                # Print to console
                print('{mode} Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, batch_idx, N, batch_time=batch_time, loss=losses, eta=eta, mode='val'))
            if batch_idx % self.cfg.tb_log_interval == 0:
                self.tb_logger.scalar_summary('PIRL_Loss_val', losses.val, niter)

            # Update running loss and no of pseudo correct predictions for epoch
            correct += get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr)

            # Update memory bank representation for images from current batch
            all_images_mem_new = self.all_images_mem.clone().detach()
            all_images_mem_new[batch_img_indices] = (self.beta * all_images_mem_new[batch_img_indices]) + \
                                                    ((1 - self.beta) * vi_batch)
            self.all_images_mem = all_images_mem_new.clone().detach()


            del i_batch, i_t_patches_batch, vi_batch, vi_t_batch, mn_arr, mem_rep_of_batch_imgs
            del img_mem_rep_probs_arr, img_pair_probs_arr

        test_acc = correct / no_test_samples
        # Log progress
        self.tb_logger.scalar_summary('epoch_PIRL_Acc_val', test_acc, epoch)
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, losses.val, correct, no_test_samples, 100. * correct / no_test_samples))



class ModelTrainTest():

    def __init__(self, network, device, model_file_path, threshold=1e-4):
        super(ModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader,
              no_train_samples, no_val_samples):
        self.network.train()
        train_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.network(data)

            loss = F.nll_loss(output, target)
            loss.backward()

            clip_grad_norm_(self.network.parameters(), params_max_norm)
            optimizer.step()

            correct += get_count_correct_preds(output, target)
            train_loss += loss.item()
            cnt_batches += 1

            del data, target, output

        train_loss /= cnt_batches
        val_loss, val_acc = self.test(epoch, val_data_loader, no_val_samples)

        if val_loss < self.val_loss - self.threshold:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), self.model_file_path)

        train_acc = correct / no_train_samples

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, train_loss, correct, no_train_samples, 100. * correct / no_train_samples))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader, no_test_samples):
        self.network.eval()
        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss

            correct += get_count_correct_preds(output, target)

            del data, target, output

        test_loss /= no_test_samples
        test_acc = correct / no_test_samples
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, no_test_samples, 100. * correct / no_test_samples))

        return  test_loss, test_acc

if __name__ == '__main__':
    img_pair_probs_arr = torch.randn((256,))
    img_mem_rep_probs_arr = torch.randn((256,))
    print (get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr))