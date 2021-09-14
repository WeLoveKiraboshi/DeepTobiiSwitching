import numpy as np
import argparse
import time
import datetime
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import pickle
import tqdm

# for val
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as vmodels

from models.ResnetVAE18 import *
from models.ResnetVAE50 import *
from switching.config_loader import Config
from utils.tb_logger import Logger
from utils.utils import AverageMeter
from switching.dataloader import getTrainingTestingData, getTestData
from loss import VAEloss, VAEloss_NormalContrasive, VAEloss_MarginTripletContrasive, DeepInfoMaxLoss, RegressionLoss,SwAVContrastiveLoss
from utils.torch import to_cpu


# SimCLR
from models.simclr import SimCLR
from models.modules import NT_Xent, get_resnet, NT_Xent_MSE
from models.modules.transformations import TransformsSimCLR
from models.modules.sync_batchnorm import convert_model

#Deep Info Max
from models.DeepInfoMax import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator

# MoCo v1
from models.moco.builder import MoCo

# BYOL v1
import models.BYOL.resnet_base_network
from models.BYOL.BYOLTrainer import BYOLTrainer

#CPC v2 and v1
from models.CPC.cpc_models.CPC import CPC
from models.CPC.cpc_models.MobileNetV2_Encoder import MobileNetV2_Encoder
from models.CPC.cpc_models.ResNetV2_Encoder import PreActResNetN_Encoder
from models.CPC.cpc_models.WideResNet_Encoder import Wide_ResNet_Encoder
from models.CPC.cpc_models.PixelCNN_GIM import PixelCNN

##PCL v1
import models.pcl.loader
import models.pcl.builder


#PIRL v1
from models.pirl.models import pirl_resnet
from models.pirl.random_seed_setter import set_random_generators_seed
from models.pirl.train_test_helper import PIRLModelTrainTest
from torch.optim.lr_scheduler import CosineAnnealingLR

models_func = {
'resnet18': ResNet18_VAE,
'resnet50': ResNet50_VAE,
}

# Argument Parser
parser = argparse.ArgumentParser(description='Project-Kajita Multi camera switching via tobbi sensor using VAE')
parser.add_argument('--cfg', default=None)
parser.add_argument('--meta', default=None)
parser.add_argument('--mode', default='train')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=int, default=0, help='IDs of GPUs to use')

args = parser.parse_args()
cfg = Config(args.cfg, create_dirs=True, meta_id=args.meta)

"""setup"""
dtype = torch.float32
torch.set_default_dtype(dtype)
torch.cuda.memory_summary(device=None, abbreviated=False)

if args.gpus == 2:
    gpus = (0, 1)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')
elif args.gpus == 1:
    device = torch.device('cuda', index=int(args.gpuids)) if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpuids)

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
tb_logger = Logger(cfg.tb_dir)

if cfg.init_train == False and cfg.checkpoint:
    assert os.path.isfile(cfg.checkpoint), \
        "=> no model found at '{}'".format(cfg.checkpoint)
    print("=> loading model '{}'".format(cfg.checkpoint))
    if '.pth' in cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint)
        if type(checkpoint) is dict:
            model_state = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model_state = checkpoint
        model.load_state_dict(model_state)  # if saved_model stat
    elif '.pkl' in cfg.checkpoint:
        model_cp = pickle.load(open(cfg.checkpoint, "rb"))
        model.load_state_dict(model_cp['VAEresnet'], strict=False)
    else:
        print('couln not distinguish file type for checkpoint: {}'.format(cfg.checkpoint))
else:
    """network"""
    print('train init..... this model will be trained from scratch !')
    if cfg.network == 'resnet18' or cfg.network == 'resnet50':
        model = models_func[cfg.network](cfg.z_dim, cfg.v_hdim, cfg.dropout_p, cfg.pre_train)
    elif cfg.network == 'resnet18_simCLR':
        # initialize ResNet
        encoder = get_resnet('resnet18', pretrained=cfg.pre_train)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, cfg.z_dim, n_features)
    elif cfg.network == 'resnet50_simCLR':
        # initialize ResNet
        encoder = get_resnet('resnet50', pretrained=cfg.pre_train)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, cfg.z_dim, n_features)
    elif cfg.network == 'deep_infomax':
        # initialize ResNet
        model= Encoder()
    elif cfg.network == 'MoCo_v1_resnet18':
    	#encoder = get_resnet('resnet18', pretrained=cfg.pre_train)
    	model = MoCo(vmodels.__dict__['resnet18'], cfg.moco_dim, cfg.moco_k, cfg.moco_m, cfg.moco_t, cfg.moco_mlp)
    elif cfg.network == 'BYOL_v1_resnet18':
        online_network = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=cfg.pre_train, hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
        target_network = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=False, hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
        # predictor network
        predictor =  models.BYOL.mlp_head.MLPHead(in_channels=online_network.projetion.net[-1].out_features, mlp_hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
    elif cfg.network == 'CPC_v1_resnet18':
        # Define Autrogressive Network
        enc = PreActResNetN_Encoder(encoder='resnet18', use_classifier=False, cfg=cfg)
        ar = PixelCNN(in_channels=enc.encoding_size)
        model = CPC(enc, ar, cfg.pred_directions, cfg.pred_steps, cfg.neg_samples)
        #model.load_state_dict()
    elif cfg.network == 'PCL_v1_resnet18':
        model = models.pcl.builder.MoCo(vmodels.__dict__['resnet18'], cfg.low_dim, cfg.pcl_r, cfg.moco_m, cfg.temperature, cfg.mlp)
    elif cfg.network == 'PIRL_v1_resnet18':
        # If using Resnet18
        model = pirl_resnet('res18', cfg.non_linear_head)
    else:
        print('none model indicated')
        exit(0)


torch.backends.cudnn.benchmark = True

if cfg.network == 'BYOL_v1_resnet18':
    pass
else:
    if args.gpus == 2:
        model = torch.nn.DataParallel(model)  # make parallel
    elif args.gpus == 1:
        model.to(device)



if cfg.loss == "VAELoss":
    loss_criteria = VAEloss()
elif cfg.loss == "VAELossWithNormalContrasive":
    loss_criteria = VAEloss_NormalContrasive(alpha=cfg.contrastive_alpha, beta=cfg.contrastive_beta, gamma=cfg.contrastive_gamma)
elif cfg.loss == 'VAELossWithMarginTripletContrastive':
    loss_criteria = VAEloss_MarginTripletContrasive(alpha=cfg.contrastive_alpha, beta=cfg.contrastive_beta,
                                             gamma=cfg.contrastive_gamma)
elif cfg.loss == 'NT_Xent':
    world_size = 1 * args.gpus
    loss_criteria = NT_Xent(cfg.bs, cfg.temperature, world_size)
elif cfg.loss == 'NT_Xent_MSE':
    world_size = 1 * args.gpus
    loss_criteria = NT_Xent_MSE(cfg.bs, cfg.temperature, world_size)
elif cfg.loss == 'DeepInfoMaxLoss':
    loss_criteria = DeepInfoMaxLoss(alpha=0.5, beta=1.0, gamma=0.1)
elif cfg.loss == 'BinaryCrossEntropy':
    loss_criteria = nn.CrossEntropyLoss().cuda(args.gpuids)
elif cfg.loss == 'RegressionLoss':
     loss_criteria = RegressionLoss()
elif cfg.loss == 'PCL_v1':
    pass

if cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
elif cfg.optimizer == 'SGD':
    if cfg.network == 'BYOL_v1_resnet18':
        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),lr=cfg.lr, momentum = cfg.momentum, weight_decay=cfg.weightdecay)
    elif cfg.network == 'PIRL_v1_resnet18':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weightdecay)
        scheduler = CosineAnnealingLR(optimizer, cfg.tmax_for_cos_decay, eta_min=1e-4, last_epoch=-1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum = cfg.momentum, weight_decay=cfg.weightdecay)
else:
    print('None optimizer indicated')
    exit(0)

if cfg.network == 'BYOL_v1_resnet18':
    if args.mode == 'train':
        online_network.train()
        target_network.train()
        predictor.train()
    else:
        online_network.eval()
        target_network.eval()
        predictor.eval()
else:
    if args.mode == 'train':
        model.train()
    else:
        model.eval()




from torchsummary import summary
# model_ft = vmodels.resnet18(pretrained=True).to(device)
# # # (batch_size, grid_size, grid_size, channels, patch_size, patch_size),(7, 7, 3, 56, 56)
#summary(model, [(3, 224, 224), (9, 3, 64, 64)])
# print(vmodels.__dict__['resnet18'])

def run_epoch(dataset, mode='train', epoch=0):
    global model, optimizer, loss_criteria
    batch_time = AverageMeter()
    losses = AverageMeter()
    contrastive_losses = AverageMeter()
    N = len(dataset)
    end = time.time()

    for i, sample_batched in enumerate(dataset):
        if len(sample_batched) == 1: #input is only Tobii image
            image_tobii = torch.autograd.Variable(sample_batched["tobii"].cuda())
            if cfg.network == 'deep_infomax':
                y, M = model.forward(image_tobii)
                # rotate images to create pairs for comparison
                M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                loss = loss_criteria(y, M, M_prime)
            elif cfg.network == 'CPC_v1_resnet18':
                # from torchvision.utils import save_image
                # for i in range(image_tobii.shape[1]):
                #     for j in range(image_tobii.shape[2]):
                #         sample = image_tobii[0, i, j, :, :, :] #bs, 27, 27, 3, 16, 16
                #         print(sample.shape)
                #         save_image(sample.squeeze()[[2, 1, 0]], 'tobii_'+str(i)+'_'+str(j)+'.png')
                loss = model(image_tobii)
                loss = torch.mean(loss, dim=0)  # take mean over all GPUs
            else:
                output, mu, logvar, z = model.forward(image_tobii)
                loss = loss_criteria(output, image_tobii, mu, logvar)
            losses.update(loss.data.item(), image_tobii.size(0))
        elif len(sample_batched) == 2:
            image_tobii= torch.autograd.Variable(sample_batched["tobii"].cuda(args.gpuids, non_blocking=True))
            image_augmented_tobii = torch.autograd.Variable(sample_batched["augmented_tobii"].cuda(args.gpuids, non_blocking=True))
            if cfg.network == 'resnet18' or cfg.network == 'resnet50':
                output, mu, logvar,z = model.forward(image_tobii)
                output_aug, mu_aug, logvar_aug,z_aug = model.forward(image_augmented_tobii)
                tobii_elem = {'recon_x': output, 'x': image_tobii, 'mu': mu, 'logvar': logvar, 'z': z}
                tobii_aug_elem = {'recon_x': output_aug, 'x': image_augmented_tobii, 'mu': mu_aug, 'logvar': logvar_aug,
                                  'z': z_aug}
                loss = loss_criteria(tobii_elem, tobii_aug_elem)
            elif cfg.network == 'resnet18_simCLR' or cfg.network == 'resnet50_simCLR':
                # positive pair, with encoding
                h_i, h_j, z_i, z_j = model(image_tobii, image_augmented_tobii)
                loss = loss_criteria(z_i, z_j)
            elif cfg.network == 'MoCo_v1_resnet18':
                output, target = model(im_q=image_tobii, im_k=image_augmented_tobii)
                loss = loss_criteria(output, target)
            losses.update(loss.data.item(), image_tobii.size(0))
        elif len(sample_batched) == 3:
            image_tobii = torch.autograd.Variable(sample_batched["tobii"].cuda())
            augmented_tobii_pos = torch.autograd.Variable(sample_batched["augmented_tobii"].cuda())
            augmented_tobii_neg = torch.autograd.Variable(sample_batched["augmented_tobii_neg"].cuda())
            # import matplotlib.pyplot as plt
            # for idx in range(8):
            #     i = np.random.randint(0, 64)
            #     img = sample_batched["augmented_tobii_neg"][i]
            #     img = img.permute(1, 2, 0)
            #     img = img[:, :, (2, 1, 0)]
            #     plt.imshow(img)
            #     plt.show()
            output, mu, logvar, z = model.forward(image_tobii)
            output_aug_pos, mu_aug_pos, logvar_aug_pos, z_aug_pos = model.forward(augmented_tobii_pos)
            output_aug_neg, mu_aug_neg, logvar_aug_neg, z_aug_neg = model.forward(augmented_tobii_neg)
            tobii_elem = {'recon_x': output, 'x': image_tobii, 'mu': mu, 'logvar': logvar, 'z': z}
            tobii_aug_elem_pos = {'recon_x': output_aug_pos, 'x': augmented_tobii_pos, 'mu': mu_aug_pos, 'logvar': logvar_aug_pos,
                              'z': z_aug_pos}
            tobii_aug_elem_neg = {'recon_x': output_aug_neg, 'x': augmented_tobii_neg, 'mu': mu_aug_neg,
                              'logvar': logvar_aug_neg, 'z': z_aug_neg}
            loss, contrastive_loss = loss_criteria(tobii_elem, tobii_aug_elem_pos, tobii_aug_elem_neg)
            losses.update(loss.data.item(), image_tobii.size(0))
            contrastive_losses.update(contrastive_loss.data.item(), image_tobii.size(0))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        if i % cfg.tb_log_interval == 0:
            if cfg.loss == "VAELoss":
                tb_logger.scalar_summary('VAELoss_'+mode, losses.val, niter)
            elif cfg.loss == "VAELossWithNormalContrastive":
                tb_logger.scalar_summary('VAELoss_' + mode, losses.val, niter)
                tb_logger.scalar_summary('ContrastiveLoss_' + mode,
                                         torch.mean(torch.sum(torch.pow(z - z_aug, 2) / 2, axis=-1)), niter)
            elif cfg.loss == 'NT_Xent':
                tb_logger.scalar_summary(cfg.loss+'_Loss_' + mode, losses.val, niter)
            elif cfg.loss == 'BinaryCrossEntropy':
                tb_logger.scalar_summary(cfg.loss + '_Loss_' + mode, losses.val, niter)
            elif cfg.loss == "VAELossWithMarginTripletContrastive":
                tb_logger.scalar_summary('VAELoss_' + mode, losses.val, niter)
                tb_logger.scalar_summary('ContrastiveLoss_' + mode, contrastive_losses.val , niter)
            elif cfg.network == 'CPC_v1_resnet18':
                tb_logger.scalar_summary('CPC_ContrastiveLoss_' + mode, losses.val, niter)
            elif cfg.network == 'SwAV_v1_resnet18':
                tb_logger.scalar_summary('CPC_ContrastiveLoss_' + mode, losses.val, niter)



        """clean up gpu memory"""
        torch.cuda.empty_cache()
        if len(sample_batched) == 1:
            if cfg.network == 'deep_infomax':
                pass
            elif cfg.network == 'CPC_v1_resnet18':
                del image_tobii
        elif len(sample_batched) == 2:
            if cfg.network == 'resnet18' or cfg.network == 'resnet50':
                del image_tobii, image_augmented_tobii, output, output_aug, mu, logvar, z, mu_aug, logvar_aug, z_aug
            elif cfg.network == 'resnet18_simCLR' or cfg.network == 'resnet50_simCLR':
                del image_tobii, image_augmented_tobii, h_i, h_j, z_i, z_j
            elif cfg.network == 'MoCo_v1_resnet18':
                del image_tobii, image_augmented_tobii, output, target

    #tb_logger.LogProgressImage_tobii(model, mode, dataset, epoch)
    if mode == 'train':
        with to_cpu(model):
            if epoch % cfg.save_model_interval == 0:
                model_path = cfg.model_dir + "ckpt_{}_{}.pkl".format(mode, int(epoch + 1))
                model_cp = {'VAEresnet': model.state_dict()}
                pickle.dump(model_cp, open(model_path, 'wb'))


def train():
    if cfg.network == 'BYOL_v1_resnet18':
        trainer = BYOLTrainer(online_network=online_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    predictor=predictor,
                    device=device,
                    tb_logger = tb_logger,
                    config = cfg,
                    modes = args.mode)
        #for epoch in range(cfg.epochs):
        trainer.train()
            #trainer.train(dataloader_val, mode='val', epoch=epoch)
    elif cfg.network == 'PIRL_v1_resnet18':
        model_train_test_obj = PIRLModelTrainTest(
            network=model, device=device, cfg=cfg, modes = args.mode,optimizer=optimizer,lrscheduler=scheduler, tb_logger=tb_logger )
        model_train_test_obj.train()
    else:
        dataloader_train, dataloader_val, train_image_indices, val_image_indices = getTrainingTestingData(mode=args.mode, batch_size=cfg.bs,
                                                                  is_shuffle=cfg.shuffle, seed=cfg.seed,
                                                                  base_dir=cfg.data_dir, imsize=cfg.imsize,
                                                                  takes=cfg.takes, is_augment=cfg.augment,
                                                                  augment_type=cfg.augment_type,
                                                                  separate_augment=cfg.separate_augment,
                                                                  num_worker=cfg.num_worker,
                                                                  cfg = cfg)
        for epoch in range(cfg.epochs):
            run_epoch(dataloader_train, mode='train', epoch=epoch)
            run_epoch(dataloader_val, mode='val', epoch=epoch)

    print('Finished Training. model saved ')

if __name__ == '__main__':
    train()



