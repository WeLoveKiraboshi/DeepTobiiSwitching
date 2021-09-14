import numpy as np
import argparse
import time
import datetime
import os
import sys
sys.path.append(os.getcwd())
import pickle
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models.ResnetVAE18 import *
from models.ResnetVAE50 import *
from switching.config_loader import Config
from switching.dataloader import getTestData
from utils.tb_logger import Logger

from switching.eval.eval_metrics import TripletLoss, MSE, KLdiv, NT_Xent, DotSim
from switching.eval.eval_utils import calc_confusion_matrix_my
from tqdm import tqdm

from utils.utils import AverageMeter
import torchvision.models as vmodels


# SimCLR
from models.simclr import SimCLR
from models.modules import NT_Xent, get_resnet
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

#SwAv resnet18
from models.SwAV.src.resnet18 import SwAV_Resnet18
from models.SwAV.src.utils import init_distributed_mode

##PCL v1
import models.pcl.loader
import models.pcl.builder
import torch.distributed as dist
import faiss

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
parser.add_argument('--mode', default='test')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=int, default=0, help='IDs of GPUs to use')
parser.add_argument('--cfg', default=None)
args = parser.parse_args()
"""setup"""
dtype = torch.float32
torch.set_default_dtype(dtype)
torch.backends.cudnn.benchmark = True
if args.gpus == 2:
    gpus = (0, 1)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')
elif args.gpus == 1:
    device = torch.device('cuda', index=int(args.gpuids)) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpuids)

def set_model(meta_):
    cfg = Config(args.cfg, create_dirs=False, meta_id=meta_)
    if cfg.network == 'resnet18' or cfg.network == 'resnet50':
        model = models_func[cfg.network](cfg.z_dim, cfg.v_hdim, cfg.dropout_p, cfg.pre_train)
    elif cfg.network == 'resnet18_simCLR':
        encoder = get_resnet('resnet18', pretrained=cfg.pre_train)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, cfg.z_dim, n_features)
    elif cfg.network == 'resnet50_simCLR':
        encoder = get_resnet('resnet50', pretrained=cfg.pre_train)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, cfg.z_dim, n_features)
    elif cfg.network == 'deep_infomax':
        # initialize ResNet
        model= Encoder()
    elif cfg.network == 'MoCo_v1_resnet18':
        model = MoCo(vmodels.__dict__['resnet18'], cfg.moco_dim, cfg.moco_k, cfg.moco_m, cfg.moco_t, cfg.moco_mlp)
    elif cfg.network == 'BYOL_v1_resnet18':
        model = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=cfg.pre_train, hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
        #target_network = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=False, hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
        # predictor network
        #predictor =  models.BYOL.mlp_head.MLPHead(in_channels=online_network.projetion.net[-1].out_features, mlp_hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
    elif cfg.network == 'SwAV_v1_resnet18':
        #init_distributed_mode(args)
        encoder = get_resnet('resnet18', pretrained=cfg.pre_train)
        model = SwAV_Resnet18(encoder=encoder, output_dim=cfg.z_dim, hidden_mlp=cfg.v_hdim, nmb_prototypes=cfg.nmb_prototypes, eval_mode=True)
        #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif cfg.network == 'CPC_v1_resnet18':
        # Define Autrogressive Network
        enc = PreActResNetN_Encoder(encoder='resnet18', use_classifier=False, cfg=cfg, eval=True)
        ar = PixelCNN(in_channels=enc.encoding_size)
        model = CPC(enc, ar, cfg.pred_directions, cfg.pred_steps, cfg.neg_samples, is_eval=True)
    elif cfg.network == 'PCL_v1_resnet18':
        model = models.pcl.builder.MoCo(vmodels.__dict__['resnet18'], cfg.low_dim, cfg.pcl_r, cfg.moco_m, cfg.temperature, cfg.mlp)
    elif cfg.network == 'PIRL_v1_resnet18':
        # If using Resnet18
        model = pirl_resnet('res18', cfg.non_linear_head)

    if cfg.checkpoint != None:
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
            if cfg.network == 'BYOL_v1_resnet18':
                model_cp = pickle.load(open(cfg.checkpoint, "rb"))
                model.load_state_dict(model_cp['online_network_state_dict'], strict=False)
            else:
                if cfg.network == 'CPC_v1_resnet18':
                    model_cp = pickle.load(open(cfg.checkpoint, "rb"))
                    #print(model_cp['VAEresnet'].keys())
                    #print(model_cp['VAEresnet']['enc.conv1.weight'])
                    model.load_state_dict(model_cp['VAEresnet'], strict=False)
                else:
                    model_cp = pickle.load(open(cfg.checkpoint, "rb"))
                    model.load_state_dict(model_cp['VAEresnet'], strict=False)
        else:
            print('couln not distinguish file type for checkpoin: {}'.format(cfg.checkpoint))
    else:
        print('no trained model indicated...')
        print('would you like to continue this test without any save model ? [y or n] ')
        str = input()
        if str == 'y':
            print('network [{}], weight pretrained ImageNet= {}'.format(cfg.network, cfg.pre_train))
            pass
        else:
            exit(0)
    return model.to(device), cfg


def run_epoch(model, dataset, mode='test', epoch=0,cfg=None):
    from torchsummary import summary
    # # model_ft = vmodels.resnet18(pretrained=True).to(device)
    # # # # (batch_size, grid_size, grid_size, channels, patch_size, patch_size),(7, 7, 3, 56, 56)
    summary(model, (3, 224, 224))


    if cfg.metrics == 'mse':
        lower_better = True
        metrics_criteria = MSE()
    elif cfg.metrics == 'cos':
        lower_better = False
        metrics_criteria = nn.CosineSimilarity(dim=0, eps=1e-6)
    model.eval()
    N = len(dataset)
    resultlist = []
    #recon loss evaluate
    mse = nn.MSELoss(reduce=False)
    losses_tobii = AverageMeter()
    losses_cams = AverageMeter()


    with torch.no_grad():
        for i, sample_batched in tqdm(enumerate(dataset)):
            image_tobii = sample_batched["tobii"].cuda() #torch.autograd.Variable(sample_batched["tobii"].cuda())
            image_cams = torch.cat([sample_batched['cam1'], sample_batched['cam2'],sample_batched['cam3'],sample_batched['cam4'],sample_batched['cam5']], 0).cuda()
            if cfg.network == 'resnet18' or cfg.network == 'resnet50':
                output_t, mu_t, logvar_t, z_t = model.forward(image_tobii)
                output_c, mu_c, logvar_c, z_c = model.forward(image_cams)
                z_t = z_t.squeeze(0)

                recon_loss_tobii = torch.sum(torch.sum(torch.sum(mse(output_t, image_tobii), axis=-1), axis=-1),
                                             axis=-1) / (3 * 224 * 224)
                recon_loss_cams = torch.mean(
                    torch.sum(torch.sum(torch.sum(mse(output_c, image_cams), axis=-1), axis=-1), axis=-1) / (
                                3 * 224 * 224))
                losses_tobii.update(recon_loss_tobii.data.item(), image_tobii.size(0))
                losses_cams.update(recon_loss_cams.data.item(), image_tobii.size(0))
            elif cfg.network == 'resnet18_simCLR' or cfg.network == 'resnet50_simCLR':
                h_t, h_c, z_t, z_c = model.forward(image_tobii, image_cams)
                z_t = z_t.squeeze(0)
            elif cfg.network == 'MoCo_v1_resnet18':
                z_t = model.test_foward(im_q=image_tobii)
                z_c = model.test_foward(im_q=image_cams)
            elif cfg.network == 'BYOL_v1_resnet18':
                z_t = model(image_tobii)
                z_c = model(image_cams)
            elif cfg.network == 'SwAV_v1_resnet18':
                z_t = model(image_tobii)
                z_c = model(image_cams)
            elif cfg.network == 'CPC_v1_resnet18':
                z_t = model(image_tobii)[:, :, 0, 0]
                z_c = model(image_cams)[:, :, 0, 0]
            elif cfg.network == 'PCL_v1_resnet18':
                z_t = model.test_foward(image_tobii)
                z_c = model.test_foward(image_cams)
            elif cfg.network == 'PIRL_v1_resnet18':
                z_t = model.test_foward(image_tobii)
                z_c = model.test_foward(image_cams)
            # print(z_t.shape)
            # print(z_c.shape)


            sort_list = torch.stack(
                [metrics_criteria(z_t, z_c[0, :]),
                 metrics_criteria(z_t, z_c[1, :]),
                 metrics_criteria(z_t, z_c[2, :]),
                 metrics_criteria(z_t, z_c[3, :]),
                 metrics_criteria(z_t, z_c[4, :])])
            if lower_better:
                sorted_list = torch.argsort(sort_list, dim=0)
            else:
                sorted_list = torch.argsort(-sort_list, dim=0)
            pred_idx = int(sorted_list[0] + 1)

            resultlist.append(
                [pred_idx, int(sample_batched["gt_label"][0][0]), int(sample_batched["gt_label"][0][1]), int(sample_batched["gt_label"][0][2]),
                 int(sample_batched["gt_label"][0][3]), int(sample_batched["gt_label"][0][4])]
            )
            #print('[{}] --- selected cam id : cam {} '.format(i, pred_idx))

            """clean up gpu memory"""
            torch.cuda.empty_cache()
            del image_tobii
        calc_confusion_matrix_my(resultlist)
    print('mse reconstruction loss : [tobii]={loss_t.val:.5f}    [cams]={loss_c.avg:.5f}'.format(loss_t=losses_tobii,
                                                                                                 loss_c=losses_cams))




def test():
    #metalist = ['surgery_04']
    metalist = ['surgery_01', 'surgery_02', 'surgery_03', 'surgery_04']
    data_dir = '/media/yukisaito/07f730b8-fe5f-4163-9acb-1079c6fee6a0/'
    imsize = 224
    model, cfg = set_model('sequence')
    for meta_ in metalist:
        print('**********************************************************')
        print('loading.... meta {}'.format(meta_))
        dataloader_test = getTestData(mode='test', batch_size=1, base_dir=data_dir, imsize=imsize, takes=[meta_], num_worker=args.gpus*4, cfg=cfg)
        run_epoch(model, dataloader_test, mode='test', epoch=0, cfg=cfg)
        print('Finished evaluating')

if __name__ == '__main__':
    test()
