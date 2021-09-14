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
import csv
from switching.eval.eval_metrics import TripletLoss, MSE, KLdiv, NT_Xent, DotSim
from switching.eval.eval_utils import calc_confusion_matrix_my, return_micro_precision_score
from tqdm import tqdm

from utils.utils import AverageMeter

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
parser.add_argument('--seq', default=None)
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

def set_model(ckpt_idx, cfg, meta):
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
        model = MoCo(models.__dict__['resnet18'], cfg.moco_dim, cfg.moco_k, cfg.moco_m, cfg.moco_t, cfg.moco_mlp)
    elif cfg.network == 'BYOL_v1_resnet18':
        model = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=cfg.pre_train, hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)

    ckpt_path = os.path.join('results', cfg.id, meta, 'models', 'ckpt_train_'+str(ckpt_idx)+'.pkl')
    if os.path.exists(ckpt_path):
        model_cp = pickle.load(open(ckpt_path, "rb"))
        model.load_state_dict(model_cp['VAEresnet'], strict=False)
        print('ckpt loaded. path = {}'.format(ckpt_path))
        Flag = True
    else:
        print('none ckpt model found.... {}'.format(ckpt_path))
        Flag = False

    return model.to(device), Flag


def run_epoch(model, dataset, mode='test', epoch=0,cfg=None):
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
        precision_score = return_micro_precision_score(resultlist)
        ckpt_list = {'ckpt_epoch': epoch, 'tobii_recon': losses_tobii.val, 'cams_recon': losses_cams.val, 'micro_precision': precision_score}
    print('mse reconstruction loss : [tobii]={loss_t.val:.5f}    [cams]={loss_c.avg:.5f}    precision={prec:.5f}'.format(loss_t=losses_tobii,
                                                                                                 loss_c=losses_cams, prec=precision_score))
    return ckpt_list




def test():
    seq_ = args.seq #'surgery_01'
    meta_ = 'sequence'
    data_dir = '/media/yukisaito/ssd1/'
    imsize = 224
    start_id = 45
    end_id = 69

    print('loading.... meta {}'.format(meta_))
    ckpt_result_list = []
    cfg = Config(args.cfg, create_dirs=False, meta_id=meta_)
    dataloader_test = getTestData(mode='test', batch_size=1, base_dir=data_dir, imsize=imsize, takes=[seq_],
                                  num_worker=args.gpus * 4, cfg=cfg)
    for ckpt_idx in range(start_id, end_id, 2):
        print('**********************************************************')
        model, flag = set_model(ckpt_idx, cfg, meta_)
        if flag:
            ckpt_list = run_epoch(model, dataloader_test, mode='test', epoch=ckpt_idx, cfg=cfg)
            ckpt_result_list.append(ckpt_list)
        else:
            break
    print('Finished evaluating')



    with open(os.path.join('results', cfg.id, meta_, 'testdata_'+args.seq+'_epoch['+str(start_id)+'-'+str(end_id)+'].csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'tobii_recon', 'cams_recon', 'precision'])
        for row in ckpt_result_list:
            writer.writerow([row['ckpt_epoch'], row['tobii_recon'], row['cams_recon'], row['micro_precision']])



if __name__ == '__main__':
    test()
