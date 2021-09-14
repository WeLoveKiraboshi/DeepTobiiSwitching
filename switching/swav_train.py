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

#SwAv resnet18
from models.SwAV.src.resnet18 import SwAV_Resnet18
from models.SwAV.src.utils import init_distributed_mode

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
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")

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
    model = Encoder()
elif cfg.network == 'MoCo_v1_resnet18':
    # encoder = get_resnet('resnet18', pretrained=cfg.pre_train)
    model = MoCo(vmodels.__dict__['resnet18'], cfg.moco_dim, cfg.moco_k, cfg.moco_m, cfg.moco_t, cfg.moco_mlp)
elif cfg.network == 'BYOL_v1_resnet18':
    online_network = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=cfg.pre_train,
                                                              hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(
        device)
    target_network = models.BYOL.resnet_base_network.ResNet18(network='resnet18', is_pretrained=False,
                                                              hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(
        device)
    # predictor network
    predictor = models.BYOL.mlp_head.MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                                             mlp_hidden_size=cfg.v_hdim, projection_size=cfg.z_dim).to(device)
elif cfg.network == 'CPC_v1_resnet18':
    # Define Autrogressive Network
    enc = PreActResNetN_Encoder(encoder='resnet18', use_classifier=False, cfg=cfg)
    ar = PixelCNN(in_channels=enc.encoding_size)
    model = CPC(enc, ar, cfg.pred_directions, cfg.pred_steps, cfg.neg_samples)
    # model.load_state_dict()
elif cfg.network == 'SwAV_v1_resnet18':
    # init_distributed_mode(args)
    encoder = get_resnet('resnet18', pretrained=cfg.pre_train)
    model = SwAV_Resnet18(encoder=encoder, output_dim=cfg.z_dim, hidden_mlp=cfg.v_hdim,
                          nmb_prototypes=cfg.nmb_prototypes, eval_mode=False, )
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
else:
    print('none model indicated')
    exit(0)

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

if cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
elif cfg.optimizer == 'SGD':
    if cfg.network == 'BYOL_v1_resnet18':
        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),lr=cfg.lr, momentum = cfg.momentum, weight_decay=cfg.weightdecay)
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


@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    #dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()



from torchsummary import summary
# model_ft = vmodels.resnet18(pretrained=True).to(device)
# # # (batch_size, grid_size, grid_size, channels, patch_size, patch_size),(7, 7, 3, 56, 56)
# summary(model, (3, 224, 224))
# print(vmodels.__dict__['resnet18'])

def run_epoch(dataset, mode='train', epoch=0, queue=None):
    global model, optimizer, loss_criteria
    batch_time = AverageMeter()
    losses = AverageMeter()

    N = len(dataset)
    end = time.time()
    cfg.use_the_queue = False

    for i, inputs in enumerate(dataset):
        if cfg.network == 'SwAV_v1_resnet18':
            # from torchvision.utils import save_image
            #
            # for idx1 in range(len(inputs)): #8, bs, 3, 224, 224
            #     for idx2 in range(inputs[0].size(0)):
            #         sample = inputs[idx1][idx2]
            #         save_image(sample.squeeze()[[2, 1, 0]], 'tobii_'+str(idx1)+'_'+str(idx2)+'.png')
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)
            embedding, output = model(inputs)
            embedding = embedding.detach()

            loss = 0
            bs = inputs[0].size(0)
            for j, crop_id in enumerate(cfg.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    # time to use the queue
                    if queue is not None:
                        if cfg.use_the_queue or not torch.all(queue[j, -1, :] == 0):
                            cfg.use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[j],
                                model.prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[j, bs:] = queue[j, :-bs].clone()
                        queue[j, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    # get assignments
                    q = distributed_sinkhorn(out, cfg.epsilon, cfg.sinkhorn_iterations)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(cfg.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / cfg.temperature
                    subloss_comp = torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    if not torch.isnan(subloss_comp):
                        subloss -= subloss_comp
                loss += subloss / (np.sum(cfg.nmb_crops) - 1)

            if not torch.isnan(loss):
                loss /= len(cfg.crops_for_assign)
                losses.update(loss.data.item(), bs)

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
            if cfg.network == 'SwAV_v1_resnet18':
                tb_logger.scalar_summary('CPC_ContrastiveLoss_' + mode, losses.val, niter)

        """clean up gpu memory"""
        torch.cuda.empty_cache()
        if cfg.network == 'deep_infomax':
            pass
        elif cfg.network == 'CPC_v1_resnet18' or cfg.network == 'SwAV_v1_resnet18':
            del inputs



    #tb_logger.LogProgressImage_tobii(model, mode, dataset, epoch)
    if mode == 'train':
        with to_cpu(model):
            if epoch % cfg.save_model_interval == 0:
                model_path = cfg.model_dir + "ckpt_{}_{}.pkl".format(mode, int(epoch + 1))
                model_cp = {'VAEresnet': model.state_dict()}
                pickle.dump(model_cp, open(model_path, 'wb'))
    return (epoch, losses.avg), queue


def train():

    dataloader_train, dataloader_val = getTrainingTestingData(mode=args.mode, batch_size=cfg.bs,
                                                                  is_shuffle=cfg.shuffle, seed=cfg.seed,
                                                                  base_dir=cfg.data_dir, imsize=cfg.imsize,
                                                                  takes=cfg.takes, is_augment=cfg.augment,
                                                                  augment_type=cfg.augment_type,
                                                                  separate_augment=cfg.separate_augment,
                                                                  num_worker=cfg.num_worker,
                                                                  cfg = cfg)
    start_epoch = 14
    for epoch in range(start_epoch, cfg.epochs, 1):
        # build the queue
        queue = None
        queue_path = os.path.join(cfg.model_dir, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        cfg.queue_length -= cfg.queue_length % (cfg.bs * args.world_size)

        # optionally starts a queue
        if cfg.queue_length > 0 and epoch >= cfg.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(cfg.crops_for_assign),
                cfg.queue_length,
                cfg.z_dim,
            ).cuda()
        scores, queue = run_epoch(dataloader_train, mode='train', epoch=epoch, queue=queue)
        scores_val, queue = run_epoch(dataloader_val, mode='val', epoch=epoch,queue=queue)

    print('Finished Training. model saved ')

if __name__ == '__main__':
    train()



