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
from tqdm import tqdm

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
import torch.distributed as dist
import faiss

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
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')


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




# from torchsummary import summary
# summary(model, [(3, 224, 224), (3,224,224)])


def compute_features(eval_loader, model, cfg):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), cfg.low_dim).cuda()
    for i, samples in tqdm(enumerate(eval_loader)):
        with torch.no_grad(): #samples["augmented_tobii"]
            images = samples["tobii"].cuda(non_blocking=True)
            feat = model(images, is_eval=True)
            features[samples["index"]] = feat
    # dist.barrier()
    # dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()


def run_kmeans(x, cfg, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(cfg.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg_faiss = faiss.GpuIndexFlatConfig()
        cfg_faiss.useFloat16 = False
        cfg_faiss.device = args.gpuids
        index = faiss.GpuIndexFlatL2(res, d, cfg_faiss)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = cfg.temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / cfg.epochs))
    else:  # stepwise lr schedule
        for milestone in cfg.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run_epoch(dataset, mode='train', epoch=0, cfg=None, cluster_result=None):
    global model, optimizer, loss_criteria
    batch_time = AverageMeter()
    losses = AverageMeter()
    contrastive_losses = AverageMeter()
    acc_inst = AverageMeter() #'Acc@Inst', ':6.2f'
    acc_proto = AverageMeter() #'Acc@Proto', ':6.2f'

    N = len(dataset)
    end = time.time()

    for i, sample_batched in enumerate(dataset):
        image_tobii= torch.autograd.Variable(sample_batched["tobii"].cuda(args.gpuids, non_blocking=True))
        image_augmented_tobii = torch.autograd.Variable(sample_batched["augmented_tobii"].cuda(args.gpuids, non_blocking=True))
        index = sample_batched["index"]
        output, target, output_proto, target_proto = model(im_q=image_tobii, im_k=image_augmented_tobii,
                                                           cluster_result=cluster_result, index=index)
        loss = loss_criteria(output, target)
        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(output_proto, target_proto):
                loss_proto += loss_criteria(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto.update(accp[0], image_tobii.size(0))

            # average loss across all sets of prototypes
            loss_proto /= len(cfg.num_cluster)
            loss += loss_proto

        losses.update(loss.item(), image_tobii.size(0))
        acc = accuracy(output, target)[0]
        acc_inst.update(acc[0], image_tobii.size(0))

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
                  'Acc {acc.val:.4f} ({acc.avg:.4f})'
                  .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta, mode=mode, acc=acc_inst))
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
                tb_logger.scalar_summary('Clustering_AccInst_' + mode, acc_inst.val, niter)
                tb_logger.scalar_summary('Clustering_AccProto_' + mode, acc_proto.val, niter)





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
            elif cfg.network == 'PCL_v1_resnet18':
                del image_tobii, image_augmented_tobii, output, target, output_proto, target_proto

    #tb_logger.LogProgressImage_tobii(model, mode, dataset, epoch)
    if mode == 'train':
        with to_cpu(model):
            if epoch % cfg.save_model_interval == 0:
                model_path = cfg.model_dir + "ckpt_{}_{}.pkl".format(mode, int(epoch + 1))
                model_cp = {'VAEresnet': model.state_dict()}
                pickle.dump(model_cp, open(model_path, 'wb'))







def train():
    dataloader_train, dataloader_val = getTrainingTestingData(mode=args.mode, batch_size=cfg.bs,
                                                              is_shuffle=cfg.shuffle, seed=cfg.seed,
                                                              base_dir=cfg.data_dir, imsize=cfg.imsize,
                                                              takes=cfg.takes, is_augment=cfg.augment,
                                                              augment_type=cfg.augment_type,
                                                              separate_augment=cfg.separate_augment,
                                                              num_worker=cfg.num_worker,
                                                              cfg=cfg)
    for epoch in range(cfg.epochs):
        cluster_result = None
        if epoch >= cfg.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(dataloader_train, model, cfg)

            # placeholder for clustering result
            cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in cfg.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(dataloader_train), dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), cfg.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

            if args.gpuids == 0:
                features[
                    torch.norm(features, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
                features = features.numpy()
                cluster_result = run_kmeans(features, cfg, args)  # run kmeans clustering on master node
                # save the clustering result
                # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))

            #dist.barrier()
            # broadcast clustering result
            # for k, data_list in cluster_result.items():
            #     for data_tensor in data_list:
            #         dist.broadcast(data_tensor, 0, async_op=False)
        adjust_learning_rate(optimizer, epoch, args, cfg)

        run_epoch(dataloader_train, mode='train', epoch=epoch,cfg=cfg, cluster_result=cluster_result)
        #run_epoch(dataloader_val, mode='val', epoch=epoch)

    print('Finished Training. model saved ')

if __name__ == '__main__':
    train()



