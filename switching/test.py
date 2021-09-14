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

from utils.utils import AverageMeter

from switching.eval.eval_metrics import TripletLoss, MSE, KLdiv, NT_Xent, DotSim
from switching.eval.eval_utils import calc_confusion_matrix_my

from tqdm import tqdm
from torchvision.utils import save_image


models_func = {
'resnet18': ResNet18_VAE,
'resnet50': ResNet50_VAE,
}

# Argument Parser
parser = argparse.ArgumentParser(description='Project-Kajita Multi camera switching via tobbi sensor using VAE')
parser.add_argument('--cfg', default=None)
parser.add_argument('--meta', default=None)
parser.add_argument('--mode', default='test')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=int, default=0, help='IDs of GPUs to use')
parser.add_argument('--imsave', type=bool, default=False, help='if save images(inputs, recon)')
args = parser.parse_args()
cfg = Config(args.cfg, create_dirs=False, meta_id=args.meta)

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

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


model = models_func[cfg.network](cfg.z_dim, cfg.v_hdim, cfg.dropout_p, cfg.pre_train)
if cfg.checkpoint != None:
    assert os.path.isfile(cfg.checkpoint), \
        "=> no model found at '{}'".format(cfg.checkpoint)
    print("=> loading model '{}'".format(cfg.checkpoint))
    if '.pth' in cfg.checkpoint:
        try:
            checkpoint = torch.load(cfg.checkpoint)
            if type(checkpoint) is dict:
                model_state = checkpoint['model']
                print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
            else:
                model_state = checkpoint
            model.load_state_dict(model_state)  # if saved_model stat
        except:
            model = torch.load(cfg.checkpoint)
    elif '.pkl' in cfg.checkpoint:
        model_cp = pickle.load(open(cfg.checkpoint, "rb"))
        model.load_state_dict(model_cp['VAEresnet'], strict=False)
    else:
        print('couln not distinguish file type for checkpoin: {}'.format(cfg.checkpoint))

else:
    print('no trained model indicated...')
    exit(0)
if args.gpus == 2:
    model = torch.nn.DataParallel(model)  # make parallel
elif args.gpus == 1:
    model.to(device)

model.eval()
# from torchsummary import summary
# summary(model, input_size=(3, cfg.imsize, cfg.imsize))
# print('Model created')

lower_better = True
if cfg.metrics == 'cos':
    metrics_criteria = nn.CosineSimilarity(dim=1, eps=1e-6)
    lower_better = False
elif cfg.metrics == 'mse':
    metrics_criteria = MSE()
    lower_better = True
elif cfg.metrics == 'contrasive':
    metrics_criteria = NT_Xent(temperature=0.5)
    lower_better = True
elif cfg.metrics == 'dot':
    metrics_criteria = DotSim(temperature = 0.5)
    lower_better = False

from concurrent import futures
from multiprocessing import Process
def run_epoch(dataset, mode='test', epoch=0):
    global model
    N = len(dataset)
    resultlist = []
    mse = nn.MSELoss(reduce=False)
    losses_tobii = AverageMeter()
    losses_cams = AverageMeter()

    with torch.no_grad():
        for i, sample_batched in tqdm(enumerate(dataset)):
            image_tobii = sample_batched["tobii"].cuda()  # torch.autograd.Variable(sample_batched["tobii"].cuda())
            image_cams = torch.cat(
                [sample_batched['cam1'], sample_batched['cam2'], sample_batched['cam3'], sample_batched['cam4'],
                 sample_batched['cam5']], 0).cuda()
            if cfg.network == 'resnet18' or cfg.network == 'resnet50':
                output_t, mu_t, logvar_t, z_t = model.forward(image_tobii)
                output_c, mu_c, logvar_c, z_c = model.forward(image_cams)
                z_t = z_t.squeeze(0)

                recon_loss_tobii = torch.sum(torch.sum(torch.sum(mse(output_t, image_tobii), axis=-1), axis=-1), axis=-1)/(3*224*224)
                recon_loss_cams = torch.mean(torch.sum(torch.sum(torch.sum(mse(output_c, image_cams), axis=-1), axis=-1), axis=-1)/(3*224*224))
                losses_tobii.update(recon_loss_tobii.data.item(), image_tobii.size(0))
                losses_cams.update(recon_loss_cams.data.item(), image_tobii.size(0))
            elif cfg.network == 'resnet18_simCLR' or cfg.network == 'resnet50_simCLR':
                h_t, h_c, z_t, z_c = model.forward(image_tobii, image_cams.cuda())
                z_t = z_t.squeeze(0)
            elif cfg.network == 'MoCo_v1_resnet18':
                z_t = model.test_foward(im_q=image_tobii)
                z_c = model.test_foward(im_q=image_cams)

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

            result_comp = [pred_idx, int(sample_batched["gt_label"][0][0]), int(sample_batched["gt_label"][0][1]), int(sample_batched["gt_label"][0][2]),
                 int(sample_batched["gt_label"][0][3]), int(sample_batched["gt_label"][0][4])]
            resultlist.append(result_comp)
            if args.imsave and i % 130 == 0:
                dir_path = os.path.join(cfg.result_dir, 'iter_' + str(i))
                os.makedirs(dir_path, exist_ok=True)
                save_image(sample_batched["tobii"].squeeze()[[2,1,0]], os.path.join(dir_path, 'tobii.png'))
                save_image(output_t.squeeze()[[2, 1, 0]], os.path.join(dir_path, 'recon_tobii.png'))
                for cam_idx in range(5):
                    save_image(sample_batched["cam"+str(cam_idx+1)].squeeze()[[2, 1, 0]], os.path.join(dir_path, 'cam'+str(cam_idx+1)+'.png'))
                    save_image(output_c[cam_idx][[2, 1, 0]], os.path.join(dir_path, 'recon_cam'+str(cam_idx+1)+'.png'))
                np.savetxt(os.path.join(dir_path, 'labels.txt'), np.array(result_comp))



            """clean up gpu memory"""
            torch.cuda.empty_cache()
            del image_tobii
        calc_confusion_matrix_my(resultlist)
    print('mse reconstruction loss : [tobii]={loss_t.val:.5f}    [cams]={loss_c.avg:.5f}'.format(loss_t=losses_tobii, loss_c=losses_cams))




def test():
    cfg.bs = 1
    dataloader_test = getTestData(mode='test', batch_size=cfg.bs, base_dir=cfg.data_dir, imsize=cfg.imsize, takes=cfg.takes['test'], num_worker=cfg.num_worker, cfg=cfg)
    run_epoch(dataloader_test, mode='test', epoch=0)
    print('Finished evaluating')

if __name__ == '__main__':
    test()
