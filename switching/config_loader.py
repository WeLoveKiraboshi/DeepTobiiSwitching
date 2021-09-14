import yaml
import os
from utils.tools import recreate_dirs


class Config:

    def __init__(self, cfg_id, create_dirs=False, meta_id=None):
        self.id = cfg_id
        self.data_dir = '/media/yukisaito/07f730b8-fe5f-4163-9acb-1079c6fee6a0/'
        self.cfg_name = 'config/%s.yml' % cfg_id
        self.meta_name = self.data_dir + 'meta/meta_%s.yml' % meta_id
        if not os.path.exists(self.cfg_name):
            print("Config file doesn't exist: %s" % self.cfg_name)
            exit(0)
        if not os.path.exists(self.meta_name):
            print("Config meta file doesn't exist: %s" % self.meta_name)
            exit(0)
        cfg = yaml.load(open(self.cfg_name, 'r'), Loader=yaml.FullLoader)
        self.meta = yaml.load(open(self.meta_name, 'r'), Loader=yaml.FullLoader)
        #self.ignore_index = yaml.load(open('%s/meta/invalid.yaml' % (self.data_dir)), Loader=yaml.FullLoader)
        self.imsize = cfg.get('imsize', 224)
        self.takes = {x: self.meta[x] for x in ['train', 'test', 'val']}

        # create dirs
        self.base_dir = '/home/yukisaito/DeepTobiiSwitching'
        self.cfg_dir = '%s/results/%s/%s' % (self.base_dir, cfg_id, meta_id)
        self.model_dir = '%s/models/' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.tb_dir)
        self.save_model_interval = cfg.get('save_model_interval', 5)
        self.tb_log_interval = cfg.get('tb_log_interval', 100) # unit iter

        self.seed = cfg['seed']
        self.network = cfg['network']
        self.v_hdim = cfg.get('v_hdim', 1024)
        if self.network == 'MoCo_v1_resnet18':
            self.moco_dim = cfg.get('moco_dim', 64)
            self.moco_k = cfg.get('moco_k', 65536)
            self.moco_m = cfg.get('moco_m', 0.999)
            self.moco_t = cfg.get('moco_t', 0.07)
            self.moco_mlp = cfg.get('moco_mlp', True)
        elif self.network == 'PCL_v1_resnet18':
            self.low_dim = cfg.get('low_dim', 64)
            self.pcl_r = cfg.get('pcl_r', 16384)
            self.moco_m = cfg.get('moco_m', 0.999)
            self.temperature = cfg.get('temperature', 0.2)
            self.mlp = cfg.get('mlp', True)
            self.num_cluster = cfg.get('num_cluster', [25000,50000,100000])
            self.warmup_epoch = cfg.get('warmup_epoch', 10)
            self.schedule = cfg.get('schedule', [120, 160])
        elif self.network == 'PIRL_v1_resnet18':
            self.non_linear_head = cfg.get('non_linear_head', False)
            self.count_negatives = cfg.get('count_negatives', 6400)
            self.temp_parameter = cfg.get('temp_parameter', 0.07)
            self.beta = cfg.get('beta', 0.5)
            self.tmax_for_cos_decay = cfg.get('tmax_for_cos_decay', 70)
        else:
            self.z_dim = cfg['z_dim']
            self.dropout_p = cfg.get('dropout', 0.3)
            if self.network == 'BYOL_v1_resnet18':
                self.BYOL_m = cfg.get('BYOL_m', 0.996) # momentum update
            elif self.network == 'CPC_v1_resnet18':
                self.grid_size = cfg.get('grid_size', 28)
                self.pred_steps = cfg.get('pred_steps', 3)
                self.pred_directions = cfg.get('pred_directions', 4)
                self.norm = cfg.get('norm', 'layer')
                self.num_classes = cfg.get('num_classes', 5)
                self.neg_samples = cfg.get('neg_samples', 16)
            elif self.network == 'SwAV_v1_resnet18':
                self.nmb_prototypes = cfg.get('nmb_prototypes', 3000)
                self.queue_length = cfg.get('queue_length', 0)
                self.epoch_queue_starts = cfg.get('epoch_queue_starts', 15)
                self.use_the_queue = cfg.get('use_the_queue', False)
                self.temperature = cfg.get('temperature', 0.1)
                self.crops_for_assign = cfg.get('crops_for_assign', [1,0])
                self.nmb_crops = cfg.get('nmb_crops', [2, 6])
                self.sinkhorn_iterations = cfg.get('sinkhorn_iterations', 3)
                self.epsilon = cfg.get('epsilon', 0.05)
                self.size_crops = cfg.get('size_crops',[224, 96])
                self.min_scale_crops = cfg.get('min_scale_crops', [0.14, 0.05])
                self.max_scale_crops = cfg.get('max_scale_crops', [1, 0.14])

        self.epochs = cfg['epochs']
        self.shuffle = cfg.get('shuffle', False)
        self.augment = cfg.get('augment', False)
        self.separate_augment = cfg.get('separate_augment', False)
        self.bs = cfg.get('bs', 2)

        self.optimizer = cfg.get('optimizer', 'Adam')
        self.lr = cfg['lr']
        self.momentum = cfg.get('momentum', 0.9)
        self.weightdecay = cfg.get('weight_decay', 0.0)

        self.loss = cfg.get('loss', 'VAEloss')
        self.pre_train = cfg.get('pre_train', True)
        self.metrics = cfg.get('eval_metrics', 'mse')

        self.augment_type = cfg.get('augment_type', ['jitter', 'rotate', 'perspective', 'center_crop', 'gaussian_blur'])

        if self.loss == 'VAELossWithNormalContrasive' or self.loss == 'VAELossWithMarginTripletContrastive':
            if cfg.get('contrastive', None) != None:
                self.contrastive_alpha = cfg['contrastive'].get('alpha', 1.0)
                self.contrastive_beta = cfg['contrastive'].get('beta', 1.0)
                self.contrastive_gamma = cfg['contrastive'].get('gamma', 10.0)
            else:
                self.contrastive_alpha = 1.0
                self.contrastive_beta = 1.0
                self.contrastive_gamma = 10.0
            print('Loading Contrastive Loss params...')
            print('use : {}'.format(self.loss))
            print('param : alpha={} beta={} gamma={}'.format(self.contrastive_alpha, self.contrastive_beta,
                                                             self.contrastive_gamma))
        #generate negative sample for contrastive loss
        if self.loss == 'VAELossWithMarginTripletContrastive':
            self.augment_type_neg = cfg.get('augment_type_neg', ['jitter', 'rotate', 'perspective', 'center_crop', 'gaussian_blur', 'random_mask'])
            self.contrastive_mode = 'pos_neg'
        else:
            self.contrastive_mode = 'pos'
            self.augment_type_neg = None

        #loss for simCLR contrastive loss
        if self.loss == 'NT_Xent' or self.loss == 'NT_Xent_MSE':
            self.temperature = cfg['NT_Xent'].get('temperature', 0.5)

        #loss for DeepInfoMax Loss
        if self.loss == 'DeepInfoMaxLoss':
            if cfg.get('DeepInfoMaxLoss', None) != None:
                self.contrastive_alpha = cfg['DeepInfoMaxLoss'].get('alpha', 0.5)
                self.contrastive_beta = cfg['DeepInfoMaxLoss'].get('beta', 1.0)
                self.contrastive_gamma = cfg['DeepInfoMaxLoss'].get('gamma', 0.1)
            else:
                self.contrastive_alpha = 0.5
                self.contrastive_beta = 1.0
                self.contrastive_gamma = 0.1


        self.checkpoint = cfg.get('checkpoint', None)
        if self.checkpoint == None:
            print("No checkpoint loaded... in config_loader.py")

        #if train from scratch, please turn on
        self.init_train = cfg.get('init_train', True)


        #num_workers if any for especially dataloader
        self.num_worker = cfg.get('num_workers', 4)
