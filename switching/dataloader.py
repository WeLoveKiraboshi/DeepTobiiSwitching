from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np
import os

#CPC training
from models.CPC.data.image_preprocessing import PatchifyAugment


from itertools import permutations
import csv
from sklearn.utils import shuffle
from skimage.transform import resize


import sys
sys.path.append(os.getcwd())
import switching.augment as augment
import cv2
from PIL import Image, ImageDraw, ImageFilter
_check_pil = lambda x: isinstance(x, Image.Image)
_check_np_img = lambda x: isinstance(x, np.ndarray)

#PIRL jigssaw
from models.pirl.dataset_helpers import get_nine_crops, pirl_full_img_transform, pirl_stl10_jigsaw_patch_transform




data = {'tobii': {}, 'cam': {}, 'px': {}, 'py': {}, 'seq': {}, 'gt_label': {}}
count = 0

Image.MAX_IMAGE_PIXELS = 2000000000
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)


def make_cropped_img_mask(img, x_pos, y_pos):
    if x_pos == 0 and y_pos == 0:
        print('Error : gaze position (px, py) is zero.')
        exit(0)
    elif x_pos < 112:
        img = img.crop((0, 0, 224, 224))#left, upper, right, lower
    elif 112 <= x_pos and x_pos <= 304:
        img = img.crop((x_pos - 112, 0, x_pos + 112, 224))
    elif 304 < x_pos:
        img = img.crop((192, 0, 416, 224))
    return img

def mask_circle_solid(pil_img, background_color, blur_radius, offset=0):
    background = Image.new(pil_img.mode, pil_img.size, background_color)

    offset = 2 #blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    return Image.composite(pil_img, background, mask)


class myDatasetMemory(Dataset):
    def __init__(self, data, my_train, imsize, is_augment=False, augment_type=None, separate_augment=False, mode='train', cfg=None):
        self.data, self.my_dataset = data, my_train
        self.imsize = imsize # 224
        self.mode = mode
        self.is_augment = is_augment
        self.augment_type = augment_type
        self.separate = separate_augment
        self.cfg = cfg

        if cfg.network == 'SwAV_v1_resnet18':
            assert len(cfg.size_crops) == len(cfg.nmb_crops)
            assert len(cfg.min_scale_crops) == len(cfg.nmb_crops)
            assert len(cfg.max_scale_crops) == len(cfg.nmb_crops)
            self.trans = getSwAVTransform(cfg=cfg)
            self.return_index = False

    def __getitem__(self, idx):
        sample = self.my_dataset[idx]
        rgb_tobii_path = self.data['tobii'][sample]
        seq = self.data['seq'][sample]

        rgb_tobii = Image.open(rgb_tobii_path).resize((416, 224))
        rgb_tobii = make_cropped_img_mask(rgb_tobii, data['px'][sample], data['py'][sample])

        if self.cfg.network == 'CPC_v1_resnet18' and self.mode == 'train':
            #rgb_tobii_np = np.asarray(rgb_tobii).transpose((2,0,1))
            sample = {'tobii': getCPCTransform(seq, self.cfg.grid_size)(rgb_tobii)}
            return sample
        elif self.cfg.network == 'SwAV_v1_resnet18' and self.mode == 'train':
            multi_crops = list(map(lambda trans: trans(rgb_tobii), self.trans))
            if self.return_index:
                return index, multi_crops
            return multi_crops
        elif self.cfg.network == 'PIRL_v1_resnet18' and self.mode == 'train':
            # Get nine crops for the image
            image_tensor = getNoTransform()(rgb_tobii)
            nine_crops = get_nine_crops(rgb_tobii)
            # Form the jigsaw order for this image
            original_order = np.arange(9)
            permuted_order = np.copy(original_order)
            np.random.shuffle(permuted_order)
            # Permut the 9 patches obtained from the image
            permuted_patches_arr = [None] * 9
            for patch_pos, patch in zip(permuted_order, nine_crops):
                permuted_patches_arr[patch_pos] = patch
            # Apply data transforms
            # TODO: Remove hard coded values from here
            tensor_patches = torch.zeros(9, 3, 64, 64)
            for ind, patch in enumerate(permuted_patches_arr):
                patch_tensor = getPIRLTransform(seq=seq, grid_size=64)(patch) #pirl_stl10_jigsaw_patch_transform(patch)
                tensor_patches[ind] = patch_tensor
            return [image_tensor, tensor_patches], idx


        rgb_tobii = mask_circle_solid(rgb_tobii, (0, 0, 0), 2)

        if self.mode == 'test':
            cam_path = self.data['cam'][sample]
            cam1 = Image.open(cam_path[:-12] + '1' + cam_path[-11:]).resize((224, 224))
            cam2 = Image.open(cam_path[:-12] + '2' + cam_path[-11:]).resize((224, 224))
            cam3 = Image.open(cam_path[:-12] + '3' + cam_path[-11:]).resize((224, 224))
            cam4 = Image.open(cam_path[:-12] + '4' + cam_path[-11:]).resize((224, 224))
            cam5 = Image.open(cam_path[:-12] + '5' + cam_path[-11:]).resize((224, 224))
            labels = np.array(self.data['gt_label'][sample]) #.reshape(1, 1)
            sample = {'tobii': getNoTransform()(rgb_tobii), 'cam1':getNoTransform()(cam1), 'cam2':getNoTransform()(cam2),
                      'cam3':getNoTransform()(cam3), 'cam4':getNoTransform()(cam4),'cam5':getNoTransform()(cam5),
                      'gt_label': torch.from_numpy(labels).clone()}
            return sample
        if self.mode == 'train':
            if self.is_augment:
                if self.separate:
                    sample = {'tobii': getAugmentTransform(self.augment_type, seq, True)(rgb_tobii),
                              'augmented_tobii': getAugmentTransform(self.augment_type, seq, True)(rgb_tobii),
                              'index': idx}
                else:
                    sample = {'tobii': getNoTransform()(rgb_tobii),
                              'augmented_tobii': getAugmentTransform(self.augment_type, seq)(rgb_tobii)}
                return sample
            else:
                sample = {'tobii': getNoTransform()(rgb_tobii)}
                return sample

    def __len__(self):
        return len(self.my_dataset)








class ToTensor(object):
    def __init__(self):
       pass

    def __call__(self, sample):
        tobii_ = cv2.cvtColor(np.asarray(sample), cv2.COLOR_RGB2BGR)
        return torch.clamp(self.to_tensor(tobii_), 0, 1)

    def to_tensor(self, pic):
        img = torch.from_numpy(np.transpose(pic, (2, 0, 1)).copy())
        return img.float().div(255)


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor()
    ])

colorjitter = [augment.ColorJitter(brightness=(0.7, 1.25), contrast=(1.2, 1.45),saturation=(1.0, 1.3)), #for surgery
               augment.ColorJitter(brightness=(0.6, 1.15), contrast=(0.6, 1.5), saturation=(1.0, 1.0)),  #for surgery2
               augment.ColorJitter(brightness=(0.6, 1.15), contrast=(0.6, 1.5), saturation=(1.0, 1.0)),  #for surgery3
               augment.ColorJitter(brightness=(0.8, 1.1), contrast=(0.7, 1.25), saturation=(1.0, 1.0))]  #for surgery4

def getAugmentTransform(augment_type, seq, separate=False):
    if separate:
        return transforms.Compose([
           augment.RandomChoice([colorjitter[seq],
                                       augment.RandomRotation(180),
                                       augment.RandomPerspective(distortion_scale=0.5, p=0.5,
                                                                 interpolation=Image.BICUBIC),
                                       augment.CenterCrop(180, True),
                                       augment.RandomGaussBlur(radius=[0.5, 1.5])]),
            ToTensor()
        ])
    else:
        compound_augmentation = []
        for elem in augment_type:
            if elem == 'jitter':
                compound_augmentation.append(colorjitter[seq])
            elif elem == 'rotate':
                compound_augmentation.append(augment.RandomRotation(180))
            elif elem == 'perspective':
                compound_augmentation.append(
                    augment.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC))
            elif elem == 'center_crop':
                try:
                    compound_augmentation.append(augment.RandomChoice([
                        augment.CenterCrop(200),
                        augment.CenterCrop(180)]))
                    compound_augmentation.append(augment.Resize((224, 224)))
                except:
                    print('Exceed Max PIL Image Pixel size: may be DOS attack...')
            elif elem == 'gaussian_blur':
                compound_augmentation.append(augment.RandomGaussBlur(radius=[0.5, 1.5]))
        compound_augmentation.append(augment.mask_circle_solid((0, 0, 0), 1, 0))
        compound_augmentation.append(ToTensor())
        return transforms.Compose(compound_augmentation)

def getSwAVTransform(cfg=None):
    size_dataset = -1
    if size_dataset >= 0:
        samples = self.samples[:size_dataset]
    trans = []
    for i in range(len(cfg.size_crops)):
        randomresizedcrop = transforms.RandomResizedCrop(
            cfg.size_crops[i],
            scale=(cfg.min_scale_crops[i], cfg.max_scale_crops[i]),
        )
        trans.extend([transforms.Compose([
            randomresizedcrop,
            augment.RandomChoice([augment.RandomChoice(colorjitter),
                                augment.RandomRotation(180),
                                augment.RandomPerspective(distortion_scale=0.5, p=0.5,interpolation=Image.BICUBIC),
                                augment.RandomGaussBlur(radius=[0.5, 1.5])]),
            ToTensor(),])] * cfg.nmb_crops[i])
    return trans


def getCPCTransform(seq, grid_size=7):
    compound_augmentation = []
    if np.random.rand() > 0.5:
        compound_augmentation.append(augment.RandomChoice([colorjitter[seq],
                                                           augment.RandomRotation(180),
                                                           augment.RandomPerspective(distortion_scale=0.5, p=0.5,
                                                                                     interpolation=Image.BICUBIC),
                                                           augment.CenterCrop(180, True),
                                                           augment.RandomGaussBlur(radius=[0.5, 1.5])]))
    compound_augmentation.append(ToTensor())
    compound_augmentation.append(PatchifyAugment(gray=False, grid_size=grid_size))
    return transforms.Compose(compound_augmentation)

def getPIRLTransform(seq=0, grid_size=64):
    compound_augmentation = []
    if np.random.rand() > 0.5:
         compound_augmentation.append(augment.RandomChoice([colorjitter[seq],
                                                            #augment.RandomRotation(180),
                                                            #augment.RandomPerspective(distortion_scale=0.5, p=0.5,interpolation=Image.BICUBIC),
                                                            #augment.CenterCrop(180, True),
                                                            augment.RandomGaussBlur(radius=[0.5, 1.5])]))
    compound_augmentation.append(transforms.RandomCrop(grid_size, padding=1))
    compound_augmentation.append(ToTensor())
    return transforms.Compose(compound_augmentation)




def read_csv_file(base_dir, mode, csv_path, take_seq):
    global data, count
    first_row_flag = False
    if mode=="test":
        csvpath = base_dir + 'labels/' + mode+ '/myannotation_' + csv_path
        data = {'tobii': {}, 'cam': {}, 'px': {}, 'py': {}, 'seq': {}, 'gt_label': {}}
    else:
        csvpath = base_dir + 'labels/' + mode+ '/' + csv_path
    if not os.path.exists(csvpath):
        print('Error : file does not exists ', csvpath)
    with open(csvpath) as f:
        reader = csv.reader(f)
        for row in reader:
            if first_row_flag == False and mode=="train":
                first_row_flag = True
                continue
            tobii_path = base_dir + 'frames/%s/'%take_seq + row[0]
            cam_path = base_dir + 'frames/%s/'%take_seq + 'cam1/' + '%06d.png' % (int(row[3]))
            if not os.path.exists(cam_path) or not os.path.exists(tobii_path):
                print('Error : file does nott exists ',cam_path)
                continue
            data['tobii'].update({count: tobii_path})
            data['cam'].update({count: cam_path})
            data['px'].update({count: int(float(row[1]) * 416)})
            data['py'].update({count: int(float(row[2]) * 224)})
            data['seq'].update({count: int(take_seq[-1:])-1}) #note that 'seq' is in range 0~3
            if mode == 'train':
                data['gt_label'].update({count: int(row[4])})
            if mode == 'test':
                data['gt_label'].update({count: [int(row[4]), int(row[5]),int(row[6]),int(row[7]),int(row[8])]})
            count += 1
    print('load... ' + csv_path + ' done.  data_length = {}'.format(len(data['tobii'])))

def get_my_data(mode, base_dir, takes=None, split_ratio=0.8, SEED=1, is_shuffle=True):
    global homography_list, occlusion_list
    if mode=='test':
        for take_seq in takes:
            read_csv_file(base_dir, mode, take_seq+'.csv', take_seq)
    else:
        for take_seq in takes[mode]:
            read_csv_file(base_dir, mode, take_seq+'.csv', take_seq)


    if mode == "train":
        keys = list(data['tobii'].keys())
        train = keys[:int(len(keys)*split_ratio)]
        val = keys[int(len(keys)*split_ratio):]
        if is_shuffle:
            train = shuffle(train, random_state=SEED)
            val = shuffle(val, random_state=SEED)
        return data, train, val
    else:
        return data, list(data['tobii'].keys())



def getTrainingTestingData(mode, batch_size, is_shuffle, seed, base_dir, imsize, takes, is_augment, augment_type, separate_augment, num_worker=4, cfg=None):
    data, train, val = get_my_data(mode=mode, base_dir=base_dir, takes=takes, split_ratio=0.8, SEED=seed, is_shuffle=is_shuffle)
    transformed_training = myDatasetMemory(data, train, imsize, is_augment,augment_type, separate_augment=separate_augment, mode='train', cfg=cfg)
    transformed_validation = myDatasetMemory(data, val, imsize, is_augment, augment_type, separate_augment=separate_augment, mode='train', cfg=cfg)

    return DataLoader(transformed_training, batch_size, shuffle=True,drop_last =True, num_workers=num_worker), \
           DataLoader(transformed_validation, batch_size,shuffle=True,drop_last =True, num_workers=num_worker), train, val


def getTestData(mode, batch_size, base_dir='/mnt/hdd1/saito/proj-kajita/', imsize=224, takes=None, num_worker=4, cfg=None):
    data, test = get_my_data(mode, base_dir, takes=takes, split_ratio=0.8, SEED=0, is_shuffle=False)
    transformed_testing = myDatasetMemory(data, test, imsize, False, None,  separate_augment=False, mode='test', cfg=cfg)

    return DataLoader(transformed_testing, batch_size, shuffle=False, num_workers=num_worker)
