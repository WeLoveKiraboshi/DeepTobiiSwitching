"""
File: logger.py
Modified by: Senthil Purushwalkam
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
Email: spurushw<at>andrew<dot>cmu<dot>edu
Github: https://github.com/senthilps8
Description:
"""

#import tensorflow as tf
from torch.autograd import Variable
import numpy as np
import scipy.misc
import os
import torch
from os import path

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils.utils import colorize

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir, name=None):
        """Create a summary writer logging to log_dir."""
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            print('Tensorboard Log is logged : ', os.path.join(log_dir, name))
            self.writer = SummaryWriter(logdir="{}".format(os.path.join(log_dir, name)))
        else:
            print('Tensorboard Log is logged : ', log_dir)
            self.writer = SummaryWriter(logdir="{}".format(log_dir))

    def scalar_summary(self, tags, values, step):
        """Log a scalar variable.
        """
        self.writer.add_scalar(tag=tags, scalar_value=values, global_step=step)
        self.writer.flush()

    def LogProgressImage_tobii(self, model, tags, data_loader, epoch):
        with torch.no_grad():
            for i, sample_batched in enumerate(data_loader):
                image = torch.autograd.Variable(sample_batched["augmented_tobii"].cuda())
                output, mu, logvar, z = model(image)
                # PyTorch tensor NCHW  ->  rgb
                permute = [2, 1, 0]
                image = image[:, permute]
                output = output[:, permute]
                self.writer.add_image(str(tags) + '.1.Image', vutils.make_grid(image.data, nrow=6, normalize=False),
                                      epoch)
                self.writer.add_image(str(tags) + '.2.Reconstructed',
                                      vutils.make_grid(output.data, nrow=6, normalize=False), epoch)
                self.writer.add_image(str(tags) + '.3.Diff', colorize(
                    vutils.make_grid(torch.abs(output - image).data, nrow=6, normalize=True)), epoch)
                del image
                del output
                break

