import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer
import numpy as np


class NT_Xent_MSE(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent_MSE, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.MSELoss() #nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def mse_axis(self, a, b):
        sim = torch.torch.zeros([a.shape[0], b.shape[0]])
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                sim[i, j] = self.similarity_f(a[i, :], b[j, :])
        return sim

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        #loss = nn.MSELoss(reduction='none')
        #a = torch.tensor(np.array([[1., 2., 7.], [3., 4., 3.]]))
        #b = torch.tensor(np.array([[5., 6., 9.], [7., 8., 2.]]))
        mse_mat = self.mse_axis(z, z)
        #torch.sum(torch.square(predicted_x - target), axis=1) / (predicted_x.size()[1])
        sim = mse_mat / self.temperature #self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
