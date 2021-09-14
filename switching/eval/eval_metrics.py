import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def calculate_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.sum(F.mse_loss(anchor, positive, reduce=False, reduction='none'), axis=-1) #self.calculate_euclidean(anchor, positive)
        distance_negative = torch.sum(F.mse_loss(anchor, negative, reduce=False, reduction='none'), axis=-1) #self.calculate_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        #print('d_pos = {}, n_pos = {}, losses = {}'.format(distance_positive, distance_negative, losses))

        return losses.mean() if size_average else losses.sum()



class NT_Xent(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.bs = 1

    def forward(self,out_1, out_2):
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.bs, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.bs, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss  # lower is better



class DotSim(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.bs = 1

    def forward(self, out_1, out_2):
        return torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)


class DotSim(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.bs = 1

    def forward(self, out_1, out_2):
        return torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)



class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_1, out_2):
        mse = nn.MSELoss(reduce=False)
        return torch.sum(mse(out_1, out_2), axis=-1)


class KLdiv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_mean_1, z_log_var_1, z_mean_2, z_log_var_2):
        kl_loss = (z_log_var_2 - z_log_var_1) + (torch.exp(z_log_var_1) / torch.exp(z_log_var_2)) + (
                    (z_mean_1 - z_mean_2).pow(2) / torch.exp(z_log_var_2)) - 1
        kl_loss = torch.sum(kl_loss, axis=-1)[0]
        kl_loss *= 0.5
        return kl_loss


