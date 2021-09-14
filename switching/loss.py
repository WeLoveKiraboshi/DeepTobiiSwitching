import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DeepInfoMax import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator

class VAEloss(nn.Module):
    def __init__(self,):
        nn.Module.__init__(self)


    def forward(self, recon_x, x, mu, logvar):
        # x = x.reshape(x.shape[0], -1)
        # recon_x = x.reshape(recon_x.shape[0], -1)
        kld = -0.5 * torch.sum(1 + logvar - mu * mu - torch.exp(logvar), axis=-1)
        recon = torch.sum(torch.sum(torch.sum(
        F.binary_cross_entropy(recon_x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), reduce=False, reduction='none'),axis=-1), axis=-1), axis=-1)
        loss = torch.mean(recon + kld)
        # print('reconstruction loss = ', recon)
        # print('KL div = ', kld)
        return loss


class VAEloss_NormalContrasive(nn.Module):
    def __init__(self,alpha=1, beta=1,gamma=10):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def KL_divergence(self, z_mean_1, z_log_var_1, z_mean_2, z_log_var_2):
        kl_loss = (z_log_var_2 - z_log_var_1) + (np.exp(z_log_var_1) / np.exp(z_log_var_2)) + (
                np.square(z_mean_1 - z_mean_2) / np.exp(z_log_var_2)) - 1
        kl_loss = np.sum(kl_loss, axis=-1)
        kl_loss *= 0.5
        return kl_loss

    def JS_divergence(self, z_mean_1, z_log_var_1, z_mean_2, z_log_var_2):
        M_mean = z_mean_1 + z_mean_2
        M_log_var = z_log_var_1 + z_log_var_2
        return (KL_divergence(z_mean_1, z_log_var_1, M_mean, M_log_var) + KL_divergence(z_mean_2, z_log_var_2, M_mean,
                                                                                        M_log_var)) / 2

    def forward(self, elem1, elem2):
        kld1 = -0.5 * torch.sum(1 + elem1['logvar'] - elem1['mu'] * elem1['mu'] - torch.exp(elem1['logvar']), axis=-1)
        recon1 = torch.sum(torch.sum(torch.sum(F.binary_cross_entropy(elem1['recon_x'].permute(0, 2, 3, 1), elem1['x'].permute(0, 2, 3, 1),
                                                                      reduce=False, reduction='none'),axis=-1), axis=-1), axis=-1)
        kld2 = -0.5 * torch.sum(1 + elem2['logvar'] - elem2['mu'] * elem2['mu'] - torch.exp(elem2['logvar']), axis=-1)
        recon2 = torch.sum(torch.sum(torch.sum(F.binary_cross_entropy(elem2['recon_x'].permute(0, 2, 3, 1), elem2['x'].permute(0, 2, 3, 1),
                                             reduce=False, reduction='none'), axis=-1), axis=-1), axis=-1)
        contrasive = torch.sum(torch.pow(elem1['mu'] - elem2['mu'], 2) / 2, axis=-1)
        #KL_divergence(elem1['mu'], elem1['logvar'], elem2['mu'], elem2['logvar'])
        loss = torch.mean((recon1 + recon2)*self.alpha + (kld1+kld2)*self.beta + contrasive*self.gamma)
        return loss











class VAEloss_MarginTripletContrasive(nn.Module):
    def __init__(self,alpha=1, beta=1,gamma=10, m=10.0):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = m

    def KL_divergence(self, z_mean_1, z_log_var_1, z_mean_2, z_log_var_2):
        kl_loss = (z_log_var_2 - z_log_var_1) + (np.exp(z_log_var_1) / np.exp(z_log_var_2)) + (
                np.square(z_mean_1 - z_mean_2) / np.exp(z_log_var_2)) - 1
        kl_loss = np.sum(kl_loss, axis=-1)
        kl_loss *= 0.5
        return kl_loss

    def JS_divergence(self, z_mean_1, z_log_var_1, z_mean_2, z_log_var_2):
        M_mean = z_mean_1 + z_mean_2
        M_log_var = z_log_var_1 + z_log_var_2
        return (KL_divergence(z_mean_1, z_log_var_1, M_mean, M_log_var) + KL_divergence(z_mean_2, z_log_var_2, M_mean,
                                                                                        M_log_var)) / 2

    def forward(self, elem1, elem2, elem3):
        kld1 = -0.5 * torch.sum(1 + elem1['logvar'] - elem1['mu'] * elem1['mu'] - torch.exp(elem1['logvar']), axis=-1)
        recon1 = torch.sum(torch.sum(torch.sum(F.binary_cross_entropy(elem1['recon_x'].permute(0, 2, 3, 1), elem1['x'].permute(0, 2, 3, 1),
                                                                      reduce=False, reduction='none'),axis=-1), axis=-1), axis=-1)
        kld2 = -0.5 * torch.sum(1 + elem2['logvar'] - elem2['mu'] * elem2['mu'] - torch.exp(elem2['logvar']), axis=-1)
        recon2 = torch.sum(torch.sum(torch.sum(F.binary_cross_entropy(elem2['recon_x'].permute(0, 2, 3, 1), elem2['x'].permute(0, 2, 3, 1),
                                             reduce=False, reduction='none'), axis=-1), axis=-1), axis=-1)
        kld3 = -0.5 * torch.sum(1 + elem3['logvar'] - elem3['mu'] * elem3['mu'] - torch.exp(elem3['logvar']), axis=-1)
        recon3 = torch.sum(torch.sum(
            torch.sum(F.binary_cross_entropy(elem3['recon_x'].permute(0, 2, 3, 1), elem3['x'].permute(0, 2, 3, 1),
                                             reduce=False, reduction='none'), axis=-1), axis=-1), axis=-1)
        pos_dis = torch.norm(elem1['mu'] - elem2['mu'], dim=-1)
        neg_dis = torch.norm(elem1['mu'] - elem3['mu'], dim=-1)
        margin_contrasive = torch.clamp(pos_dis + self.margin - neg_dis, min=0.0)
        #print(margin_contrasive)
        #print('recon = {}, KL={}, contrastive={}'.format(torch.mean(recon1 + recon2), torch.mean(kld1+kld2), torch.mean(margin_contrasive)))
        loss = torch.mean((recon1 + recon2 + recon3)*self.alpha + (kld1+kld2+kld3)*self.beta + margin_contrasive*self.gamma)
        return loss, torch.mean(margin_contrasive)





class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR

class RegressionLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)






class SwAVContrastiveLoss(nn.Module):
    def __init__(self, crops_for_assign=[1,0], nmb_crops=0, temperature=0.5, use_the_queue=False, epsilon=0.05, sinkhorn_iterations=3):
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        self.temperature = temperature
        self.use_the_queue = use_the_queue
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations


    def set_queue(self, use_the_queue=False):
        self.use_the_queue = use_the_queue

    def foward(self, bs=64, output=None, model=None, embedding=None, queue=None):
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if self.use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out, self.epsilon, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
        return loss, queue

@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
