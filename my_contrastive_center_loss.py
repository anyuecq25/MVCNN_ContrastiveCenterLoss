import torch
import torch.nn as nn
import numpy as np
#margin=50000
margin=5000
class MyContrastiveCenterLoss(nn.Module):  # modified from center loss, we add the term from contrastive center loss, but change it to margin-d
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=40, feat_dim=4096, use_gpu=True):
        super(MyContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss1 = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size  #original center loss

        #add contrastive loss
        #batch_size = x.size()[0]  # for example batch_size=6
        expanded_centers = self.centers.expand(batch_size, -1, -1)  # shape 6,3,2
        expanded_hidden = x.expand(self.num_classes, -1, -1).transpose(1, 0)  # shape 6,3,2
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)  # shape 6,3
        #distances_same = distance_centers.gather(1, labels.unsqueeze(1))  # 6,1
        intra_distances = loss1#distances_same.sum()  # means inner ,distance in the same class
        inter_distances = distance_centers.sum().sub(loss1*batch_size)  # distance between different class ,sub =minus
        inter_distances=inter_distances/batch_size/self.num_classes
        epsilon = 1e-6
        #
        loss2=np.max([margin-inter_distances,0])
        loss=loss1+0.1*loss2

        return loss
