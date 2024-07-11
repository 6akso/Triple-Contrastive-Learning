"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class Fourdloss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(Fourdloss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],''at least 3 dimensions are required')
        if len(features.shape) > 3:features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        # # print("logits.shape",logits.shape)
        # print("logits",logits)
        #
        # print('anchor_dot_contrast.shape',anchor_dot_contrast.shape)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
        mask = mask * logits_mask
        # print("mask.shape:",mask.shape)
        # print("mask",mask)
        # mask-out self-contrast cases
        # exp_logits=torch.exp(logits )*mask


        mask2 = torch.ne(labels, labels.T).to(device)
        mask2 = mask2.repeat(anchor_count, contrast_count)


        negtive_dot_negtive_sum = torch.zeros(mask2.size(0))
        negtivesum = torch.zeros(mask2.size(0))

        labels2 = torch.flatten(labels)

        labels2 = torch.cat([labels2, labels2.repeat(anchor_count)])
        for i in range(mask2.size(0)):
            k=0
            if mask2[i].sum() == 0:  # skip if no negative samples for anchor i
                continue
            for j in range(mask2.size(0)):
                if (labels2[i] != labels2[j]):
                    k=k+1
                    mask3 = mask2.clone()
                   
                    negtive1 = anchor_feature[j]
                    negtive2 = contrast_feature[mask3[j]]
                    dot_product = (torch.matmul(negtive2, negtive1.unsqueeze(1)).squeeze() / 0.035).to(device)
                    maxdot, _ = dot_product.max(dim=0)
                    exp_terms = torch.exp(dot_product - maxdot)
                    # compute sum of exponentiation terms
                    negtive_dot_negtive_sum[i] = negtive_dot_negtive_sum[i]+exp_terms.sum()
            negtivesum[i]=k

        negtive_dot_negtive_sum = negtive_dot_negtive_sum.to(device)
        x = negtive_dot_negtive_sum.view(-1, 1).to(device)
        negtivesum=negtivesum.view(-1, 1).to(device)
        # print("x",x)
        # print("这是一次1batchsize的结果", x)
        negtivesum=(torch.repeat_interleave(negtivesum, mask.size(1), dim=1)).to(device)
        x = (torch.repeat_interleave(x, mask.size(1), dim=1)).to(device)
        # print("x.shape",x.shape)
        # print("x",x)
        # compute log_prob
        # exp_logits = torch.exp(logits) *mask
        # print("exp_logits",exp_logits)
        x=x/negtivesum
        log_prob = - (torch.log((x+ 1e-12)))
        # print("log_prob",log_prob)

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print("mean",mean_log_prob_pos)
        #
        # # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print("loss",loss)
        # loss = loss.view(anchor_count, batch_size).mean()
        mask_sum = (mask.sum(dim=1)).to(device)
        mask_sum = (torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)).to(device)  # torch.where()函数的作用是根据条件返回两个张量中的元素，语法如下：torch.where(condition, x, y)其中，condition是一个布尔张量，表示条件；x和y是两个张量，它们的形状和数据类型必须相同。如果condition中的元素为True，则返回x中对应位置的元素；否则返回y中对应位置的元素。
        # compute log-likelihood
        pos_logits = -((mask * log_prob).sum(dim=1) / mask_sum.detach()).to(device)  # （batch,1）每行是相应achor
        loss = pos_logits.view(anchor_count, batch_size).mean()

        # print("loss",loss)
        return loss
