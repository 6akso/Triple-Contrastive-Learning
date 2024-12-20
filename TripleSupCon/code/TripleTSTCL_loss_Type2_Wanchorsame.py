import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)


class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super(SupConLoss, self).__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1)).float()
            mask = mask * (1 - torch.eye(mask.size(0))).to(device)

        anchor_dot_target = torch.matmul(anchor, target.t()) / self.temp
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum
        loss = -pos_logits.mean()
        return loss

    def nt_xent_loss2(self, anchor, target, labels):
        with torch.no_grad():
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1)).float()
            mask2 = torch.ne(labels, labels.transpose(0, 1)).float()
            mask3 = torch.eq(labels, labels.transpose(0, 1)).float()
            negtive_dot_negtive_sum = torch.zeros(mask2.size(0)).to(device)
            negtivesum = torch.zeros(mask2.size(0)).to(device)
    
        for i in range(mask2.size(0)):
            if mask2[i].sum() == 0:
                continue
            negtive_mask = mask2[i]
            same_label_indices = torch.where(mask3[i])[0]
            negtive_indices = torch.cat((same_label_indices, torch.where(negtive_mask)[0])).unique()
            if negtive_indices.size(0) == 0:
                continue
            negtive_features = anchor[negtive_indices]
            if negtive_features.size(0) == 0:
                continue
            different_label_mask = (
                    labels[negtive_indices].unsqueeze(1) != labels[negtive_indices].unsqueeze(0)).float()
            different_label_mask = different_label_mask.squeeze(-1)
    
            same_label_mask = torch.zeros_like(different_label_mask)
            for idx in same_label_indices:
                same_label_mask[idx, :] = 1
    
            dot_product_matrix = torch.matmul(negtive_features, negtive_features.T) / 0.05
            dot_product_matrix = dot_product_matrix * (1 - same_label_mask) * different_label_mask
            maxdot = dot_product_matrix.max(dim=1, keepdim=True)[0]
            exp_terms = torch.exp(dot_product_matrix - maxdot)
    
            negtive_dot_negtive_sum[i] = exp_terms.sum()
            negtivesum[i] = ((1 - same_label_mask) * different_label_mask).sum()
    
        x = negtive_dot_negtive_sum.view(-1, 1)
        negtivesum = negtivesum.view(-1, 1)
        negtivesum = torch.repeat_interleave(negtivesum, mask.size(1), dim=1)
        x = torch.repeat_interleave(x, mask.size(1), dim=1)
    
        if not torch.all(negtivesum == 0):
            x = x / negtivesum
        log_prob = -torch.log(x + 1e-12).to(device)
        mask_sum = mask.sum(dim=1).to(device)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum).to(device)
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -pos_logits.mean()
    
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        triple_loss = 0.5 * self.alpha * self.nt_xent_loss2(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss + triple_loss
