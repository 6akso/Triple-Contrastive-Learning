import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)


class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            mask = mask ^ torch.diag_embed(torch.diag(mask))

        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - (torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12))
        # in case that mask.sum(1) is zer
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def nt_xent_loss2(self, anchor, target, labels):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        labels = labels.unsqueeze(-1)
        mask = torch.eq(labels, labels.transpose(0, 1))
        mask = mask ^ torch.diag_embed(torch.diag(mask))
        mask2 = torch.ne(labels, labels.transpose(0, 1))
        mask3 = torch.eq(labels, labels.transpose(0, 1))
        negtive_dot_negtive_sum = torch.zeros(mask2.size(0))
        negtivesum = torch.zeros(mask2.size(0))

        for i in range(mask2.size(0)):
            if mask2[i].sum() == 0:
                continue
            negtive_mask = mask2[i]
            # Include the anchor and samples with the same label as the anchor
            # same_label_indices = torch.where(mask3[i])[0]
            # negtive_indices = torch.cat((same_label_indices, torch.where(negtive_mask)[0])).unique()
            negtive_indices = torch.where(negtive_mask)[0]
            if negtive_indices.size(0) == 0:
                continue
            negtive_features = anchor[negtive_indices]
            if negtive_features.size(0) == 0:
                continue
            different_label_mask = (
                    labels[negtive_indices].unsqueeze(1) != labels[negtive_indices].unsqueeze(0)).float()
            dot_product_matrix = torch.matmul(negtive_features, negtive_features.T) / 0.05

            # Apply mask to only consider different labels
            dot_product_matrix = dot_product_matrix * different_label_mask

            # Max for numerical stability
            maxdot = dot_product_matrix.max(dim=1, keepdim=True)[0]
            exp_terms = torch.exp(dot_product_matrix - maxdot)

            # Sum the exponentiation terms and count the negatives
            negtive_dot_negtive_sum[i] = exp_terms.sum()
            negtivesum[i] = different_label_mask.sum()

        x = negtive_dot_negtive_sum.view(-1, 1)
        negtivesum = negtivesum.view(-1, 1)
        negtivesum = torch.repeat_interleave(negtivesum, mask.size(1), dim=1)
        x = torch.repeat_interleave(x, mask.size(1), dim=1)

        if not torch.all(negtivesum == 0):
            x = x / negtivesum
        log_prob = (- (torch.log(x + 1e-12))).to(device)
        mask_sum = (mask.sum(dim=1)).to(device)
        mask_sum = (torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)).to(device)
        # compute log-likelihood
        pos_logits = ((mask * log_prob).sum(dim=1) / mask_sum.detach()).to(device)
        loss = (-1 * pos_logits.mean()).to(device)

        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        triple_loss=0.5*self.alpha * self.nt_xent_loss2(normed_cls_feats, normed_cls_feats, targets)

        return ce_loss + cl_loss+triple_loss





