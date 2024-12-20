import torch
from torch import nn
import torch.nn.functional as F

class QuadrupletLoss(torch.nn.Module):
    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target,labels):
            with torch.no_grad():
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                labels = labels.unsqueeze(-1)
                mask = torch.eq(labels, labels.transpose(0, 1))
                mask = mask ^ torch.diag_embed(torch.diag(mask))
                mask2 = torch.ne(labels, labels.transpose(0, 1))
                negtive_dot_negtive_sum = torch.zeros(mask2.size(0))
                negtivesum=torch.zeros(mask2.size(0))
            for i in range(mask2.size(0)):
                if mask2[i].sum() == 0:
                    continue
                negtive_mask = mask2[i]
                negtive_indices = torch.where(negtive_mask)[0]
                negtive_features = anchor[negtive_indices]
                if negtive_features.size(0) == 0:
                    continue
                different_label_mask = (
                        labels[negtive_indices].unsqueeze(1) != labels[negtive_indices].unsqueeze(0)).float()
                different_label_mask = different_label_mask.squeeze(-1)

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
            if(not torch.all(negtivesum == 0)):
                x=x/negtivesum
            log_prob = (- (torch.log(x+ 1e-12))).to(device)
            mask_sum = (mask.sum(dim=1)).to(device)
            mask_sum = (torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)).to(device)
            # compute log-likelihood
            pos_logits = ((mask * log_prob).sum(dim=1) / mask_sum.detach() ).to(device)
            loss = (-1 * pos_logits.mean()).to(device)

            return loss

class QuadrupletLoss2(QuadrupletLoss):

    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        cl_loss_1 = (0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)).to(device)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)
        clloss=(cl_loss_2+cl_loss_1)
        return clloss



