import torch
import torch.nn as nn


class Fourdloss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.035, contrast_mode='all', base_temperature=0.07):
        super(Fourdloss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
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

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        if labels is not None:
            mask2 = torch.ne(labels, labels.T).to(device)
        else:
            mask2 = ~mask.bool()
        mask2 = mask2.repeat(anchor_count, contrast_count)

        if labels is not None:
            mask3 = torch.eq(labels, labels.T).to(device)
        else:
            mask3 = mask.bool()
        mask3 = mask3.repeat(anchor_count, contrast_count)

        negtive_dot_negtive_sum = torch.zeros(mask2.size(0)).to(device)
        negtivesum = torch.zeros(mask2.size(0)).to(device)

        labels2 = torch.flatten(labels)
        labels2 = torch.cat([labels2, labels2.repeat(anchor_count)]).to(device)

        for i in range(mask2.size(0)):
            if mask2[i].sum() == 0:
                continue
            negtive_mask = mask2[i]
            # Include the anchor and samples with the same label as the anchor
            same_label_indices = torch.where(mask3[i])[0]
            negtive_indices = torch.cat((same_label_indices, torch.where(negtive_mask)[0])).unique()
            if negtive_indices.size(0) == 0:
                continue
            negtive_features = contrast_feature[negtive_indices]
            if negtive_features.size(0) == 0:
                continue
            different_label_mask = (
                    labels2[negtive_indices].unsqueeze(1) != labels2[negtive_indices].unsqueeze(0)).float()
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

        log_prob = -torch.log(x + 1e-6)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        pos_logits = -(mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = pos_logits.view(anchor_count, batch_size).mean()

        return loss
