import torch.nn as nn


class TotalLoss(nn.Module):
    def __init__(self, loss_param, num_samples_per_classes, cuda_id):
        super().__init__()
        self.loss_param = loss_param
        self.loss_types = list(loss_param.keys())

    def forward(
        self, logits, targets, emb=None, emb_norm=None, step=None, summary_writer=None
    ):
        total_loss = 0

        if "MSE" in self.loss_types:
            total_loss += self.loss_param["MSE"]["w"] * nn.MSELoss()(logits, targets)

        # TODO: реализуйте objective function из статьи https://arxiv.org/pdf/1704.08619.pdf

        return total_loss
