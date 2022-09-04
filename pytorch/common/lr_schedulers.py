import math

from pytorch.common.abstract.abstract_lr_scheduler import AbstractLRScheduler


class ApplyLR:
    def __init__(self, scale_lr=[1, 1], scale_lr_fc=[1, 1]):
        self.scale_lr = scale_lr
        self.scale_lr_fc = scale_lr_fc

    def apply(self, optimizer, lr, batch_idx):
        # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
        if batch_idx < self.scale_lr[1]:
            damping = 0.5 * (
                1.0 + math.cos(math.pi * (batch_idx / float(self.scale_lr[1])))
            )
            if self.scale_lr[0] < 1.0:
                lr *= 1.0 - (1.0 - self.scale_lr[0]) * damping
            else:
                lr *= 1.0 + (self.scale_lr[0] - 1.0) * damping
        try:
            for param_group in optimizer.param_groups:
                if "fc" in param_group["name"]:
                    param_group["lr"] = lr
                    if batch_idx < self.scale_lr_fc[1]:
                        damping = 0.5 * (
                            1.0
                            + math.cos(math.pi * (batch_idx / float(self.scale_lr_fc[1])))
                        )
                        if self.scale_lr_fc[0] > 1.0:
                            param_group["lr"] *= (
                                1.0 + (self.scale_lr_fc[0] - 1.0) * damping
                            )
                        else:
                            param_group["lr"] *= (
                                1.0 - (1.0 - self.scale_lr_fc[0]) * damping
                            )
                else:
                    param_group["lr"] = lr
        except:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


class SGDR_scheduler(AbstractLRScheduler):
    def __init__(
        self, optimizer, lr_start, lr_end, lr_period, scale_lr=[1, 1], scale_lr_fc=[1, 1]
    ):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_period = lr_period
        self.lr_curr = lr_start
        self.apply_lr = ApplyLR(scale_lr, scale_lr_fc)
        self.batch_idx = 0

    def step(self):
        # returns normalised anytime sgdr schedule given period and batch_idx
        # best performing settings reported in paper are T_0 = 10, T_mult=2
        # so always use T_mult=2
        while self.batch_idx / float(self.lr_period) > 1.0:
            self.batch_idx = self.batch_idx - self.lr_period
            self.lr_period *= 2.0
            self.lr_start *= 0.2
            self.lr_end *= 0.2

        radians = math.pi * (self.batch_idx / float(self.lr_period))
        self.lr_curr = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (
            1.0 + math.cos(radians)
        )
        self.apply_lr.apply(self.optimizer, self.lr_curr, self.batch_idx)
        self.batch_idx += 1

    def get_lr(self):
        return [self.lr_curr]
