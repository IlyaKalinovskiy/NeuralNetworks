import os
import logging

import numpy as np
import torch

from tensorboardX import SummaryWriter

from pytorch.common.losses import *

logger = logging.getLogger("root")


class NetTrainer:
    def __init__(
        self,
        logs_dir,
        cuda_id,
        experiment_name,
        snapshot_dir,
        config_data=None,
        split_summary=True,
    ):
        """
        :param logs_dir: directory with TB logs
        :param cuda_id: id of cuda device for training
        :param experiment_name: unique id for current experiment
        """
        if split_summary:
            self.train_summary = SummaryWriter(
                os.path.join(logs_dir, experiment_name + "_train")
            )
            self.val_summary = SummaryWriter(
                os.path.join(logs_dir, experiment_name + "_val")
            )
        else:
            self.train_summary = SummaryWriter(os.path.join(logs_dir, experiment_name))
            self.val_summary = SummaryWriter(os.path.join(logs_dir, experiment_name))
        self.logs_dir = logs_dir
        self.cuda_id = cuda_id
        self.experiment_name = experiment_name
        self.snapshot_dir = snapshot_dir
        self.config_data = config_data
        self.net_output = None
        self.target = None
        self.current_iter = 0

    def train(
        self,
        model,
        optimizer,
        scheduler,
        train_data_batcher,
        val_data_batcher,
        val_iter,
        batch_processor,
        loss_function,
        max_iter,
        step_size,
        snapshot_iter,
        step_print=100,
        accuracy_function=None,
    ):
        """
        :param model: nn.Module network model that will be trained
        :param optimizer: optimizer from torch.optim
        :param scheduler: scheduler from torch.optim
        :param train_data_batcher: train data loading class
        :param val_data_batcher: validation data loading class
        :param batch_processor: batch pre- and post- processing unit
        :param loss_function: loss function for given net
        :param max_iter: max number of train iterations
        :param step_size: lr shcheduler step size
        :param snapshot_iter: save model every snapshot_iter iterations
        :param val_iter: run validation every validate_iter steps
        :param accuracy_function: accuracy function for given net
        """

        self.batch_size = train_data_batcher.batch_size

        # start data batchers
        train_data_batcher.start()
        if not val_data_batcher is None:
            val_data_batcher.start()

        # enable train mode of pyTroch model
        model.train()
        self.current_iter = 0
        while self.current_iter <= max_iter:
            if self.current_iter % step_size == 0:
                scheduler.step()

            if self.current_iter % snapshot_iter == 0 and self.current_iter > 0:
                snapshot_folder = os.path.join(self.snapshot_dir, self.experiment_name)
                if not os.path.exists(snapshot_folder):
                    os.makedirs(snapshot_folder)
                save_path = os.path.join(
                    snapshot_folder,
                    self.experiment_name + "_iter_" + str(self.current_iter) + ".model",
                )
                model.cpu()
                torch.save(
                    {
                        "config_data": self.config_data,
                        "model": model,
                        "current_iter": self.current_iter,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "train_data_sampler": train_data_batcher.data_sampler,
                        "batch_processor": batch_processor,
                        "loss_function": loss_function,
                    },
                    save_path,
                )
                model.cuda(self.cuda_id)

            current_lr = scheduler.get_lr()
            self.train_summary.add_scalar(
                "learning_rate", current_lr[0], global_step=self.current_iter
            )

            batch = train_data_batcher.next_batch()
            data, target = batch_processor.pre_processing(batch)

            optimizer.zero_grad()

            logits = model(data)
            loss = loss_function(logits, target)

            loss.backward()
            optimizer.step()

            if self.current_iter % step_print == 0:
                self.train_summary.add_scalar(
                    "loss", loss.data, global_step=self.current_iter
                )
                logger.critical(
                    "%s: iteration: %d: Loss: %f, lr: %f"
                    % (self.experiment_name, self.current_iter, loss.data, current_lr[0])
                )

            if (not val_data_batcher is None) and self.current_iter % val_iter == 0:
                model.eval()
                self.val(
                    model,
                    val_data_batcher,
                    batch_processor,
                    loss_function,
                    accuracy_function,
                )
                model.train()

            self.current_iter += 1

    def val(
        self,
        model,
        val_data_batcher,
        batch_processor,
        loss_function,
        accuracy_function,
        threshold=0.15,
    ):
        valid_loss = 0
        targets = np.zeros((0, 2), dtype=np.float32)
        predict = np.zeros((0, 2), dtype=np.float32)
        while True:
            batch = val_data_batcher.next_batch()
            if batch is None:
                break

            data, target = batch_processor.pre_processing(batch)

            logits = model(data)
            loss = loss_function(logits, target)
            targets = np.concatenate((targets, target.cpu().data.numpy()), axis=0)
            predict = np.concatenate((predict, logits.cpu().data.numpy()), axis=0)
            valid_loss += loss.data

        valid_acc = accuracy_function(targets, predict)
        logger.critical(
            "%s: validate. Iteration: %d: Accuracy (valence, arousal): %.3f%% %.3f%%"
            % (self.experiment_name, self.current_iter, valid_acc[0], valid_acc[1])
        )
        self.val_summary.add_scalar(
            "valid_accuracy_valence", valid_acc[0], global_step=self.current_iter
        )
        self.val_summary.add_scalar(
            "valid_accuracy_arousal", valid_acc[1], global_step=self.current_iter
        )

        valid_loss *= self.batch_size / val_data_batcher.data_sampler.get_dataset_size()
        logger.critical(
            "%s: validate. Iteration: %d: Loss: %f"
            % (self.experiment_name, self.current_iter, valid_loss)
        )
        self.val_summary.add_scalar("loss", valid_loss, global_step=self.current_iter)
