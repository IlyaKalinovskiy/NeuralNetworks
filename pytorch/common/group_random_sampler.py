import sys
import logging
import itertools

from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F

from pytorch.common.abstract.abstract_data_sampler import AbstractDataSampler
from pytorch.common.batcher.batch_primitives import Batch, DataGroup, DataSample

logger = logging.getLogger("root")


class GroupRandomSampler(AbstractDataSampler):
    """Random batch sampler for datasets like celeba attributes.

    Returns batches randomly.

    """

    def __init__(
        self,
        data,
        num_sample_per_classes=4,
        samples_is_randomize=True,
        step_size_for_samples=0,
        is_shuffle=True,
    ):
        super(AbstractDataSampler, self).__init__()
        # initial random shuffling of the dataset
        self.data = data
        self.num_sample_per_classes = num_sample_per_classes
        self.samples_is_randomize = samples_is_randomize
        self.step_size_for_samples = step_size_for_samples
        self.is_shuffle = is_shuffle

        if self.is_shuffle:
            shuffle(self.data)
        self.current_idx = 0
        self.is_last_batch = False

        # final preparing of the data
        self.dataset_size = len(self.data)
        self.samples_count = 0
        if type(self.data[0]) is DataGroup:
            for clip in self.data:
                self.samples_count += len(clip.data_samples)
        else:
            self.samples_count += len(self.data)

    def gen_samples_id(
        self,
        data_samples,
        num_sample_per_classes,
        samples_is_randomize=True,
        step_size_for_samples=1,
    ):
        num_samples = len(data_samples)
        if samples_is_randomize:
            if num_samples >= num_sample_per_classes:
                samples_id = np.random.choice(
                    num_samples, num_sample_per_classes, replace=False
                )
            else:
                samples_id = [i for i in range(num_samples)]
                samples_id = list(
                    itertools.chain(
                        samples_id,
                        np.random.choice(
                            num_samples, num_sample_per_classes - num_samples
                        ),
                    )
                )
        else:
            if step_size_for_samples > 0:
                step = step_size_for_samples
            else:
                step = np.random.randint(4, size=1)[0]
            max_idx = (num_samples - num_sample_per_classes) // step
            if max_idx <= 0:
                max_idx = num_samples
            begin_idx = np.random.randint(max_idx, size=1)[0]
            samples_id = [
                (begin_idx + step * i) % num_samples
                for i in range(num_sample_per_classes)
            ]
        samples_id.sort()
        return samples_id

    def gen_samples(self, data_list):
        if type(self.data[0]) is DataSample:
            return data_list

        batch_data = []
        for data in data_list:
            samples_id = self.gen_samples_id(
                data.data_samples,
                self.num_sample_per_classes,
                self.samples_is_randomize,
                self.step_size_for_samples,
            )
            for id in samples_id:
                batch_data.append(data.data_samples[id])

        return batch_data

    def sampling(self, batch_size, external_info=None):
        if self.is_last_batch:
            # got last batch. Reset state and return None batch
            self.is_last_batch = False
            return None

        if self.current_idx + batch_size >= self.dataset_size:
            last_single_epoch_batch = (
                hasattr(external_info, "single_epoch") and external_info.single_epoch
            )
            if last_single_epoch_batch is True:
                # save last part of data
                data_list = self.data[self.current_idx :]
                if self.is_shuffle:
                    shuffle(data_list)
                self.is_last_batch = True

            # restart dataset batching
            if self.is_shuffle:
                shuffle(self.data)
            self.current_idx = 0

            if last_single_epoch_batch is True:
                # additionally, if we need only one epoch
                # return last part of the data and set is_last_batch to True
                return Batch(self.gen_samples(data_list))

        # if we not in the end of dataset just return next batch
        data_list = self.data[self.current_idx : self.current_idx + batch_size]
        self.current_idx += batch_size
        if self.is_shuffle:
            shuffle(data_list)
        return Batch(self.gen_samples(data_list))

    def update(self, labels, logits, step, summary_writer):
        softmax = F.softmax((logits - torch.max(logits)).mul(10.0), dim=1)
        values, indices = torch.max(softmax, 1)
        values, indices = values.data, indices.data

        TP = []
        for i, label in enumerate(labels):
            s_id = int(label)
            if indices[i] == s_id:
                TP.append(s_id)

        sys.stdout.write("\rACC={:0.3f}%\t".format(len(TP) / len(labels) * 100.0))
        sys.stdout.flush()

        return False

    def get_dataset_size(self):
        return self.dataset_size

    def get_samples_count(self):
        return self.samples_count
