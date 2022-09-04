# common
import logging


# multiprocessing
from multiprocessing import JoinableQueue

# local
from pytorch.common.abstract.abstract_data_batcher import AbstractDataBatcher
from pytorch.common.batcher.batch_disk_reader import BatchDiskReader

logger = logging.getLogger("root")


class ImagesBatcher(AbstractDataBatcher):
    def __init__(
        self,
        queue_size,
        batch_size,
        data_sampler,
        image_processor=None,
        audio_processor=None,
        single_epoch=False,
        cache_data=False,  # TODO: implement me!
        disk_reader_process_num=1,
    ):
        """Class for creating sequence of data batches for training or validation.

        :param queue_size: queue size for Batch readers
        :param batch_size: size of batches generated
        :param dataset_parser: dataset structure-related parser with all images and labels
        :param image_processor: image reading and preprocessing routine
        :param data_sampler: knows how to sample batches from dataset
        :param single_epoch: if enabled, image batcher finish one epoch with None batch
        :param cache_data: do we need to store all data in batcher memory?
        :param disk_reader_process_num: how many disk readers do we need?

        """
        super(AbstractDataBatcher, self).__init__()

        # set parameters
        self.batch_size = batch_size
        self.epoch_is_finished = False
        self.batch_queue_balance = 0
        if single_epoch:
            self.sampler_external_info = type(
                "sampler_external_info", (object,), dict(single_epoch=True)
            )
        else:
            self.sampler_external_info = None

        # parse given dataset and init data sampler
        self.data_sampler = data_sampler

        # set queues
        if queue_size == -1:
            queue_size = self.data_sampler.dataset_size() / self.batch_size + 1
        self.task_queue = JoinableQueue(queue_size)
        self.batch_queue = JoinableQueue(queue_size)

        # init batch disk readers and start they
        self.data_readers = []
        print("disk_reader_process_num:", disk_reader_process_num)
        for i in range(disk_reader_process_num):
            self.data_readers.append(
                BatchDiskReader(
                    self.task_queue, self.batch_queue, image_processor, audio_processor
                )
            )

    def start(self):
        self.epoch_is_finished = False

        # start batch disk readers
        for reader in self.data_readers:
            reader.start()

        # fill task queue with batches to start async reading from disk
        self.fill_task_queue()

    def fill_task_queue(self):
        try:
            while True:
                if not self.task_queue.full():
                    batch = self.data_sampler.sampling(
                        self.batch_size, self.sampler_external_info
                    )
                    if batch is not None:
                        self.task_queue.put_nowait(batch)
                        self.batch_queue_balance += 1
                    else:
                        self.epoch_is_finished = True
                        break
                else:
                    break
        except Exception as e:  # Queue.Full:
            logger.error("ImagesBatcher: ", e)

    def next_batch(self):
        """Returns next batch from data."""
        if self.epoch_is_finished and self.batch_queue_balance == 0:
            self.epoch_is_finished = False
            self.fill_task_queue()
            return None

        batch = self.batch_queue.get(block=True)
        self.batch_queue.task_done()
        self.batch_queue_balance -= 1
        if not self.epoch_is_finished:
            # fill task queue
            self.fill_task_queue()
        return batch

    def update_sampler(self, target, logits, step, summary_writer):
        if hasattr(self.data_sampler, "update"):
            labels = target.cpu().data.numpy()
            is_update_sampler = self.data_sampler.update(
                labels, logits, step, summary_writer
            )
        # if is_update_sampler:
        #    self.clear_queue()

    def clear_queue(self):
        try:
            while True:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
        except Exception as e:
            pass
        try:
            while True:
                self.batch_queue.get_nowait()
                self.batch_queue.task_done()
        except Exception as e:
            pass
        self.fill_task_queue()

    def finish(self):
        for data_reader in self.data_readers:
            data_reader.deactivate()

        while not self.task_queue.empty():
            self.task_queue.get()
            self.task_queue.task_done()

        is_anybody_alive = [
            data_reader.is_alive() for data_reader in self.data_readers
        ].count(True) > 0
        while not self.batch_queue.empty() or is_anybody_alive:
            try:
                self.batch_queue.get(timeout=1)
                self.batch_queue.task_done()
                is_anybody_alive = [
                    data_reader.is_alive() for data_reader in self.data_readers
                ].count(True) > 0
            except Exception as e:
                pass

        self.task_queue.join()
        self.batch_queue.join()
        for data_reader in self.data_readers:
            data_reader.join()
