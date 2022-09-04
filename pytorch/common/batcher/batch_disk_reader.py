# common
import logging

from multiprocessing.queues import Queue

import torch

from pytorch.common.abstract.abstract_process_worker import AbstractProcessWorker
from pytorch.common.batcher.batch_primitives import Batch

logger = logging.getLogger("root")


class BatchDiskReader(AbstractProcessWorker):
    def __init__(self, in_queue, out_queue, image_processor, audio_processor):
        AbstractProcessWorker.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.image_processor = image_processor
        self.audio_processor = audio_processor

    def do_work_once(self):
        try:
            input_batch = self.in_queue.get(timeout=1)
        except Exception as e:
            try:
                if e is Queue.Empty:
                    logger.error("Disk_reader: do_work_once() got empty input queue")
            except:
                pass
            return

        try:
            out_samples = []
            for data_sample in input_batch.data_samples:
                try:
                    # load data
                    if not self.image_processor is None:
                        image = self.image_processor.process(data_sample.img_rel_path)
                        if image is None:
                            logger.error(
                                "BatchDiskReader: cant load image from %s"
                                % data_sample.img_rel_path
                            )
                            continue
                        data_sample.image = image

                    if not self.audio_processor is None:
                        spec = self.audio_processor.process(data_sample.wav_rel_path)
                        if spec is None:
                            logger.error(
                                "BatchDiskReader: cant load wav from %s"
                                % data_sample.wav_rel_path
                            )
                            continue
                        data_sample.image = spec

                    out_samples.append(data_sample)
                except Exception as e:
                    logger.error("BatchDiskReader: %s" % e)

            if not self.audio_processor is None:
                max_len = max([x.image.size(1) for x in out_samples]) + 1
                hspec = out_samples[0].image.size(0)
                for x in out_samples:
                    spec_padded = torch.FloatTensor(hspec, max_len)
                    spec_padded.zero_()
                    spec_padded[:, : x.image.size(1)] = x.image
                    x.image = spec_padded

            # put data to out queue
            self.out_queue.put(Batch(out_samples))
        except Exception as e:
            logger.error("BatchDiskReader: %s" % e)

        self.in_queue.task_done()
