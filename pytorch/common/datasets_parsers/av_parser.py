import os
import codecs

import cv2
import numpy as np

from tqdm import tqdm

# local
from pytorch.common.abstract.abstract_dataset_parser import AbstractDatasetParser
from pytorch.common.batcher.batch_primitives import DataGroup, DataSample


class AVDBParser(AbstractDatasetParser):
    def __init__(
        self,
        dataset_root,
        file_list,
        max_num_clips=0,
        max_num_samples=0,
        ungroup=False,
        load_image=False,
        normalize=False,
    ):
        """Suppose face dataset stored as:

        :param dataset_root: root folder for dataset

        """
        super(AbstractDatasetParser, self).__init__()

        self.dataset_root = dataset_root
        self.file_list = file_list
        self.max_num_clips = max_num_clips
        self.max_num_samples = max_num_samples
        self.ungroup = ungroup
        self.load_image = load_image
        self.normalize = normalize

        self.dataset_size = sum(
            1 for line in codecs.open(self.file_list, "r", encoding="utf8")
        )
        self.train_label_names = "file_name cls_id"

        # reserve memory for data
        self.data = []
        self.class_num = 0

        self.read_data_samples()

    def get_data(self):
        return self.data

    def get_dataset_root(self):
        return self.dataset_root

    def get_dataset_size(self):
        return self.dataset_size

    def get_labels_names(self):
        return self.label_names

    def get_class_num(self):
        return self.class_num

    def read_data_samples(self):
        # Read file_list.txt file to get all meta
        with codecs.open(self.file_list, "r", "utf8") as markup_file:
            # read all data
            progresser = tqdm(
                iterable=range(0, self.dataset_size),
                desc="AVDB meta parsing",
                total=self.dataset_size,
                unit="images",
            )

            prev_idx = None
            data_samples = []
            num_persons = 0
            class_num = 0
            label_per_clip = []
            valence_per_clip = []
            arousal_per_clip = []
            for i in progresser:
                datas = markup_file.readline().strip().split()
                im_path = os.path.join(self.dataset_root, datas[0])
                wav_path = None
                valence = float(datas[1])
                arousal = float(datas[2])
                text_labels = ""

                if len(datas) == 140:
                    label = int(datas[3])
                    if "Ryerson" in im_path:
                        if label == 1:
                            continue
                        text_labels = [
                            "neutral",
                            "calm",
                            "happy",
                            "sad",
                            "angry",
                            "fearful",
                            "disgust",
                            "surprised",
                        ][label - 1]
                        wav_path = (
                            os.path.dirname(im_path.replace("Video", "Audio"))
                            .replace(".mp4", ".wav")
                            .replace("\\01", "\\03")
                        )
                    elif "OMGEmotionChallenge" in im_path:
                        wav_path = (
                            os.path.dirname(im_path.replace("frames", "wave")) + ".wav"
                        )
                        text_labels = [
                            "anger",
                            "disgust",
                            "fear",
                            "happy",
                            "neutral",
                            "sad",
                            "surprise",
                        ][label]
                    landmarks = [
                        [float(datas[2 * k + 4]), float(datas[2 * k + 5])]
                        for k in range(68)
                    ]
                    idx = os.path.dirname(datas[0])
                else:
                    label = int((valence + 10) * (arousal + 10))
                    landmarks = [
                        [float(datas[2 * k + 3]), float(datas[2 * k + 4])]
                        for k in range(68)
                    ]
                    idx = os.path.dirname(datas[0])

                class_num = max(class_num, int(label))

                if prev_idx is None:
                    prev_idx = idx
                if prev_idx != idx:
                    if self.ungroup:
                        for ds in data_samples:
                            self.data.append(ds)
                    else:
                        self.data.append(
                            DataGroup(
                                folder_rel_path=os.path.dirname(
                                    data_samples[0].img_rel_path
                                ),
                                wav_rel_path=wav_path,
                                data_samples=data_samples,
                                idx=len(self.data),
                            )
                        )
                        self.data[-1].labels = int(np.median(label_per_clip))
                        self.data[-1].valence = np.mean(valence_per_clip)
                        self.data[-1].arousal = np.mean(arousal_per_clip)
                        self.data[-1].text_labels = data_samples[0].text_labels
                        label_per_clip, valence_per_clip, arousal_per_clip = [], [], []
                    data_samples = []
                    prev_idx = idx
                    num_persons += 1

                if self.max_num_samples == 0 or len(data_samples) < self.max_num_samples:
                    data_samples.append(
                        DataSample(
                            img_rel_path=im_path,
                            wav_rel_path=wav_path,
                            labels=label,
                            valence=valence,
                            arousal=arousal,
                            landmarks=landmarks,
                        )
                    )
                    data_samples[-1].idx = idx
                    data_samples[-1].text_labels = text_labels
                    if self.load_image:
                        data_samples[-1].image = cv2.imread(im_path)
                    label_per_clip.append(label)
                    valence_per_clip.append(valence)
                    arousal_per_clip.append(arousal)

                if self.max_num_clips > 0 and num_persons == self.max_num_clips:
                    break

            if len(data_samples) > 0:
                if self.ungroup:
                    for ds in data_samples:
                        self.data.append(ds)
                else:
                    self.data.append(
                        DataGroup(
                            folder_rel_path=os.path.dirname(data_samples[0].img_rel_path),
                            wav_rel_path=wav_path,
                            data_samples=data_samples,
                            idx=len(self.data),
                        )
                    )
                    self.data[-1].labels = int(np.median(label_per_clip))
                    self.data[-1].valence = np.mean(valence_per_clip)
                    self.data[-1].arousal = np.mean(arousal_per_clip)
                    self.data[-1].text_labels = data_samples[0].text_labels
                num_persons += 1

            self.class_num = class_num
            self.data.sort(key=lambda x: x.idx)

            if self.normalize:
                if self.ungroup:
                    max_valence = max([abs(sample.valence) for sample in self.data])
                    max_arousal = max([abs(sample.arousal) for sample in self.data])
                    for sample in self.data:
                        sample.valence /= max_valence
                        sample.arousal /= max_arousal
                else:
                    max_valence = max(
                        [
                            abs(sample.valence)
                            for clip in self.data
                            for sample in clip.data_samples
                        ]
                    )
                    max_arousal = max(
                        [
                            abs(sample.arousal)
                            for clip in self.data
                            for sample in clip.data_samples
                        ]
                    )
                    for clip in self.data:
                        clip.valence /= max_valence
                        clip.arousal /= max_arousal
                        for sample in clip.data_samples:
                            sample.valence /= max_valence
                            sample.arousal /= max_arousal
