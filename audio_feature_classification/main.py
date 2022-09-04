import sys

sys.path.append("../")

import pickle
import random

from pathlib import Path

import numpy as np

from accuracy import Accuracy
from tqdm import tqdm
from voice_feature_extraction import OpenSMILE

from pytorch.common.datasets_parsers.av_parser import AVDBParser


def get_data(dataset_root, file_list, max_num_clips: int = 0):
    dataset_parser = AVDBParser(dataset_root, file_list, max_num_clips=max_num_clips)
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data


def calc_features(data, opensmile_root_dir, opensmile_config_path):
    vfe = OpenSMILE(opensmile_root_dir, opensmile_config_path)

    progresser = tqdm(
        iterable=range(0, len(data)),
        desc="calc audio features",
        total=len(data),
        unit="files",
    )

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        try:
            voice_feat = vfe.process(clip.wav_rel_path)
            feat.append(voice_feat)
            targets.append(clip.labels)
        except Exception as e:
            print(f"error calc voice features! {e}")
            data.remove(clip)

    print("feat count:", len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        pass
        # TODO: выполните сокращение размерности признаков с использованием PCA

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn

    y_pred = []
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = "exp_1"
    max_num_clips = 0  # загружайте только часть данных для отладки кода (0 - все данные)
    use_dump = False  # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = Path("/path/to/data")
    if 1:
        train_dataset_root = base_dir / "Ryerson/Video"
        train_file_list = base_dir / "Ryerson/train_data_with_landmarks.txt"
        test_dataset_root = base_dir / "Ryerson/Video"
        test_file_list = base_dir / "Ryerson/test_data_with_landmarks.txt"
    else:
        train_dataset_root = (
            base_dir / "OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames"
        )
        train_file_list = (
            base_dir
            / "OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt"
        )
        test_dataset_root = (
            base_dir / "OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames"
        )
        test_file_list = (
            base_dir
            / "OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt"
        )

    # opensmile configuration
    opensmile_root_dir = Path("path/to/opensmile-2.3.0")
    # TODO: поэкспериментируйте с различными конфигурационными файлами библиотеки OpenSmile
    opensmile_config_path = opensmile_root_dir / "config/avec2013.conf"

    if not use_dump:
        # load dataset
        train_data = get_data(
            train_dataset_root, train_file_list, max_num_clips=max_num_clips
        )
        test_data = get_data(
            test_dataset_root, test_file_list, max_num_clips=max_num_clips
        )

        # get features
        train_feat, train_targets = calc_features(
            train_data, opensmile_root_dir, opensmile_config_path
        )
        test_feat, test_targets = calc_features(
            test_data, opensmile_root_dir, opensmile_config_path
        )

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        with open(experiment_name + ".pickle", "wb") as f:
            pickle.dump(
                [train_feat, train_targets, test_feat, test_targets, accuracy_fn],
                f,
                protocol=2,
            )
    else:
        with open(experiment_name + ".pickle", "rb") as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(
                f
            )

    # run classifiers
    classification(
        train_feat,
        test_feat,
        train_targets,
        test_targets,
        accuracy_fn=accuracy_fn,
        pca_dim=0,
    )
