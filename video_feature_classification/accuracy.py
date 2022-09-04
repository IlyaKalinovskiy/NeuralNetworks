import matplotlib

matplotlib.use("Agg")
import itertools

from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix


class Accuracy:
    def __init__(self, data, experiment_name=""):
        super().__init__()
        self.data = []
        for clip in data:
            self.data.append([0] * len(clip.data_samples))
        self.experiment_name = experiment_name
        self.target_all = [sample.labels for clip in data for sample in clip.data_samples]
        self.target_all = np.asarray(self.target_all, dtype=np.int32)
        self.target_clips = [clip.labels for clip in data]
        self.target_clips = np.asarray(self.target_clips, dtype=np.int32)
        self.target_names = sorted([str(int(l)) for l in Counter(self.target_all).keys()])

    def plot_confusion_matrix(
        self, cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """This function prints and plots the confusion matrix.

        Normalization can be applied by setting `normalize=True`.

        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        print(title + "\n", cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    def calc_cnf_matrix(self, target, predict):
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(target, predict)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        title = "Confusion matrix"
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.target_names, title=title)
        plt.savefig(self.experiment_name + "_" + title + ".png")

        # Plot normalized confusion matrix
        title = "Normalized confusion matrix"
        plt.figure()
        self.plot_confusion_matrix(
            cnf_matrix, classes=self.target_names, normalize=True, title=title
        )
        plt.savefig(self.experiment_name + "_" + title + ".png")

    def by_frames(self, predict):
        predict = np.asarray(predict, np.int32)
        assert self.target_all.shape[0] == predict.shape[0], "Invalid predict!"
        print(
            classification_report(
                self.target_all, predict, target_names=self.target_names
            )
        )
        self.calc_cnf_matrix(self.target_all, predict)

    def by_clips(self, predict):
        predict_clips = []
        idx = 0
        for smaple_per_clip in self.data:
            pred = predict[idx : idx + len(smaple_per_clip)]
            idx += len(smaple_per_clip)
            predict_clips.append(int(np.median(pred)))

        predict_clips = np.asarray(predict_clips, dtype=np.int32)
        assert self.target_clips.shape[0] == predict_clips.shape[0], "Invalid predict!"

        print(
            classification_report(
                self.target_clips, predict_clips, target_names=self.target_names
            )
        )
        self.calc_cnf_matrix(self.target_clips, predict_clips)


class AccuracyRegression:
    def __init__(self, data, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        self.target_clips = [[clip.valence, clip.arousal] for clip in data]
        self.target_clips = np.asarray(self.target_clips, dtype=np.float32)
        self.target_names = ["Valence", "Arousal"]

    def by_clips(self, targets, predict):
        predict_clips = np.asarray(predict, dtype=np.float32)

        result = []
        for k, name in enumerate(self.target_names):
            target = torch.from_numpy(self.target_clips[:, k])
            pred = torch.from_numpy(predict_clips[:, k])
            test_acc = torch.nonzero(
                F.relu(-(target - pred).abs_() + self.threshold)
            ).size(0)
            test_acc *= 100 / self.target_clips.shape[0]
            test_err = F.relu((target - pred).abs_() - self.threshold)
            test_err = test_err[test_err.nonzero()]
            result.append(test_acc)
            print(name + ":")
            print("   accuracy per clips: %0.3f%%" % test_acc)
            print(
                f"   error per clips: mean={test_err.mean():0.3f}, std={test_err.std():0.3f}"
            )
        print("---------\n")
        return result

    def __call__(self, targets, predict):
        return self.by_clips(targets, predict)
