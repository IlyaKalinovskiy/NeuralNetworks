import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr


class Accuracy:
    def __init__(self, data, threshold=0.1):
        super().__init__()
        self.data = data
        self.threshold = threshold

        self.target_all = [[sample.valence, sample.arousal] for sample in self.data]
        self.target_all = np.asarray(self.target_all, dtype=np.float32)

        idx = self.data[0].idx
        valence_per_clip, arousal_per_clip = [], []
        self.target_clips = []
        for sample in self.data:
            if idx != sample.idx:
                self.target_clips.append(
                    [np.mean(valence_per_clip), np.mean(arousal_per_clip)]
                )
                valence_per_clip, arousal_per_clip = [], []
                idx = sample.idx
            valence_per_clip.append(sample.valence)
            arousal_per_clip.append(sample.arousal)
        self.target_clips.append([np.mean(valence_per_clip), np.mean(arousal_per_clip)])
        self.target_clips = np.asarray(self.target_clips, dtype=np.float32)

        self.target_names = ["Valence", "Arousal"]

    def calc_ccc(self, targets, predict):
        true_mean = np.mean(targets)
        pred_mean = np.mean(predict)
        rho, _ = pearsonr(predict, targets)
        std_predictions = np.std(predict)
        std_gt = np.std(targets)
        ccc = (
            2
            * rho
            * std_gt
            * std_predictions
            / (std_predictions**2 + std_gt**2 + (pred_mean - true_mean) ** 2)
        )
        return ccc

    def by_frames(self, predict):
        result = []
        for k, name in enumerate(self.target_names):
            target = torch.from_numpy(self.target_all[:, k])
            pred = torch.from_numpy(predict[:, k])
            test_acc = torch.nonzero(
                F.relu(-(target - pred).abs_() + self.threshold)
            ).size(0)
            test_acc *= 100 / self.target_all.shape[0]
            test_err = F.relu((target - pred).abs_() - self.threshold)
            test_err = test_err[test_err.nonzero()]
            ccc = self.calc_ccc(self.target_all[:, k], predict[:, k])
            result.append(test_acc)
            print(name + ":")
            print("   accuracy per frames: %0.3f%%" % test_acc)
            print(
                f"   error per frames: frames={test_err.mean():0.3f}, std={test_err.std():0.3f}"
            )
            print("   concordance correlation coefficient per frames: %0.3f" % ccc)
        print("---------\n")
        return result

    def by_clips(self, predict):
        idx = self.data[0].idx
        valence_per_clip, arousal_per_clip = [], []
        predict_clips = []
        for sample, pred_val in zip(self.data, predict):
            if idx != sample.idx:
                predict_clips.append(
                    [np.mean(valence_per_clip), np.mean(arousal_per_clip)]
                )
                valence_per_clip, arousal_per_clip = [], []
                idx = sample.idx
            valence_per_clip.append(pred_val[0])
            arousal_per_clip.append(pred_val[1])
        predict_clips.append([np.mean(valence_per_clip), np.mean(arousal_per_clip)])
        predict_clips = np.asarray(predict_clips, dtype=np.float32)

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
        self.by_frames(targets, predict)
        return self.by_clips(targets, predict)


class Accuracy3D:
    def __init__(self, data, depth=1, threshold=0.1):
        super().__init__()
        self.depth = depth
        self.threshold = threshold
        self.target_names = ["Valence", "Arousal"]

    def by_clips(self, targets, predict):
        target_clips = np.asarray(targets, dtype=np.float32)
        predict_clips = np.asarray(predict, dtype=np.float32)

        result = []
        for k, name in enumerate(self.target_names):
            target = torch.from_numpy(target_clips[:, k])
            pred = torch.from_numpy(predict_clips[:, k])
            test_acc = torch.nonzero(
                F.relu(-(target - pred).abs_() + self.threshold)
            ).size(0)
            test_acc *= 100 / target_clips.shape[0]
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
