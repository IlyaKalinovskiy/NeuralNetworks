import numpy as np


class Mirroring:
    """Apply mirror trick for face images, randomly mirror input images."""

    def __init__(self, m_prob):
        self.m_prob = m_prob

    def flip(self, img):
        """Run mirroring."""
        if np.random.uniform(0, 1) < self.m_prob:
            return np.fliplr(img)
        else:
            return img
