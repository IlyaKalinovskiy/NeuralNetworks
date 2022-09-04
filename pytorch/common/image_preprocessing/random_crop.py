import numpy as np


class Cropper:
    """Apply crop trick for images."""

    def __init__(self, extend_size):
        self.extend_size = extend_size

    def crop(self, img):
        """Run cropping."""
        pad_width = tuple([self.extend_size] * 2)
        padded = np.pad(img, [pad_width, pad_width, (0, 0)], mode="reflect")
        dx = np.random.randint(2 * self.extend_size)
        dy = np.random.randint(2 * self.extend_size)
        res = padded[dx : dx + img.shape[0], dy : dy + img.shape[1], :]
        return res
