import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


class TorchImageProcessor:
    """Simple data processors."""

    def __init__(
        self,
        image_size,
        is_color,
        mean,
        scale,
        crop_size=0,
        pad=28,
        color="BGR",
        use_cutout=False,
        use_mirroring=False,
        use_random_crop=False,
        use_center_crop=False,
        use_random_gray=False,
    ):
        """Everything that we need to init."""
        pass

    def process(self, image_path):
        """Returns processed data."""
        try:
            image = cv2.imread(image_path)
        except:
            image = image_path

        if image is None:
            print(image_path)

        # TODO: реализуйте процедуры аугментации изображений используя OpenCV и TorchVision
        # на выходе функции ожидается массив numpy с нормированными значениям пикселей

        return image
