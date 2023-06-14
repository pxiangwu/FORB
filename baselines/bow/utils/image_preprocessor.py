"""
Classes and functions for performing image preprocessing
"""

import abc
from typing import Dict

import cv2
import numpy as np

DEFAULT_RESIZE_DIMENSION = 480
DEFAULT_RESIZE_TYPE = 'MIN'


class ImageProcessor(abc.ABC):

    def __init__(self, config: Dict[str, str] = None):
        self.config = config
        self.op_name = 'image_preprocessing'

    @abc.abstractmethod
    def process(self, image_path: str) -> np.ndarray:
        raise NotImplementedError


class ImageScaler(ImageProcessor):
    """
    Scales the input image based on a resize type & dimension

    The following resize types are supported:
        - `MAX`: makes the longest dimension equal to resize dimension
        - `MIN`: makes the shortest dimension equal to resize dimension
    """

    def __init__(self, config: Dict[str, str] = None, resize_dimension: int = None, resize_type: str = None):
        super(ImageScaler, self).__init__(config)

        if config is not None:
            self.resize_dimension = int(config.get('RESIZE_DIMENSION', DEFAULT_RESIZE_DIMENSION))
            self.resize_type = config.get('RESIZE_TYPE', DEFAULT_RESIZE_TYPE)
            if self.resize_type not in ('MIN', 'MAX'):
                raise ValueError(f'Unsupported RESIZE_TYPE of {self.resize_type} specified')
        else:
            self.resize_dimension = resize_dimension
            self.resize_type = resize_type

    def process(self, image_path: str) -> np.ndarray:
        image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)

        h, w, _ = image_data.shape
        if self.resize_type == 'MIN':
            scale_factor = self.resize_dimension / min(h, w)
        else:
            scale_factor = self.resize_dimension / max(h, w)
        scaled_h = int(scale_factor * h)
        scaled_w = int(scale_factor * w)
        image_data = cv2.resize(image_data, (scaled_w, scaled_h))
        return image_data
