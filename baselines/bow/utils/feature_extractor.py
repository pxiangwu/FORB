"""
Classes and functions for performing feature extraction
"""
import abc
from typing import Dict

import cv2
from loguru import logger
import numpy as np

EPS = 1e-7


class FeatureExtractor(abc.ABC):
    """Transforms image data into some feature map representing the query."""

    def __init__(self, config: Dict[str, str] = None):
        self.config = config
        self.op_name = 'feature_extraction'

    @abc.abstractmethod
    def extract(self, image_data: np.ndarray) -> Dict:
        raise NotImplementedError


class RootSift(FeatureExtractor):
    """Extracts RootSift descriptors and keypoints from image data"""

    def __init__(self, config: Dict[str, str] = None, num_features: int = None, sift: object = None, eps: float = EPS):
        super(RootSift, self).__init__(config)
        self.eps = eps

        if config is not None:
            self.num_features = int(config.get('SIFT_NUM_FEATURES', 0))
            if self.num_features:
                logger.info(f'Overriding SIFT nFeatures={self.num_features}')
                self.sift = cv2.SIFT_create(nfeatures=self.num_features)
            else:
                self.sift = cv2.SIFT_create()
        else:
            self.num_features = num_features
            self.sift = sift

    def extract(self, image_data: np.ndarray) -> Dict:
        result = self.extract_keypoints_and_descriptors(image_data)
        return result

    def extract_keypoints_and_descriptors(self, image_data: np.ndarray) -> Dict:
        raw_keypoints, descriptors = self.sift.detectAndCompute(image_data, mask=None)
        if descriptors is None or descriptors.size == 0:
            return {'keypoints': [], 'descriptors': None}

        # We return None when descriptors.shape == (1, 128) here
        # Otherwise when we use cv2.BFMatcher().knnMatch() in downstream components
        # It will not be able to find 2-NNs and will raise errors like
        # ValueError: not enough values to unpack (expected 2, got 1)
        if descriptors.shape[0] == 1:
            return {'keypoints': [], 'descriptors': None}
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + self.eps)
        descriptors = np.sqrt(descriptors)

        # Getting a list of tuples (x, y) that can be serialized
        keypoints = [kp.pt for kp in raw_keypoints]

        return {'keypoints': keypoints, 'descriptors': descriptors}
