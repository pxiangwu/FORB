"""
Classes and functions for performing homography related logic
"""
from typing import Dict

import cv2
from loguru import logger
import numpy as np

DEFAULT_BBOX_MIN_AREA = 0.05
DEFAULT_BBOX_MIN_WIDTH = 0.1
DEFAULT_BBOX_MIN_HEIGHT = 0.1
DEFAULT_N1_UPPERBOUND = 6.0
DEFAULT_N1_LOWERBOUND = 0.05
DEFAULT_N2_UPPERBOUND = 6.0
DEFAULT_N2_LOWERBOUND = 0.05
DEFAULT_N3_UPPERBOUND = 0.002


class MatchingBoundingPolygonChecker:

    def __init__(self,
                 config: Dict[str, str] = None,
                 metrics_obj: object = None,
                 bbox_min_area: float = None,
                 bbox_min_width: float = None,
                 bbox_min_height: float = None):
        self.config = config
        self.metrics_obj = metrics_obj

        if config is not None:
            self.bbox_min_area = float(config.get('MATCHING_BBOX_MIN_AREA', DEFAULT_BBOX_MIN_AREA))
            self.bbox_min_width = float(config.get('MATCHING_BBOX_MIN_WIDTH', DEFAULT_BBOX_MIN_WIDTH))
            self.bbox_min_height = float(config.get('MATCHING_BBOX_MIN_HEIGHT', DEFAULT_BBOX_MIN_HEIGHT))
        else:
            self.bbox_min_area = bbox_min_area
            self.bbox_min_width = bbox_min_width
            self.bbox_min_height = bbox_min_height

    @staticmethod
    def get_object_corners(db_image_width: int, db_image_height: int, H: np.ndarray) -> np.ndarray:
        """
        Returns a list of 4 xy coordinates representing the bounding polygon of
        a database image (i.e. marker) in a given query image based on a
        provided homography perspective transformation matrix H

        Output will be in absolute pixels like:
        array([[ 84, 128],[376, 129],[377, 470],[ 83, 471]])

        Note that the pixel values may lie outside of the image (e.g. the query image
        only contains part of the marker and rest is assumed to be off screen)
        """

        db_object_corners = np.array(
            [[[0, 0], [db_image_width, 0], [db_image_width, db_image_height], [0, db_image_height]]], dtype='float32')
        return cv2.perspectiveTransform(db_object_corners, H).astype(np.int32).squeeze()

    def is_bounding_box_good(self, object_corners: np.ndarray, query_image_width: int, query_image_height: int) -> bool:
        bounding_rect = cv2.boundingRect(object_corners)
        logger.debug(f'bounding_rect={bounding_rect}')
        _, _, w, h = bounding_rect
        rel_area = (h / query_image_height) * (w / query_image_width)
        rel_w = (w / query_image_width)
        rel_h = (h / query_image_height)

        logger.debug(f'Matching BBOX check values: rel_area={rel_area}, rel_w={rel_w}, rel_h={rel_h}')

        return rel_area >= self.bbox_min_area and \
            rel_w >= self.bbox_min_width and \
            rel_h >= self.bbox_min_height


class HomographyChecker:

    def __init__(self,
                 config: Dict[str, str] = None,
                 metrics_obj: object = None,
                 n1_upperbound: float = None,
                 n1_lowerbound: float = None,
                 n2_upperbound: float = None,
                 n2_lowerbound: float = None,
                 n3_upperbound: float = None):
        self.config = config
        self.metrics_obj = metrics_obj

        if config is not None:
            self.n1_upperbound = float(config.get('HOMOGRAPHY_N1_UPPERBOUND', DEFAULT_N1_UPPERBOUND))
            self.n1_lowerbound = float(config.get('HOMOGRAPHY_N1_LOWERBOUND', DEFAULT_N1_LOWERBOUND))
            self.n2_upperbound = float(config.get('HOMOGRAPHY_N2_UPPERBOUND', DEFAULT_N2_UPPERBOUND))
            self.n2_lowerbound = float(config.get('HOMOGRAPHY_N2_LOWERBOUND', DEFAULT_N2_LOWERBOUND))
            self.n3_upperbound = float(config.get('HOMOGRAPHY_N3_UPPERBOUND', DEFAULT_N3_UPPERBOUND))
        else:
            self.n1_upperbound = n1_upperbound
            self.n1_lowerbound = n1_lowerbound
            self.n2_upperbound = n2_upperbound
            self.n2_lowerbound = n2_lowerbound
            self.n3_upperbound = n3_upperbound

    def is_homography_good(self, H: np.ndarray) -> bool:
        """
        Performs checks on provided homography based on:
        https://answers.opencv.org/question/2588/check-if-homography-is-good/
        """
        det = H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]
        N1 = np.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0])
        N2 = np.sqrt(H[0, 1] * H[1, 1] + H[1, 1] * H[1, 1])
        N3 = np.sqrt(H[2, 0] * H[2, 1] + H[2, 1] * H[2, 1])

        logger.debug(f'Homography check values: det={det}, N1={N1}, N2={N2}, N3={N3}')

        return not (det < 0) and \
            not (N1 > self.n1_upperbound or N1 < self.n1_lowerbound) and \
            not (N2 > self.n2_upperbound or N2 < self.n2_lowerbound) and \
            not (N3 > self.n3_upperbound)
