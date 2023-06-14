"""
Classes and functions for performing result filtering
"""

import abc
from typing import Dict, List

import cv2
from loguru import logger
import numpy as np

from utils.homography import HomographyChecker, MatchingBoundingPolygonChecker

DEFAULT_LOWE_TEST_THRESHOLD = 0.8
DEFAULT_MIN_INLIERS = 0
DEFAULT_MIN_MATCH_COUNT = 0
MIN_KEYPOINT_COUNT = 4
RANSAC_REJECT_THRESHOLD = 10.0


class RetrievalResultFilter(abc.ABC):
    """Filters candidates from the result list based on some criteria"""

    def __init__(self, config: Dict[str, str] = None):
        self.config = config
        self.op_name = 'filter_results'

    @abc.abstractmethod
    def filter(self, query_image_data: np.ndarray, results: List, max_results: int) -> List:
        raise NotImplementedError


class RansacGeometricVerification(RetrievalResultFilter):
    """
    Filters candidates based on a series of geometric verification criteria.
    Makes use of ratio tests, unique keypoints check, and RANSAC to filter results.

    Can configured to optionally make use of the following additional heuristic check filters:
    - homography.HomographyChecker
    - homography.MatchingBoundingPolygonChecker

    By default results are sorted by num_inilers.
    """

    def __init__(self,
                 config: Dict[str, str] = None,
                 lowe_test_threshold: float = None,
                 min_inliers: int = None,
                 min_match_count: int = None,
                 min_keypoint_count: int = None,
                 ransac_reject_threshold: float = None,
                 homography_checker: HomographyChecker = None,
                 bounding_polygon_checker: MatchingBoundingPolygonChecker = None):
        super(RansacGeometricVerification, self).__init__(config)

        if config is not None:
            # Parse relevant keys from config
            self.lowe_test_threshold = float(config.get('LOWE_TEST_THRESHOLD', DEFAULT_LOWE_TEST_THRESHOLD))
            self.min_inliers = int(config.get('MIN_INLIERS', DEFAULT_MIN_INLIERS))
            self.min_match_count = int(config.get('MIN_MATCH_COUNT', DEFAULT_MIN_MATCH_COUNT))
            self.min_keypoint_count = int(config.get('MIN_KEYPOINT_COUNT', MIN_KEYPOINT_COUNT))
            self.ransac_reject_threshold = float(config.get('RANSAC_REJECT_THRESHOLD', RANSAC_REJECT_THRESHOLD))

            self.homography_checker = None
            self.bounding_polygon_checker = None

            if bool(config.get('ACTIVATE_HOMOGRAPHY_CHECKER', "False") == "True"):
                self.homography_checker = HomographyChecker(config=config)
                logger.info('ACTIVATE_HOMOGRAPHY_CHECKER=True')

            if bool(config.get('ACTIVATE_BOUNDING_POLYGON_CHECKER', "False") == "True"):
                self.bounding_polygon_checker = MatchingBoundingPolygonChecker(config=config)
                logger.info('ACTIVATE_BOUNDING_POLYGON_CHECKER=True')

        else:
            self.lowe_test_threshold = lowe_test_threshold
            self.min_inliers = min_inliers
            self.min_match_count = min_match_count
            self.min_keypoint_count = min_keypoint_count
            self.ransac_reject_threshold = ransac_reject_threshold
            self.homography_checker = homography_checker
            self.bounding_polygon_checker = bounding_polygon_checker

    def filter(self, query_image_data: np.ndarray, results: List) -> List:
        filtered_results = []

        for result in results:
            query_kp = result.features['query_kp']
            db_kp = result.features['db_kp']
            query_des = result.features['query_des']
            db_des = result.features['db_des']

            # ======== Perform ratio test (both ways) ========
            bf = cv2.BFMatcher()

            matches = bf.knnMatch(db_des, query_des, k=2)
            candidates = set()
            for m, n in matches:
                if m.distance < self.lowe_test_threshold * n.distance:
                    candidates.add((m.trainIdx, m.queryIdx))

            matches = bf.knnMatch(query_des, db_des, k=2)
            good = []
            for m, n in matches:
                foo = (m.queryIdx, m.trainIdx)
                if m.distance < self.lowe_test_threshold * n.distance and foo in candidates:
                    good.append([m])

            # ======== Perform ransac ========
            # Min match count check
            match_count = len(good)
            if match_count < self.min_match_count:
                continue

            # Unique key point check src = db, dst = query
            query_pts = np.zeros((match_count, 2), dtype=np.float32)
            db_pts = np.zeros((match_count, 2), dtype=np.float32)

            for idx, match in enumerate(good):
                query_pts[idx, :] = query_kp[match[0].queryIdx]
                db_pts[idx, :] = db_kp[match[0].trainIdx]
            unique_query_pts = np.unique(query_pts, axis=0).shape[0]
            unique_db_pts = np.unique(db_pts, axis=0).shape[0]
            if unique_query_pts < self.min_keypoint_count or unique_db_pts < self.min_keypoint_count:
                continue

            # Estimate homography
            perspective_transformation_matrix, mask = cv2.findHomography(
                srcPoints=db_pts,
                dstPoints=query_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reject_threshold)
            if perspective_transformation_matrix is None:
                continue

            # Optional heuristic check - nice homography
            if self.homography_checker and not self.homography_checker.is_homography_good(
                    perspective_transformation_matrix):
                continue

            # ======== Calculate inliers ========
            # SIFT may contain duplicate (x, y) keypoints when feature estimation fails.
            # Therefore the input point set for RANSAC may contain duplicates too.
            concat_pts = np.hstack((db_pts, query_pts))
            masked_pts = concat_pts[mask[:, 0] == 1]
            num_inliers = np.unique(masked_pts, axis=0).shape[0]
            inliers_ratio = float(num_inliers) / match_count
            if num_inliers < self.min_inliers:
                continue

            # Optional heuristic check - bounding polygon constraints
            object_corners = None
            if self.bounding_polygon_checker:
                object_corners = MatchingBoundingPolygonChecker.get_object_corners(result.metadata['width'],
                                                                                   result.metadata['height'],
                                                                                   perspective_transformation_matrix)
                if not self.bounding_polygon_checker.is_bounding_box_good(object_corners, query_image_data.shape[1],
                                                                          query_image_data.shape[0]):
                    continue

            # ======== Result passes filter, append to return list ========
            result.features['perspective_transformation_matrix'] = perspective_transformation_matrix
            result.features['num_inliers'] = num_inliers
            result.features['inliers_ratio'] = inliers_ratio
            filtered_results.append(result)

            filtered_results = sorted(filtered_results, key=lambda x: x.features['num_inliers'], reverse=True)
        return filtered_results
