#!/usr/bin/env python3
"""
This module extracts RootSIFT features from a list of images.
"""

from concurrent import futures
from functools import partial
import os
import pickle
import time

import click
import cv2
from loguru import logger
import numpy as np

SIFT = None


def resize_image(image_data: np.ndarray, resize_dim: int = 640) -> np.ndarray:
    h, w, _ = image_data.shape
    scale_factor = resize_dim / min(h, w)
    scaled_h = int(scale_factor * h)
    scaled_w = int(scale_factor * w)
    return cv2.resize(image_data, (scaled_w, scaled_h))


def get_keypoints_and_rootsift_descriptors(image_data, eps=1e-7):
    raw_keypoints, descriptors = SIFT.detectAndCompute(image_data, mask=None)
    if descriptors is None:
        return ([], None)
    # apply the Hellinger kernel by first L1-normalizing and taking the
    # square-root
    descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
    descriptors = np.sqrt(descriptors)
    # Getting a list of tuples (x, y) that can be serialized
    keypoints = [kp.pt for kp in raw_keypoints]
    return (keypoints, descriptors)


def extract_features_and_metadata(image_path: str, resize_dim: int):
    try:
        image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_data = resize_image(image_data, resize_dim)
        height, width, _ = image_data.shape
        keypoints, descriptors = get_keypoints_and_rootsift_descriptors(image_data)
        return (image_path, keypoints, descriptors, height, width)
    except:
        logger.exception('Failed to process {}, skipping...'.format(image_path))
        return (image_path, [], None, None, None)


def dict2numpy(descriptor_dict, descriptor_count):
    array = np.zeros((descriptor_count, 128), dtype=np.float32)
    pivot = 0
    for key in descriptor_dict.keys():
        value = descriptor_dict[key]
        nelements = value.shape[0]
        array[pivot:pivot + nelements] = value
        pivot += nelements
    return array


@click.command()
@click.option('--image_path_list', required=True, help='A list of paths to the images')
@click.option(
    '--num_features',
    type=int,
    default=0,
    help=
    'Number of best features to extract per image, features are ranked by their scores (measured in SIFT algorithm as the local contrast) [default of 0, retains all]'
)
@click.option('--workers', default=30, help='Number of workers to use in feature extraction')
@click.option('--feature_matrix_path', default='artifacts/feature_matrix.npy', help='Path of the produced feature matrix')
@click.option('--keypoints_path', default='artifacts/keypoints.pkl', help='Path of the produced keypoints file')
@click.option('--descriptors_path', default='artifacts/descriptors.pkl', help='Path of the produced descriptors file')
@click.option('--metadata_path', default='artifacts/metadata.pkl', help='Path of the produced metadata file')
@click.option('--resize_dim', type=int, required=True, help='Image resize dimension')
def main(image_path_list: str, num_features: int, workers: int, feature_matrix_path: str, keypoints_path: str,
         descriptors_path: str, metadata_path: str, resize_dim: int):
    global SIFT
    os.makedirs(os.path.dirname(feature_matrix_path), exist_ok=True)

    image_paths = []
    logger.info('Reading image paths ...')
    with open(image_path_list) as fp:
        for line in fp:
            image_paths.append(line.strip())
    logger.info('Number of images: {}'.format(len(image_paths)))

    logger.info('Extracting features and metadata...')
    logger.info('resize_dim={}'.format(resize_dim))

    tic = time.time()
    if num_features:
        logger.info('Overriding SIFT nFeatures={}'.format(num_features))
        SIFT = cv2.SIFT_create(nfeatures=num_features)
    else:
        SIFT = cv2.SIFT_create()
    logger.info('Building up work queue...')
    extract = partial(extract_features_and_metadata, resize_dim=resize_dim)

    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        extraction_results = executor.map(extract, image_paths)
    logger.info('Finished processing images - elapsed {} seconds'.format(time.time() - tic))

    logger.info('Serializing keypoints, descriptors, features, and metadata information...')
    tic = time.time()
    descriptor_dict = {}
    keypoint_dict = {}
    metadata_dict = {}
    descriptor_count = 0
    none_count = 0
    for result in extraction_results:
        image_path, keypoints, descriptors, height, width = result
        if descriptors is not None:
            descriptor_count += len(descriptors)
            descriptor_dict[image_path] = descriptors
            keypoint_dict[image_path] = keypoints
            metadata_dict[image_path] = {'height': height, 'width': width}
        else:
            none_count += 1

    logger.info('Total descriptor count: {}'.format(descriptor_count))
    logger.info('Skipped image count: {}'.format(none_count))

    local_descriptors_path = descriptors_path
    local_keypoints_path = keypoints_path
    local_metadata_path = metadata_path
    with open(local_descriptors_path, 'wb') as fp:
        pickle.dump(descriptor_dict, fp)
    with open(local_keypoints_path, 'wb') as fp:
        pickle.dump(keypoint_dict, fp)
    with open(local_metadata_path, 'wb') as fp:
        pickle.dump(metadata_dict, fp)
    del keypoint_dict
    del metadata_dict
    logger.info('Finished serializing - elapsed {} seconds'.format(time.time() - tic))

    logger.info('Constructing feature matrix from descriptors...')
    tic = time.time()
    local_feature_matrix_path = feature_matrix_path
    feature_matrix = dict2numpy(descriptor_dict, descriptor_count)
    logger.info(f'feature matrix shape = {feature_matrix.shape}')
    np.save(local_feature_matrix_path, feature_matrix)
    logger.info('Finished constructing feature matrix - elapsed {} seconds'.format(time.time() - tic))

    logger.info('Done.')


if __name__ == '__main__':
    main()
