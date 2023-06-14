#!/usr/bin/env python3
"""
This module trains a k means clustering index on GPU using FAISS.
"""

import os
import time

import click
import faiss
from loguru import logger
import numpy as np


@click.command()
@click.option('--feature_matrix_path',
              required=True,
              help='Path to feature matrix to cluster, a numpy matrix of shape (# of examples, # dimensions of feature vector)')
@click.option('--k', type=int, required=True, help='Number of centroids')
@click.option('--iterations', default=20, help='Number of clustering iterations')
@click.option('--min_points_per_centroid', type=int, default=10, help='Minimum number of points required per centroid')
@click.option('--max_points_per_centroid', type=int, default=-1, help='Maximum number of points required per centroid')
@click.option('--index_path', type=str, default='artifacts/kmeans.index', help='Path of the produced k means index file')
@click.option('--centroids_path', type=str, default='artifacts/centroids.npy', help='Path of the produced k means centroids')
@click.option('--n_gpus', type=int, default=1, help='Number of GPUs to train on')
def main(feature_matrix_path: str, k: int, iterations: int, min_points_per_centroid: int, max_points_per_centroid: int,
         index_path: str, centroids_path: str, n_gpus: int):
    logger.info('Read feature matrix')
    tic = time.time()
    feature_matrix = np.load(feature_matrix_path)
    logger.info('Loading feature matrix took {:.3f} s - shape: {}'.format(time.time() - tic, feature_matrix.shape))

    d = feature_matrix.shape[1]
    if max_points_per_centroid <= 0:
        logger.info(f'Setting max_points_per_centroid to N = {feature_matrix.shape[0]}')
        # If non positive number passed in, set value to N to avoid any sampling of data points.
        # See https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids
        max_points_per_centroid = feature_matrix.shape[0]
    logger.info(
        'Prepare for gpu training: k={}, d={}, niter={}, verbose=True, min_points_per_centroid={}, max_points_per_centroid={}, gpu={}'
        .format(k, d, iterations, min_points_per_centroid, max_points_per_centroid, n_gpus))
    kmeans = faiss.Kmeans(d,
                          k,
                          niter=iterations,
                          verbose=True,
                          min_points_per_centroid=min_points_per_centroid,
                          max_points_per_centroid=max_points_per_centroid,
                          gpu=n_gpus)

    logger.info('Start training...')
    tic = time.time()
    kmeans.train(feature_matrix)
    logger.info('Trained k means index in {:.3f} s'.format(time.time() - tic))
    logger.info('kmeans.index.is_trained = {}'.format(kmeans.index.is_trained))

    logger.info('Writing artifacts to disk')
    tic = time.time()
    local_index_path = index_path
    logger.info('Writing index to local disk')
    index_cpu = faiss.index_gpu_to_cpu(kmeans.index)
    faiss.write_index(index_cpu, local_index_path)

    local_centroids_path = centroids_path
    logger.info('Writing centroids to local disk')
    np.save(local_centroids_path, kmeans.centroids)

    logger.info('Saving artifacts took {:.3f} s'.format(time.time() - tic))

    logger.info('Done.')


if __name__ == '__main__':
    main()
