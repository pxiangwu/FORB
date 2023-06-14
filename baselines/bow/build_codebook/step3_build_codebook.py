#!/usr/bin/env python3
"""
This module trains an approximate nearest neighbor search faiss index for finding the closest centroid (i.e. visual word) to a given SIFT descriptor.

An HNSW index with scalar quantizer is trained, based on:
- https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors#hnsw-benchmarks
- https://github.com/facebookresearch/faiss/blob/5a187120a56a84038ec74ebacd9d34c93fa76972/benchs/bench_hnsw.py#L71
"""

import os
import time

import click
import faiss
from loguru import logger
import numpy as np

# Targeting fastest setup that achieves R@1 >= 0.9 on 1M sift vectors, which is hnsw_sq @ efSearch 64
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py
HNSW_EFCONSTRUCTION = 40
HNSW_EFSEARCH = 64


@click.command()
@click.option('--centroids_path',
              default='artifacts/centroids.npy',
              help='Path to feature matrix to visual words centroids numpy file, matrix of shape (# centroids, 128)')
@click.option('--codebook_path', default='artifacts/codebook.index', help='Path of the produced code bookfile (a faiss index)')
def main(centroids_path: str, codebook_path: str):
    codebook_output_path = codebook_path
    os.makedirs(os.path.dirname(codebook_output_path), exist_ok=True)

    logger.info('Read centroids file ...')
    tic = time.time()
    local_centroids_file_path = centroids_path
    centroids = np.load(local_centroids_file_path)
    logger.info('Loading centroids took {:.3f} s - shape: {}'.format(time.time() - tic, centroids.shape))

    logger.info('Start training...')
    tic = time.time()
    # Based on https://github.com/facebookresearch/faiss/blob/5a187120a56a84038ec74ebacd9d34c93fa76972/benchs/bench_hnsw.py#L71
    codebook = faiss.IndexHNSWSQ(centroids.shape[1], faiss.ScalarQuantizer.QT_8bit, 16)
    codebook.train(centroids)
    logger.info('Trained HNSW index in {:.3f} s'.format(time.time() - tic))
    logger.info('codebook.is_trained = {}'.format(codebook.is_trained))

    logger.info('Adding vectors to index...')
    tic = time.time()
    codebook.hnsw.efConstruction = HNSW_EFCONSTRUCTION
    codebook.hnsw.efSearch = HNSW_EFSEARCH
    codebook.verbose = True
    codebook.add(centroids)
    logger.info('Index constructed in {:.3f} s'.format(time.time() - tic))

    logger.info('Writing artifacts to disk...')
    tic = time.time()
    local_codebook_path = codebook_path
    logger.info('Writing codebook to local disk...')
    faiss.write_index(codebook, local_codebook_path)
    logger.info('Saving artifacts took {:.3f} s'.format(time.time() - tic))

    logger.info('Done.')


if __name__ == '__main__':
    main()
