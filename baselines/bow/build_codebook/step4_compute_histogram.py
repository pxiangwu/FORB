#!/usr/bin/env python3
"""
This module calculates the histograms of visual words for the descriptors of a set of images, and produces an inverted index of visual words to various data to make querying easier later.
"""
import collections
import os
import pickle
import time

import click
import faiss
from loguru import logger
import numpy as np


def calculate_histogram_and_inverted_indices(descriptors, keypoints, codebook):
    histogram = collections.defaultdict(int)

    visual_word_to_keypoints = collections.defaultdict(list)
    visual_word_to_descriptors = collections.defaultdict(list)
    # Query 100 at a time, based on https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors#guidelines
    descs = [descriptors[i:i + 100] for i in range(0, len(descriptors), 100)]
    j = 0
    for desc in descs:
        desc_array = np.vstack(desc)
        _, visual_words = codebook.search(desc_array, 1)

        for i in range(len(desc)):
            visual_word = visual_words[i][0]
            histogram[visual_word] += 1
            visual_word_to_keypoints[visual_word].append(keypoints[j + i])
            visual_word_to_descriptors[visual_word].append(descriptors[j + i])
        j += 100

    return histogram, visual_word_to_keypoints, visual_word_to_descriptors


@click.command()
@click.option('--codebook_path',
              default='artifacts/codebook.index',
              help='Path of the visual words codebook file (a faiss index)')
@click.option('--omp_threads', default=30, help='Number of threads to use for OpenMP in Faiss')
@click.option('--keypoints_path', default='artifacts/keypoints.pkl', help='Path of the precalculated keypoints file')
@click.option('--descriptors_path', default='artifacts/descriptors.pkl', help='Path of the precalculated descriptors file')
@click.option('--histograms_path', default='artifacts/histograms.pkl', help='Path of the produced histograms file')
@click.option('--inverted_index_path', default='artifacts/inverted_index.pkl', help='Path of the produced inverted index file')
def main(codebook_path: str, omp_threads: int, keypoints_path: str, descriptors_path: str, histograms_path: str,
         inverted_index_path: str):
    os.makedirs(os.path.dirname(histograms_path), exist_ok=True)
    os.makedirs(os.path.dirname(inverted_index_path), exist_ok=True)

    logger.info('Reading codebook ...')
    tic = time.time()
    local_codebook_file_path = codebook_path
    codebook = faiss.read_index(local_codebook_file_path)
    faiss.omp_set_num_threads(omp_threads)
    codebook.hnsw.efSearch = 64
    logger.info('Loading codebook took {:.3f} s'.format(time.time() - tic))

    logger.info('Reading precalculated RootSIFT artifacts ...')
    tic = time.time()
    local_descriptors_path = descriptors_path
    local_keypoints_path = keypoints_path
    with open(local_descriptors_path, 'rb') as fp:
        descriptor_dict = pickle.load(fp)
    with open(local_keypoints_path, 'rb') as fp:
        keypoint_dict = pickle.load(fp)
    logger.info('Loading RootSIFT artifacts took {:.3f} s'.format(time.time() - tic))

    image_uris = set(descriptor_dict)

    i = 0
    histogram_dict = {}
    inverted_index_dict = {}
    tic = time.time()
    logger.info('Starting to calculate histograms and inverted indices...')
    for k in image_uris:
        if i % 50 == 0:
            logger.info('Processed: {} in {:.2f} seconds'.format(i, time.time() - tic))
        i += 1
        descriptors = descriptor_dict[k]
        keypoints = keypoint_dict[k]
        histogram, visual_word_to_keypoints, visual_word_to_descriptors = calculate_histogram_and_inverted_indices(
            descriptors, keypoints, codebook)
        histogram_dict[k] = histogram
        inverted_index_dict[k] = {'descriptors': visual_word_to_descriptors, 'keypoints': visual_word_to_keypoints}
    del descriptor_dict
    del keypoint_dict

    logger.info(f'len(histogram_dict)={len(histogram_dict)}')
    local_histograms_path = histograms_path
    local_inverted_index_path = inverted_index_path
    with open(local_histograms_path, 'wb') as fp:
        pickle.dump(histogram_dict, fp)
    with open(local_inverted_index_path, 'wb') as fp:
        pickle.dump(inverted_index_dict, fp)
    del histogram_dict
    del inverted_index_dict
    logger.info('Finished serializing histograms and inverted indices - elapsed {} seconds'.format(time.time() - tic))

    logger.info('Done.')


if __name__ == '__main__':
    main()
