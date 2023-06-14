#!/usr/bin/env python3
import os
import pickle
import time
from typing import List, Tuple

import click
from loguru import logger
import numpy as np
import scipy


class CalculateSqrtTfidMatrix:

    def get_histograms(self, local_histograms_path: str = 'artifacts/histograms.pkl') -> dict:
        with open(local_histograms_path, 'rb') as fp:
            histogram_dict = pickle.load(fp)
        return histogram_dict

    def get_document_frequency(self, local_document_frequency_path: str = 'artifacts/document_frequency.npy') -> np.array:
        document_frequency = np.load(local_document_frequency_path)
        return document_frequency

    def sqrt_tfidf_vector(self, histogram: dict, document_frequency: np.array, n_terms: int,
                          n_docs: int) -> scipy.sparse.lil_matrix:
        vector = scipy.sparse.lil_matrix((1, n_terms), dtype=np.float32)
        for term in histogram:
            tf = histogram[term]
            idf = np.log(n_docs / (document_frequency[term] + 1.0))
            vector[0, term] = np.sqrt(tf * idf)
        return vector

    def calculate_tfidf_matrix(self, histogram_dict: dict, document_frequency: np.array, n_terms: int,
                               n_docs: int) -> Tuple[scipy.sparse.lil_matrix, List]:
        sorted_keys = sorted(histogram_dict.keys())
        histogram_count = len(sorted_keys)
        db_tfidf_lil_matrix = scipy.sparse.lil_matrix((histogram_count, n_terms), dtype=np.float32)
        doc_id = 0
        for k in sorted_keys:
            h = histogram_dict[k]
            db_tfidf_lil_matrix[doc_id] = self.sqrt_tfidf_vector(h, document_frequency, n_terms, n_docs)
            doc_id += 1
        return db_tfidf_lil_matrix, sorted_keys


class CalculateSqrtTfidMatrixRunner:

    def __init__(self, calculate_sqrt_tfidf_matrix: CalculateSqrtTfidMatrix, histograms_path: str, n_terms: int, n_docs: int,
                 document_frequency_path: str, tfidf_matrix_output_path: str, image_paths_file_path: str):
        self.calculate_sqrt_tfidf_matrix = calculate_sqrt_tfidf_matrix
        self.histograms_path = histograms_path
        self.n_terms = n_terms
        self.n_docs = n_docs
        self.document_frequency_path = document_frequency_path
        self.tfidf_matrix_output_path = tfidf_matrix_output_path
        self.image_paths_file_path = image_paths_file_path

    def run(self):
        os.makedirs(os.path.dirname(self.tfidf_matrix_output_path), exist_ok=True)

        logger.info('Reading artifacts ...')
        tic = time.time()
        histogram_dict = self.calculate_sqrt_tfidf_matrix.get_histograms(self.histograms_path)
        document_frequency = self.calculate_sqrt_tfidf_matrix.get_document_frequency(self.document_frequency_path)
        logger.info('Loading artifacts took {:.3f} s'.format(time.time() - tic))

        logger.info('Calculating TF-IDF matrix...')
        tic = time.time()
        db_tfidf_lil_matrix, sorted_keys = self.calculate_sqrt_tfidf_matrix.calculate_tfidf_matrix(
            histogram_dict, document_frequency, self.n_terms, self.n_docs)
        logger.info('Calculating TF-IDF matrix took {:.3f} s'.format(time.time() - tic))

        scipy.sparse.save_npz(self.tfidf_matrix_output_path, db_tfidf_lil_matrix.tocsr())

        with open(self.image_paths_file_path, 'w') as fp:
            for k in sorted_keys:
                fp.write('{}\n'.format(k))

        logger.info('Done.')


@click.command()
@click.option('--histograms_path', default='artifacts/histograms.pkl', help='Path of the histograms file')
@click.option('--n_terms', type=int, required=True, help='Number of total terms / visual words')
@click.option('--n_docs', type=int, required=True, help='Number of documents used in IDF calculation')
@click.option('--document_frequency_path', required=True, help='Path of the document frequency file')
@click.option('--tfidf_matrix_output_path', default='artifacts/tfidf_matrix.npz', help='Path of the produced TF-IDF matrix file')
@click.option('--image_paths_file_path',
              default='artifacts/tfidf_matrix-image_urls.txt',
              help='Path of the produced image URLs file')
def main(histograms_path: str, n_terms: int, n_docs: int, document_frequency_path: str, tfidf_matrix_output_path: str,
         image_paths_file_path: str):

    calculate_sqrt_tfidf_matrix = CalculateSqrtTfidMatrix()
    calculate_sqrt_tfidf_matrix_runner = CalculateSqrtTfidMatrixRunner(calculate_sqrt_tfidf_matrix, histograms_path, n_terms,
                                                                       n_docs, document_frequency_path, tfidf_matrix_output_path,
                                                                       image_paths_file_path)
    calculate_sqrt_tfidf_matrix_runner.run()


if __name__ == '__main__':
    main()
