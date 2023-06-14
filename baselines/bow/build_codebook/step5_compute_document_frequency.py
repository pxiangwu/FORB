#!/usr/bin/env python3
import os
import pickle
import time

import click
from loguru import logger
import numpy as np


class ComputeDocumentFrequency():

    def get_histograms(self, local_histograms_path: str = 'artifacts/histograms.pkl') -> dict:
        with open(local_histograms_path, 'rb') as fp:
            histogram_dict = pickle.load(fp)
        return histogram_dict

    def document_frequency(self, histogram_dict: dict, n_terms: int):
        document_frequency = np.zeros(n_terms, dtype=np.int32)
        for k, h in histogram_dict.items():
            for term in h:
                document_frequency[term] += 1
        return document_frequency


class ComputeDocumentFrequencyRunner():

    def __init__(self, compute_document_frequency: ComputeDocumentFrequency, histograms_path: str, n_terms: int,
                 document_frequency_path: str, n_docs_output_path: str):
        self.compute_document_frequency = compute_document_frequency
        self.histograms_path = histograms_path
        self.n_terms = n_terms
        self.document_frequency_path = document_frequency_path
        self.n_docs_output_path = n_docs_output_path

    def run(self):
        os.makedirs(os.path.dirname(self.n_docs_output_path), exist_ok=True)

        tic = time.time()
        histogram_dict = self.compute_document_frequency.get_histograms(self.histograms_path)
        logger.info('Loading artifacts took {:.3f} s'.format(time.time() - tic))

        logger.info('Calculating document frequency vector...')
        document_frequency = self.compute_document_frequency.document_frequency(histogram_dict, self.n_terms)
        np.save(self.document_frequency_path, document_frequency)

        n_docs = len(histogram_dict)
        with open(self.n_docs_output_path, 'w') as fp:
            fp.write('{}\n'.format(n_docs))
        logger.info('Done.')


@click.command()
@click.option('--histograms_path', default='artifacts/histograms.pkl', help='Path of the histograms file')
@click.option('--n_terms', type=int, required=True, help='Number of total terms / visual words')
@click.option('--document_frequency_path',
              default='artifacts/document_frequency.npy',
              help='Path of the produced document frequency file')
@click.option('--n_docs_output_path',
              default='artifacts/output-n_docs.txt',
              help='File output containing number of documents processed')
def main(histograms_path: str, n_terms: int, document_frequency_path: str, n_docs_output_path: str):

    compute_document_frequency = ComputeDocumentFrequency()
    runner = ComputeDocumentFrequencyRunner(compute_document_frequency, histograms_path, n_terms, document_frequency_path,
                                            n_docs_output_path)
    runner.run()


if __name__ == '__main__':
    main()
