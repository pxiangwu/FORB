import os
import json
import sys
import warnings

import click
import cv2
import tqdm
from loguru import logger

from utils.feature_extractor import RootSift
from utils.result_filter import RansacGeometricVerification
from utils.candidate_retriever import BagOfWordsTopKRetriever
from utils.image_preprocessor import ImageScaler

warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stderr, level="INFO")

SIFT_NUM_FEATURES = 711
N_DOCS = 15000
N_TERMS = 1000000
TOP_K = 100
MIN_INTERSECTING_VISUAL_WORDS = 2


@click.command()
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
@click.option('--object_type', required=True, help='Which object type to evaluate for')
@click.option('--search_artifacts_name',
              type=click.Choice(['all', 'oods']),
              help='The name of folder that contains search artifacts')
@click.option('--n_terms', default=1000000, help='Number of total terms / visual words')
@click.option('--n_docs', default=15000, help='Number of documents used in IDF calculation')
@click.option('--opencv_threads', default=8, help='Number of threads for running opencv')
def main(data_root_path: str, object_type: str, search_artifacts_name: str, n_terms: int, n_docs: int, opencv_threads: int):
    query_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_query.ndjson')
    query_image_info = list(map(json.loads, open(query_image_list_file)))
    query_image_paths = [info['image'] for info in query_image_info]
    query_image_paths = [os.path.join(data_root_path, img) for img in query_image_paths]
    logger.info(f'Number of query images: {len(query_image_paths)}')

    cv2.setNumThreads(opencv_threads)

    config = {
        "SIFT_NUM_FEATURES": "711",
        "FAISS_OMP_THREADS": "30",
        "N_DOCS": n_docs,
        "N_TERMS": n_terms,
        "TOP_K": "10",
        "MIN_INTERSECTING_VISUAL_WORDS": "2",
        "DESCRIPTORS_COMPUTE_STRATEGY": "use_matched_words",
        "ACTIVATE_HOMOGRAPHY_CHECKER": "True",
        "ACTIVATE_BOUNDING_POLYGON_CHECKER": "True",
        "CODEBOOK_PATH": "build_codebook/artifacts/codebook.index",
        "DOCUMENT_FREQUENCY_PATH": "build_codebook/artifacts/document_frequency.npy",
        "TFIDF_MATRIX_PATH": f"build_index/{search_artifacts_name}/tfidf_matrix.npz",
        "TFIDF_MATRIX_IMAGES_PATH": f"build_index/{search_artifacts_name}/tfidf_matrix-image_urls.txt",
        "HISTOGRAMS_PATH": f"build_index/{search_artifacts_name}/histograms.pkl",
        "INVERTED_INDEX_PATH": f"build_index/{search_artifacts_name}/inverted_index.pkl",
        "LABELS_PATH": f"build_index/{search_artifacts_name}/labels.ndjson",
        "METADATA_PATH": f"build_index/{search_artifacts_name}/metadata.pkl",
    }
    retrieval_results_file = f'retrieval_results/{object_type}_{search_artifacts_name}.ndjson'
    os.makedirs(os.path.dirname(retrieval_results_file), exist_ok='True')
    save_fp = open(retrieval_results_file, 'w')

    image_preprocessor = ImageScaler(config)
    feature_extractor = RootSift(config)
    candidate_retriever = BagOfWordsTopKRetriever(config)
    result_filter = RansacGeometricVerification(config)

    query_id = 1
    candidate_scores = []
    candidate_ids = []
    for query_image_path in tqdm.tqdm(query_image_paths):
        image_data = image_preprocessor.process(query_image_path)
        features = feature_extractor.extract(image_data)
        candidates = candidate_retriever.retrieve(features)
        results = result_filter.filter(image_data, candidates)

        result_info = {}
        result_info['query_id'] = query_id
        result_info['candidate_db_ids_before_reranking'] = [int(r.label) - 1 for r in candidates]
        result_info['global_scores_before_reranking'] = [float(r.features['similarity']) for r in candidates]
        result_info['candidate_db_ids'] = [int(r.label) - 1 for r in results]
        result_info['global_scores'] = [float(r.features['similarity']) for r in results]
        result_info['num_inlier'] = [int(r.features['num_inliers']) for r in results]
        save_fp.write(f'{json.dumps(result_info)}\n')

        query_id += 1

        candidate_scores.append(result_info['global_scores'])
        candidate_ids.append(result_info['candidate_db_ids'])

    save_fp.close()


if __name__ == '__main__':
    main()
