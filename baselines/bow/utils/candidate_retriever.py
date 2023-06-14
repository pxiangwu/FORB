"""
Classes and functions for performing candidate retrieval / subsetting
"""

import abc
import collections
import json
import pickle
from typing import Dict, List, Set, Tuple, Union

import faiss
from loguru import logger
import numpy as np
from sklearn import preprocessing
import scipy.sparse

USE_MATCHED_WORDS = 'use_matched_words'
USE_ALL = 'use_all'


class RetrievalResult:
    """Represents the result of querying some database"""

    def __init__(self, image_url: str, label: str, features: Dict = None, metadata: Dict = None):
        self.image_url = image_url
        self.label = label
        self.features = features
        self.metadata = metadata


class CandidateRetriever(abc.ABC):
    """Queries a "database" to retrieve the most relevant candidates. """

    def __init__(self, config: Dict[str, str] = None):
        self.config = config
        self.op_name = 'candidate_retrieval'

    @abc.abstractmethod
    def retrieve(self, features: Dict) -> List[RetrievalResult]:
        raise NotImplementedError

    def database_is_empty(self) -> bool:
        return False


def load_codebook_and_df(codebook_path: str, document_frequency_path: str):
    # Load visual words codebook
    codebook = faiss.read_index(codebook_path)
    logger.debug(f'codebook.ntotal = {codebook.ntotal}, codebook.d = {codebook.d}')

    document_frequency = np.load(document_frequency_path)
    return codebook, document_frequency


def load_artifacts(codebook_path: str, document_frequency_path: str, tfidf_matrix_path: str,
                   tfmatrix_matrix_images_path: str, histograms_path: str, inverted_index_path: str, labels_path: str,
                   metadata_path: str):
    codebook, document_frequency = load_codebook_and_df(codebook_path, document_frequency_path)

    # Load top k retrieval artifacts
    db_tfidf_matrix = scipy.sparse.load_npz(tfidf_matrix_path)
    db_tfidf_matrix = preprocessing.normalize(db_tfidf_matrix, copy=True)

    with open(tfmatrix_matrix_images_path, 'r') as fp:
        db_tfidf_image_urls = []
        for line in fp:
            db_tfidf_image_urls.append(line.strip())

    # Load database features artifacts
    with open(histograms_path, 'rb') as fp:
        histograms_dict = pickle.load(fp)
    words_dict = {k: set(v) for k, v in histograms_dict.items()}

    with open(inverted_index_path, 'rb') as fp:
        inverted_dict = pickle.load(fp)

    # Load database metadata
    with open(labels_path, 'r') as fp:
        labels_dict = {}
        for line in fp:
            data = json.loads(line.strip())
            labels_dict[data['image_url']] = data['label']

    with open(metadata_path, 'rb') as fp:
        metadata_dict = pickle.load(fp)

    return codebook, document_frequency, db_tfidf_matrix, db_tfidf_image_urls, words_dict, inverted_dict, labels_dict, metadata_dict


class BagOfWordsTopKRetriever(CandidateRetriever):
    """
    Retrieves the top k results from a visual bag of words database of images.
    A square root normalized histogram of words is used as a query against the
    TF-IDF matrix representing the database of images via cosine similarity.
    """

    def __init__(self,
                 config: Dict[str, str] = None,
                 n_docs: int = None,
                 n_terms: int = None,
                 top_k: int = None,
                 min_intersecting_visual_words: int = None,
                 descriptors_compute_strategy: str = None,
                 codebook: object = None,
                 document_frequency: np.ndarray = None,
                 db_tfidf_matrix: Union[scipy.sparse.csr.csr_matrix, np.ndarray] = None,
                 db_tfidf_image_urls: List[str] = None,
                 words_dict: Dict[str, Set[int]] = None,
                 inverted_dict: Dict[str, Dict] = None,
                 labels_dict: Dict[str, str] = None,
                 metadata_dict: Dict[str, Dict] = None):
        super(BagOfWordsTopKRetriever, self).__init__(config)

        if config is not None:
            # Parse relevant keys from config
            faiss.omp_set_num_threads(int(config.get('FAISS_OMP_THREADS', 20)))
            self.n_docs = int(config.get('N_DOCS'))
            self.n_terms = int(config.get('N_TERMS'))
            self.top_k = int(config.get('TOP_K', 10))
            self.min_intersecting_visual_words = int(config.get('MIN_INTERSECTING_VISUAL_WORDS', 2))
            self.descriptors_compute_strategy = config.get('DESCRIPTORS_COMPUTE_STRATEGY', USE_MATCHED_WORDS)

            if self.descriptors_compute_strategy not in [USE_MATCHED_WORDS, USE_ALL]:
                raise ValueError('Unknown descriptors computing strategy: `{}`'.format(
                    self.descriptors_compute_strategy))

            codebook_path = config.get('CODEBOOK_PATH')
            document_frequency_path = config.get('DOCUMENT_FREQUENCY_PATH')
            tfidf_matrix_path = config.get('TFIDF_MATRIX_PATH')
            tfmatrix_matrix_images_path = config.get('TFIDF_MATRIX_IMAGES_PATH')
            histograms_path = config.get('HISTOGRAMS_PATH')
            inverted_index_path = config.get('INVERTED_INDEX_PATH')
            labels_path = config.get('LABELS_PATH')
            metadata_path = config.get('METADATA_PATH')

            self.codebook, self.document_frequency, self.db_tfidf_matrix, self.db_tfidf_image_urls, \
            self.words_dict, self.inverted_dict, self.labels_dict, self.metadata_dict = load_artifacts(
                codebook_path, document_frequency_path, tfidf_matrix_path, tfmatrix_matrix_images_path, histograms_path,
                inverted_index_path, labels_path, metadata_path)
        else:
            self.n_docs = n_docs
            self.n_terms = n_terms
            self.top_k = top_k
            self.min_intersecting_visual_words = min_intersecting_visual_words
            self.descriptors_compute_strategy = descriptors_compute_strategy
            self.codebook = codebook
            self.document_frequency = document_frequency
            self.db_tfidf_matrix = db_tfidf_matrix
            self.db_tfidf_image_urls = db_tfidf_image_urls
            self.words_dict = words_dict
            self.inverted_dict = inverted_dict
            self.labels_dict = labels_dict
            self.metadata_dict = metadata_dict

    @staticmethod
    def get_histogram_and_indices(
            descriptors: np.ndarray, keypoints: List[Tuple[float, float]], codebook: object
    ) -> Tuple[Dict[int, int], Dict[int, List[Tuple[float, float]]], Dict[int, List[np.ndarray]]]:
        histogram = collections.defaultdict(int)

        visual_word_to_keypoints = collections.defaultdict(list)
        visual_word_to_descriptors = collections.defaultdict(list)
        # Query 100 at at time
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

    @staticmethod
    def sqrt_tfidf_vector(histogram: Dict[int, int], df: np.ndarray, n_terms: int,
                          n_docs: int) -> scipy.sparse.lil.lil_matrix:
        vector = scipy.sparse.lil_matrix((1, n_terms), dtype=np.float32)
        for term in histogram:
            tf = histogram[term]
            idf = np.log(n_docs / (df[term] + 1.0))
            vector[0, term] = np.sqrt(tf * idf)
        return vector

    @staticmethod
    def get_top_k_vector_matches(vector: Union[scipy.sparse.lil.lil_matrix, np.ndarray],
                                 db: Union[scipy.sparse.csr.csr_matrix, np.ndarray], k: int) -> List[int]:
        normalized_vector = preprocessing.normalize(vector, copy=True)
        distances = db @ normalized_vector.T
        ids = np.argsort(distances.toarray().flatten())[-k:][::-1]
        k_distances = distances.toarray().flatten()[ids]
        return ids, k_distances

    @staticmethod
    def get_all_keypoints_and_descriptors(
        kp1: Dict[int, List[Tuple[float, float]]], kp2: Dict[int, List[Tuple[float, float]]],
        d1: Dict[int, List[np.ndarray]], d2: Dict[int, List[np.ndarray]]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], np.ndarray, np.ndarray]:
        num_desc1 = sum(len(val) for val in d1.values())
        num_desc2 = sum(len(val) for val in d2.values())
        keypoints1 = num_desc1 * [None]
        keypoints2 = num_desc2 * [None]
        d1_first_key = next(iter(d1))
        d2_first_key = next(iter(d2))
        desc1_np = np.empty((num_desc1, d1[d1_first_key][0].shape[0]), dtype=d1[d1_first_key][0].dtype)
        desc2_np = np.empty((num_desc2, d2[d2_first_key][0].shape[0]), dtype=d2[d2_first_key][0].dtype)
        idx1 = 0
        idx2 = 0
        for k in d1.keys():
            for i in range(len(d1[k])):
                keypoints1[idx1] = kp1[k][i]
                desc1_np[idx1] = d1[k][i]
                idx1 += 1
        for k in d2.keys():
            for j in range(len(d2[k])):
                keypoints2[idx2] = kp2[k][j]
                desc2_np[idx2] = d2[k][j]
                idx2 += 1
        return keypoints1, keypoints2, desc1_np, desc2_np

    @staticmethod
    def get_matched_words_keypoints_and_descriptors(
        matched_words: Set[int], kp1: Dict[int, List[Tuple[float, float]]], kp2: Dict[int, List[Tuple[float, float]]],
        d1: Dict[int, List[np.ndarray]], d2: Dict[int, List[np.ndarray]]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], np.ndarray, np.ndarray]:
        num_desc1 = sum(len(d1[m]) for m in matched_words)
        num_desc2 = sum(len(d2[m]) for m in matched_words)
        keypoints1 = num_desc1 * [None]
        keypoints2 = num_desc2 * [None]
        first_key = next(iter(matched_words))
        desc1_np = np.empty((num_desc1, d1[first_key][0].shape[0]), dtype=d1[first_key][0].dtype)
        desc2_np = np.empty((num_desc2, d2[first_key][0].shape[0]), dtype=d2[first_key][0].dtype)
        idx1 = 0
        idx2 = 0
        for m in matched_words:
            # Get associated descriptors per matching word for each image
            for i in range(len(d1[m])):
                keypoints1[idx1] = kp1[m][i]
                desc1_np[idx1] = d1[m][i]
                idx1 += 1
            for j in range(len(d2[m])):
                keypoints2[idx2] = kp2[m][j]
                desc2_np[idx2] = d2[m][j]
                idx2 += 1
        return keypoints1, keypoints2, desc1_np, desc2_np

    @staticmethod
    def get_matching_keypoints_and_descriptors(
        matched_words: Set[int],
        kp1: Dict[int, List[Tuple[float, float]]],
        kp2: Dict[int, List[Tuple[float, float]]],
        d1: Dict[int, List[np.ndarray]],
        d2: Dict[int, List[np.ndarray]],
        descriptors_compute_strategy: str = USE_MATCHED_WORDS
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], np.ndarray, np.ndarray]:
        if descriptors_compute_strategy == USE_ALL:
            return BagOfWordsTopKRetriever.get_all_keypoints_and_descriptors(kp1, kp2, d1, d2)
        elif descriptors_compute_strategy == USE_MATCHED_WORDS:
            return BagOfWordsTopKRetriever.get_matched_words_keypoints_and_descriptors(matched_words, kp1, kp2, d1, d2)
        else:
            raise ValueError('Unknown descriptors computing strategy: `{}`'.format(descriptors_compute_strategy))

    def retrieve(self, features: Dict) -> List[RetrievalResult]:
        if features['descriptors'] is None or features['descriptors'].size == 0:
            return []
        query_des = features['descriptors']
        query_kp = features['keypoints']
        query_histogram, query_w2kp, query_w2d = BagOfWordsTopKRetriever.get_histogram_and_indices(
            query_des, query_kp, self.codebook)
        query_vector = BagOfWordsTopKRetriever.sqrt_tfidf_vector(query_histogram, self.document_frequency, self.n_terms,
                                                                 self.n_docs)

        db_ids, db_distances = BagOfWordsTopKRetriever.get_top_k_vector_matches(query_vector, self.db_tfidf_matrix,
                                                                                self.top_k)

        # Filter out results that do not meet minimum intersecting visual words count
        results = []
        query_word_set = set(query_histogram)
        for db_id, db_distance in zip(db_ids, db_distances):
            image_url = self.db_tfidf_image_urls[db_id]

            # Get db keypoint, descriptors, and visual word histogram
            # Use intersecting visual words as matches
            matched_words = query_word_set & self.words_dict[image_url]

            if len(matched_words) < self.min_intersecting_visual_words:
                continue
            db_w2kp = self.inverted_dict[image_url]['keypoints']
            db_w2d = self.inverted_dict[image_url]['descriptors']
            query_kp, db_kp, query_des, db_des = BagOfWordsTopKRetriever.get_matching_keypoints_and_descriptors(
                matched_words, query_w2kp, db_w2kp, query_w2d, db_w2d, self.descriptors_compute_strategy)

            # Append RetrievalResult
            features = {
                'query_kp': query_kp,
                'db_kp': db_kp,
                'query_des': query_des,
                'db_des': db_des,
                'similarity': db_distance
            }
            results.append(
                RetrievalResult(image_url, self.labels_dict[image_url], features, self.metadata_dict[image_url]))

        return results
