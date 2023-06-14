from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import click
import numpy as np
import tqdm
from loguru import logger

from delf import datum_io
from rerank import reranking

# File extensions.
DELG_GLOBAL_EXTENSION = '.delg_global'
DELG_LOCAL_EXTENSION = '.delg_local'

TOP_K = 100
MAX_NUM_CANDIDATES = 10
MIN_INLIER_NUM = 4
DIFFICULTY_LEVELS = ['easy', 'hard', 'medium']
OBJECTS = [
    'animated_cards',
    'photorealistic_cards',
    'bookcovers',
    'paintings',
    'currency',
    'logos',
    'packaged_goods',
    'movie_posters',
]


def read_delg_global_descriptors(input_dir, image_list):
    """Reads DELG global features.

    Args:
        input_dir: Directory where features are located.
        image_list: List of image names for which to load features.

    Returns:
        global_descriptors: NumPy array of shape (len(image_list), D), where D
            corresponds to the global descriptor dimensionality.
    """
    num_images = len(image_list)
    global_descriptors = []
    logger.info(f'Starting to collect global descriptors for {num_images} images...')

    for i in range(num_images):
        descriptor_filename = image_list[i] + DELG_GLOBAL_EXTENSION
        descriptor_fullpath = os.path.join(input_dir, descriptor_filename)
        global_descriptors.append(datum_io.ReadFromFile(descriptor_fullpath))

    logger.info('Done collecting global descriptors.')

    return np.array(global_descriptors)


@click.command()
@click.option('--object_type', required=True, help='Which object type to evaluate for')
@click.option('--database_name', type=click.Choice(['all', 'oods']), help='The name of database against which the search is run')
@click.option('--output_dir', default='retrieval_results', help='Directory where retrieval output will be written to.')
@click.option('--use_geometric_verification',
              is_flag=True,
              help='If True, performs re-ranking using local feature-based geometric verification.')
@click.option('--local_descriptor_matching_threshold',
              default=1.0,
              help='Optional, only used if `use_geometric_verification` is True. Threshold below which a pair of '
              'local descriptors is considered a potential match, and will be fed into RANSAC.')
@click.option('--ransac_residual_threshold',
              default=20.0,
              help='Optional, only used if `use_geometric_verification` is True. Residual error threshold for '
              'considering matches as inliers, used in RANSAC algorithm.')
@click.option('--use_ratio_test',
              is_flag=True,
              help='Optional, only used if `use_geometric_verification` is True. '
              'Whether to use ratio test for local feature matching.')
def main(object_type, database_name, output_dir, use_geometric_verification, local_descriptor_matching_threshold,
         ransac_residual_threshold, use_ratio_test):
    # Parse dataset to obtain query/index images, and ground-truth.
    logger.info('Parsing dataset...')

    # Get the lists of query and database images.
    query_features_dir = os.path.join(f'artifacts_{database_name}', object_type, 'query')
    query_feature_files = os.listdir(query_features_dir)
    query_list = [f.split('.')[0] for f in query_feature_files if f.endswith(DELG_GLOBAL_EXTENSION)]
    query_list = sorted(query_list, key=int)

    database_list = []
    for obj in OBJECTS:
        database_folder_path = os.path.join(f'artifacts_{database_name}', obj, 'database')
        database_feature_files = os.listdir(database_folder_path)
        database_feature_files = [f.split('.')[0] for f in database_feature_files if f.endswith(DELG_GLOBAL_EXTENSION)]
        database_feature_files = sorted(database_feature_files, key=int)
        database_feature_files = [os.path.join(database_folder_path, f) for f in database_feature_files]
        database_list += database_feature_files

    num_query_images = len(query_list)
    num_database_images = len(database_list)
    logger.info(f'Done! Found {num_query_images} queries and {num_database_images} index images')

    # Read global features.
    query_global_features = read_delg_global_descriptors(query_features_dir, query_list)
    database_global_features = read_delg_global_descriptors('', database_list)

    # Compute similarity between query and database images, potentially re-ranking
    # with geometric verification.
    ranks_before_gv = [[] for _ in range(num_query_images)]
    similarities_full_list = [[] for _ in range(num_query_images)]

    result_list = []
    similarities_list = []

    logger.info('Performing retrieval with global features...')
    for i in tqdm.tqdm(range(num_query_images)):
        # Compute similarity between global descriptors.
        similarities = np.dot(database_global_features, query_global_features[i])
        ranks_before_gv[i] = np.argsort(-similarities)
        similarities_full_list[i] = similarities

        ranks_to_save = ranks_before_gv[i][:TOP_K]
        result_list.append((query_list[i], ranks_to_save))  # (query id, top candidates)
        similarities_list.append(similarities[ranks_to_save])

    print('Finished retrieval for all queries (using global features).')

    # Create output directory if necessary.
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'{object_type}_{database_name}.ndjson'), 'w') as fp:
        global_retrievals = []
        for pred, sim in zip(result_list, similarities_list):
            output_dict = {}
            output_dict['query_id'] = int(pred[0])
            output_dict['candidate_db_ids_before_reranking'] = pred[1][:MAX_NUM_CANDIDATES].tolist()
            output_dict['global_scores_before_reranking'] = sim[:MAX_NUM_CANDIDATES].tolist()

            global_retrievals.append(output_dict)
            fp.write(f'{json.dumps(output_dict)}\n')

    logger.info('Re-ranking retrievals with local features...')
    if use_geometric_verification:

        with open(os.path.join(output_dir, f'{object_type}_{database_name}.ndjson'), 'w') as fp:
            for i in tqdm.tqdm(range(len(ranks_before_gv))):
                new_ranks, inliers_and_scores = reranking.RerankByGeometricVerification(
                    input_ranks=ranks_before_gv[i],
                    initial_scores=similarities_full_list[i],
                    query_name=query_list[i],
                    index_names=database_list,
                    query_features_dir=query_features_dir,
                    index_features_dir='',
                    junk_ids=set(),
                    local_feature_extension=DELG_LOCAL_EXTENSION,
                    ransac_seed=0,
                    descriptor_matching_threshold=local_descriptor_matching_threshold,
                    ransac_residual_threshold=ransac_residual_threshold,
                    use_ratio_test=use_ratio_test)

                output_dict = {}
                output_dict['query_id'] = int(query_list[i])
                output_dict['candidate_db_ids'] = new_ranks[:MAX_NUM_CANDIDATES]
                output_dict['global_scores'] = [float(x[1]) for x in inliers_and_scores[:MAX_NUM_CANDIDATES]]
                output_dict['inliers'] = [float(x[0]) for x in inliers_and_scores[:MAX_NUM_CANDIDATES]]
                output_dict['candidate_db_ids_before_reranking'] = global_retrievals[i]['candidate_db_ids_before_reranking']
                output_dict['global_scores_before_reranking'] = global_retrievals[i]['global_scores_before_reranking']

                fp.write(f'{json.dumps(output_dict)}\n')

    logger.info('Done re-ranking with local features.')


if __name__ == '__main__':
    main()
