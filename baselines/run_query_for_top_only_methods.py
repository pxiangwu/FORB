import os
import json

import click
import numpy as np
import tqdm

DB_SCALES = [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25]
QUERY_SCALES = [1.414, 1.0, 0.707]
DIFFICULTY_LEVELS = ['easy', 'hard', 'medium']


def get_top_k_vector_matches(query_vector: np.ndarray, db: np.ndarray, k: int):
    query_vector = np.atleast_2d(query_vector)
    distances = db @ query_vector.T
    ids = np.argsort(distances.flatten())[-k:][::-1]
    scores = distances.flatten()[ids]
    return ids, scores


@click.command()
@click.option('--object_type', required=True, help='Which object type to evaluate for')
@click.option('--method',
              type=click.Choice(['blip', 'blip2', 'slip', 'clip', 'diht', 'dino', 'dinov2']),
              help='Which top-only method to evaluate for')
@click.option('--database_name', type=click.Choice(['all', 'oods']), help='The name of database against which the search is run')
def main(object_type: str, method: str, database_name: str):
    # load global features
    query_global_features = np.load(f'{method}/features/{object_type}/query_{database_name}.npy')
    database_global_features = np.load(f'{method}/features/database_{database_name}.npy')

    print("query global feature shape:", query_global_features.shape)
    print("database global feature shape:", database_global_features.shape)

    # normalize global features
    database_global_features = database_global_features / np.linalg.norm(database_global_features, axis=1, keepdims=True)
    query_global_features = query_global_features / np.linalg.norm(query_global_features, axis=1, keepdims=True)

    retrieval_results_file = f'{method}/retrieval_results/{object_type}_{database_name}.ndjson'
    os.makedirs(os.path.dirname(retrieval_results_file), exist_ok=True)
    pred_fp = open(retrieval_results_file, 'w')

    query_id = 1
    for idx in tqdm.tqdm(range(query_global_features.shape[0])):
        query_vector = query_global_features[idx]
        ids, scores = get_top_k_vector_matches(query_vector, database_global_features, k=10)
        candidate_id_list, candidate_score_list = ids.tolist(), scores.tolist()
        candidate_score_list = [float(x) for x in candidate_score_list]

        pred_dict = {}
        pred_dict['query_id'] = query_id
        pred_dict['candidate_db_ids'] = candidate_id_list
        pred_dict['global_scores'] = candidate_score_list
        pred_fp.write(f'{json.dumps(pred_dict)}\n')
        query_id += 1

    pred_fp.close()


if __name__ == '__main__':
    main()
