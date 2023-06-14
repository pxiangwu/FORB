from typing import Dict, List
import json

import numpy as np


def load_groundtruth_file(groundtruth_file_path: str) -> Dict[str, List]:
    """
    Load groundtruth file corresponding to the queries.

    Argument:
        groundtruth_file_path: the path to the local groundtruth file (not in gcs).
    Output:
        a dictionary, which contains: groundtruth ids, and the difficulty level of the queries.
    """
    gt_records = list(map(json.loads, open(groundtruth_file_path, 'r')))

    groundtruth_ids = []
    difficulty_levels = [] if 'difficulty' in gt_records[0] else None

    for record in gt_records:
        gt_ids = record['groundtruth_ids']
        difficulty = record['difficulty'] if 'difficulty' in record else None

        groundtruth_ids.append(gt_ids)
        if difficulty is not None:
            difficulty_levels.append(difficulty)

    return {'groundtruth_ids': groundtruth_ids, 'difficulty_levels': difficulty_levels}


def get_queries_grouped_by_difficulty_levels(difficulty_level_list: List[str]) -> Dict[str, np.ndarray]:
    """
    Group the queries according to their difficulty levels.

    Argument:
        difficulty_level_list: a list of difficulty levels corresponding to the queries.
    Output:
        a dictionary containing groups of queries with different difficulty levels.
    """
    difficulty_level_array = np.array(difficulty_level_list)

    unique_difficulty_levels = np.unique(difficulty_level_array)

    query_groups = dict()
    for difficulty_level in unique_difficulty_levels:
        query_indices = np.where(difficulty_level_array == difficulty_level)[0]
        query_groups[difficulty_level] = query_indices

    return query_groups
