import json
import os

import click

from metric_helper.reader import load_groundtruth_file, get_queries_grouped_by_difficulty_levels
from metric_helper.metrics import get_map_at_k

DIFFICULTY_LEVELS = ['easy', 'hard', 'medium']
ID_OFFSET = {
    'animated_cards': 0,
    'photorealistic_cards': 12529,
    'bookcovers': 12529 + 1005,
    'paintings': 12529 + 1005 + 11209,
    'currency': 12529 + 1005 + 11209 + 1045,
    'logos': 12529 + 1005 + 11209 + 1045 + 1583,
    'packaged_goods': 12529 + 1005 + 11209 + 1045 + 1583 + 709,
    'movie_posters': 12529 + 1005 + 11209 + 1045 + 1583 + 709 + 2858,
}
OBJECT_NAME_MAPPING = {k: [k] for k in ID_OFFSET}
OBJECT_NAME_MAPPING.update({'all': ID_OFFSET.keys()})


def eval_map(gt_list, pred_list):
    ks = [1, 5]
    output_strs = []
    for k in ks:
        ap = get_map_at_k(gt_list, pred_list, k)
        output_strs.append(f"mAP@{k}: {ap:.4f}")
    print('\n'.join(output_strs))


def calibrate_gt_ids(gt_list, obj):
    offset = ID_OFFSET[obj] - 1
    for idx in range(len(gt_list)):
        gts = gt_list[idx]
        gts = [gt + offset for gt in gts]
        gt_list[idx] = gts
    return gt_list


@click.command()
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
@click.option('--method',
              type=click.Choice(['blip', 'blip2', 'slip', 'clip', 'diht', 'dino', 'dinov2']),
              help='Which top-only method to evaluate for')
@click.option('--object_type',
              default='all',
              type=str,
              help='Which object type to evaluate for. If `all`, then evaluate for all objects.')
@click.option('--key_for_candidate_db_ids',
              default='candidate_db_ids',
              type=click.Choice(['candidate_db_ids', 'candidate_db_ids_before_reranking']),
              help='The key for retrieving candidate database ids from retrieval results (which are stored in .ndjson file)')
def main(data_root_path: str, method: str, object_type: str, key_for_candidate_db_ids: str):
    candidate_ids = []
    for obj in OBJECT_NAME_MAPPING[object_type]:
        retrieval_results_file = os.path.join(method, 'retrieval_results', f'{obj}_all.ndjson')
        retrieval_results = list(map(json.loads, open(retrieval_results_file)))
        for res in retrieval_results:
            candidate_ids.append(res[key_for_candidate_db_ids])

    gt_ids = []
    difficulty_levels = []
    for obj in OBJECT_NAME_MAPPING[object_type]:
        gt_file = os.path.join(data_root_path, 'metadata', f'{obj}_gt.ndjson')
        gt_dict = load_groundtruth_file(gt_file)
        gt_dict['groundtruth_ids'] = calibrate_gt_ids(gt_dict['groundtruth_ids'], obj)

        gt_ids += gt_dict['groundtruth_ids']
        difficulty_levels += gt_dict['difficulty_levels']
    query_groups = get_queries_grouped_by_difficulty_levels(difficulty_levels)

    print(">> Overall Performance:")
    eval_map(gt_ids, candidate_ids)

    for level in DIFFICULTY_LEVELS:
        query_indices = query_groups[level]
        sub_groundtruth_ids = [gt_ids[q_idx] for q_idx in query_indices]
        sub_candidate_db_ids = [candidate_ids[q_idx] for q_idx in query_indices]

        print(f">> Difficulty: {level}")
        eval_map(sub_groundtruth_ids, sub_candidate_db_ids)


if __name__ == '__main__':
    main()
