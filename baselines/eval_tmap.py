from collections import defaultdict
import json
import math
import os

import click
import numpy as np
import tqdm

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
KS = [1, 5]
FPR_LIST = np.linspace(0, 1.0, num=11, endpoint=True)


def eval_map(gt_list, pred_list):
    outputs = {}
    for k in KS:
        ap = get_map_at_k(gt_list, pred_list, k)
        outputs[f"mAP@{k}"] = ap
    return outputs


def get_mean_map(eval_results):
    for k in KS:
        acc_list = []
        for res in eval_results:
            acc_list.append(res[f"mAP@{k}"])
        print(f"t-mAP@{k}: {np.mean(acc_list):.4f}")


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
@click.option('--key_for_candidate_db_scores',
              default='global_scores',
              type=click.Choice(['global_scores', 'global_scores_before_reranking']),
              help='The key for retrieving candidate scores from retrieval results (which are stored in .ndjson file)')
def main(data_root_path: str, method: str, object_type: str, key_for_candidate_db_ids: str, key_for_candidate_db_scores: str):
    # get thresholds using OOD queries
    pred_score_list = []
    for obj in OBJECT_NAME_MAPPING['all']:
        predictioin_file = os.path.join(method, 'retrieval_results', f'{obj}_oods.ndjson')
        predicts = list(map(json.loads, open(predictioin_file)))
        for pred in predicts:
            scores = pred[key_for_candidate_db_scores]
            if len(scores) == 0:
                pred_score_list.append(-1)  # -1 is a dummy number for taking care of empty list
            else:
                pred_score_list.append(max(scores))

    threshold_list = []
    pred_score_list = sorted(pred_score_list)
    total_num = len(pred_score_list)
    print("total number of OOD queries:", total_num)
    for fpr in FPR_LIST:
        if fpr == 1:
            threshold_list.append(-1)
        else:
            fp_num = math.ceil((1 - fpr) * total_num)
            threshold = pred_score_list[fp_num - 1]
            threshold_list.append(threshold)
    print("Thresholds:", threshold_list)

    # compute mAP at each threshold
    gt_ids = []
    difficulty_levels = []
    for obj in OBJECT_NAME_MAPPING[object_type]:
        gt_file = os.path.join(data_root_path, 'metadata', f'{obj}_gt.ndjson')
        gt_dict = load_groundtruth_file(gt_file)
        gt_dict['groundtruth_ids'] = calibrate_gt_ids(gt_dict['groundtruth_ids'], obj)

        gt_ids += gt_dict['groundtruth_ids']
        difficulty_levels += gt_dict['difficulty_levels']
    query_groups = get_queries_grouped_by_difficulty_levels(difficulty_levels)

    results = defaultdict(list)
    for threshold in tqdm.tqdm(threshold_list):
        candidate_ids = []
        for obj in OBJECT_NAME_MAPPING[object_type]:
            retrieval_results_file = os.path.join(method, 'retrieval_results', f'{obj}_all.ndjson')
            retrieval_results = list(map(json.loads, open(retrieval_results_file)))
            for res in retrieval_results:
                ids = res[key_for_candidate_db_ids]
                scores = res[key_for_candidate_db_scores]
                filtered_ids = []
                for _id, _score in zip(ids, scores):
                    if _score >= threshold:
                        filtered_ids.append(_id)
                candidate_ids.append(filtered_ids)

        results['overall'].append(eval_map(gt_ids, candidate_ids))

        for level in DIFFICULTY_LEVELS:
            query_indices = query_groups[level]
            sub_groundtruth_ids = [gt_ids[q_idx] for q_idx in query_indices]
            sub_candidate_db_ids = [candidate_ids[q_idx] for q_idx in query_indices]

            results[level].append(eval_map(sub_groundtruth_ids, sub_candidate_db_ids))

    print(">> Overall Performance:")
    get_mean_map(results['overall'])

    for level in DIFFICULTY_LEVELS:
        print(f">> Difficulty: {level}")
        get_mean_map(results[level])


if __name__ == '__main__':
    main()
