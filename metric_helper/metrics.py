from typing import List

import numpy as np


def get_map_at_k(groundtruths: List[List[int]], candidate_ids: List[List[int]], k: int) -> float:
    """
    This function computes mean average precision at K (mAP@K).

    Argument:
        groundtruths: a list, where each element is also a list containing the groundtruth ids for a given query.
        candidate_ids: a list, where each element is also a list containing the (top N) ordered retrieved ids.
        k: the top k in average precision computation.
    Output:
        mAP@K
    """
    n_row = len(candidate_ids)
    gt_num_list = [len(groundtruths[r]) for r in range(n_row)]
    precision_list = [None for _ in range(n_row)]
    relevance_list = [None for _ in range(n_row)]
    rolling_ap = 0

    for i in range(k):
        for r in range(n_row):
            if len(groundtruths[r]) == 0:
                precision_list[r] = 1 if len(candidate_ids[r]) == 0 else 0
                relevance_list[r] = 1 if len(candidate_ids[r]) == 0 else 0
            else:
                precision_list[r] = len(np.intersect1d(groundtruths[r], candidate_ids[r][:(i + 1)])) / (i + 1)
                relevance_list[r] = int(len(candidate_ids[r]) > i and candidate_ids[r][i] in groundtruths[r])

        rolling_ap += np.array(precision_list) * np.array(relevance_list)  # shape: [n_row,]

    gt_num_list = np.maximum(gt_num_list, rolling_ap)  # to cap mAP within [0, 1]
    gt_num_list = np.maximum(1.0, gt_num_list)  # to avoid division by zero

    rolling_ap = rolling_ap / gt_num_list
    ap = np.average(rolling_ap)

    return ap
