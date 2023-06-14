## Introduction

In this folder we offer implementations of different image retrieval methods. To run image retrieval, first install the dependencies listed in `baselines/requirements.txt`. These are extra dependencies in addition to those required by each specific method (see the installation instruction of each method).

Then to launch image retrieval, for methods `bow`, `fire`, and `delg`, please refer to their respective folders. For all the other methods, use the script `run_query_for_top_only_methods.py` to query against the database (with the prerequisite that the image features have been extracted; see the `README` in each of the folder for more details); see the command below:

```
python run_query_for_top_only_methods.py --object_type photorealistic_cards --method clip --database_name all
```
The arguments are:
- `object_type`: the type of object and takes value from [`animated_cards`, `photorealistic_cards`, `bookcovers`, `paintings`, `currency`, `logos`, `packaged_goods`, `movie_posters`].
- `method`: the name of baseline method. We consider the following methods: [`blip`, `blip2`, `bow`, `clip`, `delg`, `diht`, `dino`, `dinov2`, `fire`, `slip`].
- `database_name`: The name of database against which the search is run. Its value can be [`all`, `oods`], where `all` means querying against all the database images and `oods` means only querying against OOD images.

After obtaining the retrieval results for a certain method, run the following commands to compute the accuracies. Note that to successfully run the commands, we need to export the python path first:
```
export PYTHONPATH=${PYTHONPATH}:/path/to/FORB
```

1. Compute mAP:
```
python eval_map.py --data_root_path /path/to/benchmark --method clip --object_type all --key_for_candidate_db_ids candidate_db_ids
```
The arguments are
- `data_root_path`: where the benchmark data is downloaded.
- `method`: the name of baseline method.
- `object_type`: the type of object and takes value from [`all`, `animated_cards`, `photorealistic_cards`, `bookcovers`, `paintings`, `currency`, `logos`, `packaged_goods`, `movie_posters`]. The value of `all` means computing mAP averaged over all object types, whereas the other values mean computing mAP for a certain object.
- `key_for_candidate_db_ids`: The key for retrieving candidate database ids from retrieval results (which are stored in .ndjson file). It takes value from [`candidate_db_ids`, `candidate_db_ids_before_reranking`]. The value `candidate_db_ids_before_reranking` is only for methods `bow` and `delg`, and refers to the results before candidate reranking.

2. Compute t-mAP:
```
python eval_tmap.py --data_root_path /path/to/benchmark --method clip --object_type all --key_for_candidate_db_ids candidate_db_ids --key_for_candidate_db_scores global_scores
```
Here the argument `key_for_candidate_db_scores` refers to the key for retrieving candidate scores from retrieval results. It takes value from [`global_scores`, `global_scores_before_reranking`]. The value `global_scores_before_reranking` is only for methods `bow` and `delg`, and refers to the results before candidate reranking.
