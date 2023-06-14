## Introduction

This folder provides code for applying DIHT image embeddings to image retrieval task on our benchmark. See this [paper](https://arxiv.org/pdf/2301.02280.pdf) for more details about DIHT.

To launch image retrieval, follow the steps below.

### 1. Install DIHT

Following the instructions [here](https://github.com/facebookresearch/diht) to install dependencies of DiHT. Git clone DiHT inside folder `diht` and eventually this folder contains:
```
diht
  |- diht
    |- diht
    |- tests
    ...
  extract_features.py
  README.md
```

### 2. Extract Image Features

Run the following command:
```
python extract_features.py --object_type photorealistic_cards --data_root_path /path/to/benchmark --database_name all --query --database
```
where:
- `query`: a flag indicates whether to extract features for query images. 
- `database`: a flag indicates whether to extract features for database images. 
- `object_type:`: the type of object and takes value from [`animated_cards`, `photorealistic_cards`, `bookcovers`, `paintings`, `currency`, `logos`, `packaged_goods`, `movie_posters`].
- `database_name`: the name of database against which the search is run; it takes value from [`all`, `oods`].

### 3. Run Query

Use the script `run_query_for_top_only_methods.py` in parent folder to run queries.

### 4. Evaluate Retrieval Accuracy

Use scripts `eval_map.py` and `eval_tmap.py` in parent folder to compute mAP and t-mAP, respectively. See the parent folder for more details.
