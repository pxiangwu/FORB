## Introduction

This folder provides code for running DELG image retrieval method on our benchmark. See this [paper](https://arxiv.org/pdf/2001.05027.pdf) for more details about DELG.

To launch image retrieval, follow the steps below.

### 1. Install DELG

Follow the instructions [here](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md) to install DELG. Eventually the contents of folder `delg/` should be like:
```
delg
  |- parameters
  |- protobuf
  extract_features.py
  ...
```

### 2. Extract Global and Local Image Features

Run the command below to extract image features:
```
python extract_features.py --object_type photorealistic_cards --data_root_path /path/to/benchmark --database_name all
```
where:
- `object_type:`: the type of object and takes value from [`animated_cards`, `photorealistic_cards`, `bookcovers`, `paintings`, `currency`, `logos`, `packaged_goods`, `movie_posters`].
- `database_name`: the name of database against which the search is run; it takes value from [`all`, `oods`].

### 3. Run Query

Run the following command to query against all database images:
```
python run_query.py --object_type photorealistic_cards --database_name all --use_geometric_verification --use_ratio_test
```

Similarly, to query against distractor images:
```
python run_query.py --object_type photorealistic_cards --database_name oods --use_geometric_verification --use_ratio_test
```

### 4. Evaluate Retrieval Accuracy

Use scripts `eval_map.py` and `eval_tmap.py` in parent folder to compute mAP and t-mAP, respectively. See the parent folder for more details.
