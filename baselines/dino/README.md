## Introduction

This folder provides code for applying DINO image embeddings to image retrieval task on our benchmark. See this [paper](https://arxiv.org/pdf/2104.14294.pdf) for more details about DINO.

To launch image retrieval, follow the steps below.

### 1. Install DINO

Follow the instructions [here](https://github.com/facebookresearch/dino) to install the dependencies of DINO. Note that there is no need to git clone the DINO repository.

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
