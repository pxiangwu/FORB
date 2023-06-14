## Introduction

This folder provides the implementation of Bag of Words (BoW) image retrieval method. See this [paper](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) for more details.

To launch image retrieval, first install the dependencies listed in `requirements.txt`. Then follow the steps below.

### 1. Build Codebook

In folder `build_codebook/`, run the following commands to build a codebook.
```
python step0_image_list_for_building_codebook.py --data_root_path /path/to/benchmark
python step1_rootsift_feature_extraction.py --image_path_list artifacts/image_list.txt --num_features 711 --resize_dim 480
python step2_kmeans.py --feature_matrix_path artifacts/feature_matrix.npy --k 1000000 --n_gpus 4
python step3_build_codebook.py
python step4_compute_histogram.py
python step5_compute_document_frequency.py --n_terms 1000000
```

### 2. Build Search Index

In folder `build_index/`, run the following commmands to generate index files. Specifically, to build index for all the database images, run:
```
python step0_create_image_list_and_labels.py --data_root_path /path/to/benchmark
python step1_rootsift_feature_extraction.py --image_path_list all/image_list.txt --num_features 711 --resize_dim 480
python step2_compute_histogram.py --codebook_path ../build_codebook/artifacts/codebook.index
python step3_calculate_tfidf_matrix.py --document_frequency_path ../build_codebook/artifacts/document_frequency.npy --n_terms 1000000 --n_docs 15000
mv artifacts/* all/
rm -r artifacts
```

Similarly, to build index for distractor images (i.e., out-of-distribution images), run:
```
python step0_create_image_list_and_labels.py --data_root_path /path/to/benchmark
python step1_rootsift_feature_extraction.py --image_path_list oods/image_list.txt --num_features 711 --resize_dim 480
python step2_compute_histogram.py --codebook_path ../build_codebook/artifacts/codebook.index
python step3_calculate_tfidf_matrix.py --document_frequency_path ../build_codebook/artifacts/document_frequency.npy --n_terms 1000000 --n_docs 15000
mv artifacts/ oods/
rm -r artifacts
```

### 3. Query Against Database

To query against all the database images, run:
```
python run_query.py --data_root_path /path/to/benchmark --object_type animated_cards --search_artifacts_name all
```
where:
- `object_type:`: the type of object and takes value from [`animated_cards`, `photorealistic_cards`, `bookcovers`, `paintings`, `currency`, `logos`, `packaged_goods`, `movie_posters`].
- `search_artifacts_name`: the name of folder that contains search artifacts and takes value from [`all`, `oods`].

Similarly, to query against distractor images, run:
```
python run_query.py --data_root_path /path/to/benchmark --object_type animated_cards --search_artifacts_name oods
```


### 4. Evaluate Retrieval Accuracy

Use scripts `eval_map.py` and `eval_tmap.py` in parent folder to compute mAP and t-mAP, respectively. See the parent folder for more details.


