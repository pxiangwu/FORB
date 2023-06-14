## Introduction

This folder provides code for running FIRe image retrieval method on our benchmark. See this [paper](https://openreview.net/pdf?id=wogsFPHwftY) for more details about FIRe.

To launch image retrieval, follow the steps below.

### 1. Install FIRe

Follow the instructions [here](https://github.com/naver/fire) to install FIRe. Eventually the contents of folder `fire/` should be like this:
```
fire
  |- asmk
  |- cnnimageretrieval-pytorch-1.2
  |- fire
  |- how
  |- weights
  build_codebook.py
  ...
```

### 2. Export Env Variables

Suppose the path to this `fire/` folder is `/baselines/fire/`. Then export the following variables:
```
export PYTHONPATH=${PYTHONPATH}:/baselines/fire/how
export PYTHONPATH=${PYTHONPATH}:/baselines/fire/cnnimageretrieval-pytorch-1.2
export PYTHONPATH=${PYTHONPATH}:/baselines/fire/asmk
export PYTHONPATH=${PYTHONPATH}:/baselines/fire/fire
```

### 3. Build Codebook

Run the following command to build a codebook:
```
python build_codebook.py --data_root_path /path/to/benchmark
```

### 4. Run Query

Run the following command to query against all database images:
```
python run_query.py --object_type animated_cards --data_root_path /path/to/benchmark --search_artifacts_name all
```
where:
- `object_type:`: the type of object and takes value from [`animated_cards`, `photorealistic_cards`, `bookcovers`, `paintings`, `currency`, `logos`, `packaged_goods`, `movie_posters`].
- `search_artifacts_name`: the name of folder that contains search artifacts and takes value from [`all`, `oods`].

Similarly, to query against distractor images:
```
python run_query.py --object_type animated_cards --data_root_path /path/to/benchmark --search_artifacts_name oods
```

### 5. Evaluate Retrieval Accuracy

Use scripts `eval_map.py` and `eval_tmap.py` in parent folder to compute mAP and t-mAP, respectively. See the parent folder for more details.

