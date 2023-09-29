## Introduction

In this folder we provide metadata for `FORB` and its subset (*i.e.*, `FORB Subset`; see the [supplementary material](https://arxiv.org/pdf/2309.16249.pdf) for more details). The metadata contains lists of query and database images for each object type, as well as the ground-truth database images for each query.

All the files in subfolder `FORB_subset` comprise the metadata of `FORB Subset`. The other files in the current folder are for `FORB`.

**Note:** To evaluate on `FORB Subset`, simply copy the files in folder `FORB_subset` to this current directory. In addition, replace the global constant `ID_OFFSET` in `eval_map.py` and `eval_tmap.py` with the following:
```python
ID_OFFSET = {
    'animated_cards': 0,
    'photorealistic_cards': 9588,
    'bookcovers': 9588 + 935,
    'paintings': 9588 + 935 + 7443,
    'currency': 9588 + 935 + 7443 + 317,
    'logos': 9588 + 935 + 7443 + 317 + 1002,
    'packaged_goods': 9588 + 935 + 7443 + 317 + 1002 + 329,
    'movie_posters': 9588 + 935 + 7443 + 317 + 1002 + 329 + 2302,
}
``` 
