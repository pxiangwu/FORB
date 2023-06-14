## Introduction

We provide a script for downloading database images from the provided metadata. Specifically, run the following command to download images for a certain object type.

```
python run download_images.py --metadata_file_path metadata/movie_posters_query.ndjson --output_path saved_images
```
where:
- `metadata_file_path`: the path to the metadata file.
- `output_path`: the path for saving the downloaded images.
