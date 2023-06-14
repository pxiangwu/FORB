"""
Extracts DELG features for a set of images.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import click
import tensorflow as tf
import tqdm
from google.protobuf import text_format
from PIL import Image
import numpy as np

from delf import delf_config_pb2
from delf import datum_io
from delf import feature_io
from delf import extractor

# File extensions.
DELG_GLOBAL_EXTENSION = '.delg_global'
DELG_LOCAL_EXTENSION = '.delg_local'

MAX_IMAGE_SIZE = 480

# Prevent Tensorflow from using all the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def extract_global_and_local_feats(config, image_list, output_features_dir, extractor_fn):
    for idx, image_path in enumerate(tqdm.tqdm(image_list)):
        # Compose output file name and decide if image should be skipped.
        should_skip_global = True
        should_skip_local = True
        if config.use_global_features:
            output_global_feature_filename = os.path.join(output_features_dir, str(idx) + DELG_GLOBAL_EXTENSION)
            if not tf.io.gfile.exists(output_global_feature_filename):
                should_skip_global = False

        if config.use_local_features:
            output_local_feature_filename = os.path.join(output_features_dir, str(idx) + DELG_LOCAL_EXTENSION)
            if not tf.io.gfile.exists(output_local_feature_filename):
                should_skip_local = False

        if should_skip_global and should_skip_local:
            continue

        resize_factor = 1.0

        image_data = Image.open(image_path).convert('RGB')
        image_data.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.ANTIALIAS)
        im = np.asarray(image_data)

        # Extract and save features.
        extracted_features = extractor_fn(im, resize_factor)

        if config.use_global_features:
            global_descriptor = extracted_features['global_descriptor']
            datum_io.WriteToFile(global_descriptor, output_global_feature_filename)

        if config.use_local_features:
            locations = extracted_features['local_features']['locations']
            descriptors = extracted_features['local_features']['descriptors']
            feature_scales = extracted_features['local_features']['scales']
            attention = extracted_features['local_features']['attention']
            feature_io.WriteToFile(output_local_feature_filename, locations, feature_scales, descriptors, attention)


@click.command()
@click.option('--object_type', required=True, help='Which object type to evaluate for')
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
@click.option('--database_name', type=click.Choice(['all', 'oods']), help='The name of database against which the search is run')
@click.option('--delf_config_path',
              default='protobuf/r50delg_gld_config.pbtxt',
              help='Path to DelfConfig proto text file with configuration to be used for DELG extraction. '
              'Local features are extracted if use_local_features is True; '
              'global features are extracted if use_global_features is True.')
def main(object_type: str, data_root_path: str, database_name: str, delf_config_path: str):
    output_features_root_path = os.path.join(f'artifacts_{database_name}', object_type)
    output_db_features_dir = os.path.join(output_features_root_path, 'database')
    output_query_features_dir = os.path.join(output_features_root_path, 'query')
    os.makedirs(output_db_features_dir, exist_ok=True)
    os.makedirs(output_query_features_dir, exist_ok=True)

    query_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_query.ndjson')
    query_image_info = list(map(json.loads, open(query_image_list_file)))
    query_image_paths = [info['image'] for info in query_image_info]
    query_image_paths = [os.path.join(data_root_path, l.strip()) for l in query_image_paths]
    print(f'Number of query images: {len(query_image_paths)}')

    database_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_database.ndjson')
    database_image_info = list(map(json.loads, open(database_image_list_file)))
    database_image_paths = [info['image'] for info in database_image_info]
    database_image_paths = [os.path.join(data_root_path, l.strip()) for l in database_image_paths]
    print(f'Number of database images: {len(database_image_paths)}')

    # Parse DelfConfig proto.
    config = delf_config_pb2.DelfConfig()
    with tf.io.gfile.GFile(delf_config_path, 'r') as f:
        text_format.Parse(f.read(), config)

    extractor_fn = extractor.MakeExtractor(config)

    print("Extracting features for database image ...")
    tik = time.time()
    extract_global_and_local_feats(config, database_image_paths, output_db_features_dir, extractor_fn)
    print(f"Finished feature extraction. Time lapsed: {(time.time() - tik) / 60:.2f} minutes")

    print("Extracting features for query image ...")
    tik = time.time()
    extract_global_and_local_feats(config, query_image_paths, output_query_features_dir, extractor_fn)
    print(f"Finished feature extraction. Time lapsed: {(time.time() - tik) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
