import json
import os

import click
import numpy as np
from PIL import Image
import torch
from torch import nn
import tqdm

from diht import model_zoo
from diht import transforms as diht_transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

DB_SCALES = [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25]
QUERY_SCALES = [1.414, 1.0, 0.707]
DIHT_MODEL = 'diht_vitl14_336px'
CROP_SIZE = 336


def _transform(resize_size, crop_size):
    transform = diht_transforms.image_transform(
        image_size=crop_size,
        is_train=False,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        image_resize_res=resize_size,
        interpolation_mode=InterpolationMode.BICUBIC,
    )
    return transform


@click.command()
@click.option('--object_type', required=True, help='Which object type to extract features for')
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
@click.option('--database_name', type=click.Choice(['all', 'oods']), help='The name of database against which the search is run')
@click.option('--query/--no-query', default=False, help='Whether to extract features for query images')
@click.option('--database/--no-database', default=False, help='Whether to extract features for database images')
def main(object_type: str, data_root_path: str, database_name: str, query: bool, database: bool):
    if not query and not database:
        print("Did not specify to extract features for query or database images")
        return

    os.makedirs(f'features/{object_type}/', exist_ok=True)

    if query:
        query_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_query.ndjson')
        query_image_info = list(map(json.loads, open(query_image_list_file)))
        query_image_paths = [info['image'] for info in query_image_info]
        query_image_paths = [os.path.join(data_root_path, l.strip()) for l in query_image_paths]
        print(f'Number of query images: {len(query_image_paths)}')

    if database:
        database_image_list_file = os.path.join(data_root_path, 'metadata', f'{database_name}_database_images.txt')
        with open(database_image_list_file) as fp:
            database_image_paths = fp.readlines()
            database_image_paths = [os.path.join(data_root_path, l.strip()) for l in database_image_paths]
        print(f'Number of database images: {len(database_image_paths)}')

    save_database_feat_file = f'features/database_{database_name}.npy'
    save_query_feat_file = f'features/{object_type}/query_{database_name}.npy'

    # load diht model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, model = model_zoo.load_model(DIHT_MODEL, is_train=False)
    model.to(device)
    print("eval mode:", nn.Module().training)

    db_preprocess_list = []
    for scale in DB_SCALES:
        size = int(CROP_SIZE * scale)
        db_preprocess_list.append(_transform(size, CROP_SIZE))

    query_preprocess_list = []
    for scale in QUERY_SCALES:
        size = int(CROP_SIZE * scale)
        query_preprocess_list.append(_transform(size, CROP_SIZE))

    # get query features
    if query:
        query_feats = []
        for image_path in tqdm.tqdm(query_image_paths):
            image_feat_list = []

            for preprocess in query_preprocess_list:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model(image, text=None)
                    image_features = nn.functional.normalize(image_features, dim=1, p=2)
                    image_feat_list.append(image_features)

            multiscale_features = torch.mean(torch.stack(image_feat_list), dim=0)
            multiscale_features = nn.functional.normalize(multiscale_features, dim=1, p=2)

            query_feats.append(multiscale_features.cpu().numpy())

        query_feats = np.concatenate(query_feats, axis=0)
        np.save(save_query_feat_file, query_feats)
        print("query feature shape:", query_feats.shape)

    # get database features
    if database:
        database_feats = []
        for image_path in tqdm.tqdm(database_image_paths):
            image_feat_list = []

            for preprocess in db_preprocess_list:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model(image, text=None)
                    image_features = nn.functional.normalize(image_features, dim=1, p=2)
                    image_feat_list.append(image_features)

            multiscale_features = torch.mean(torch.stack(image_feat_list), dim=0)
            multiscale_features = nn.functional.normalize(multiscale_features, dim=1, p=2)

            database_feats.append(multiscale_features.cpu().numpy())

        database_feats = np.concatenate(database_feats, axis=0)
        np.save(save_database_feat_file, database_feats)
        print("database feature shape:", database_feats.shape)


if __name__ == '__main__':
    main()
