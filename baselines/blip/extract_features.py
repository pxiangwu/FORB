import json
import os

import click
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tqdm
import numpy as np
from models.blip import blip_feature_extractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 224
crop_size = 224
model_url = 'weights/model_large.pth'

model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='large')
model.eval()
model = model.to(device)

DB_SCALES = [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25]
QUERY_SCALES = [1.414, 1.0, 0.707]


def preprocess_image(image_size, crop_size, image_path, device):
    raw_image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


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

    if query:
        query_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_query.ndjson')
        query_image_info = list(map(json.loads, open(query_image_list_file)))
        query_image_paths = [info['image'] for info in query_image_info]
        query_image_paths = [os.path.join(data_root_path, l.strip()) for l in query_image_paths]
        print(f'Number of query images: {len(query_image_paths)}')

    if database:
        database_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_database.ndjson')
        database_image_info = list(map(json.loads, open(database_image_list_file)))
        database_image_paths = [info['image'] for info in database_image_info]
        database_image_paths = [os.path.join(data_root_path, l.strip()) for l in database_image_paths]
        print(f'Number of database images: {len(database_image_paths)}')

    # the file names for saved features
    save_database_feat_file = f'features/database_{database_name}.npy'
    save_query_feat_file = f'features/{object_type}/query_{database_name}.npy'

    os.makedirs(os.path.dirname(save_query_feat_file), exist_ok=True)

    # get query image features
    if query:
        print("Extracting features for query images ...")
        query_feats = []
        for image_path in tqdm.tqdm(query_image_paths):
            image_feat_list = []
            for scale in QUERY_SCALES:
                size = int(crop_size * scale)
                image = preprocess_image(size, crop_size, image_path, device)

                with torch.no_grad():
                    image_features = model(image, None, mode='image')[0, 0]
                    image_features = nn.functional.normalize(image_features, dim=0, p=2)
                    image_feat_list.append(image_features)

            multiscale_features = torch.mean(torch.stack(image_feat_list), dim=0)
            multiscale_features = nn.functional.normalize(multiscale_features, dim=0, p=2)

            query_feats.append(multiscale_features.cpu().numpy())

        query_feats = np.stack(query_feats, axis=0)
        np.save(save_query_feat_file, query_feats)
        print("query features shape:", query_feats.shape)

    # get database image features
    if database:
        print("Extracting features for database images ...")
        database_feats = []
        for image_path in tqdm.tqdm(database_image_paths):
            image_feat_list = []
            for scale in DB_SCALES:
                size = int(crop_size * scale)
                image = preprocess_image(size, crop_size, image_path, device)

                with torch.no_grad():
                    image_features = model(image, None, mode='image')[0, 0]
                    image_features = nn.functional.normalize(image_features, dim=0, p=2)
                    image_feat_list.append(image_features)

            multiscale_features = torch.mean(torch.stack(image_feat_list), dim=0)
            multiscale_features = nn.functional.normalize(multiscale_features, dim=0, p=2)

            database_feats.append(multiscale_features.cpu().numpy())

        database_feats = np.stack(database_feats, axis=0)
        np.save(save_database_feat_file, database_feats)
        print("database features shape:", database_feats.shape)


if __name__ == '__main__':
    main()
