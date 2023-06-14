import os
import time
import json

import torch
from torchvision import transforms
import click
import numpy as np
import tqdm
import yaml
from loguru import logger

import fire.fire_network as fire_network
from dataset import ImagesFromList
from asmk import asmk_method

PRETRAINED_WEIGHTS_PATH = 'weights/fire.pth'
ASMK_CONFIG_PATH = 'fire/_asmk_how_fire.yml'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8  # number of workers for loading image data
MAX_IMAGE_SIZE = 480  # maximum size of longer image side
FEATURE_NUM = 1000  # Number of local features
DATABASE_IMAGE_SCALES = [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25]  # scales for multi-scale inference (database images)
QUERY_IMAGE_SCALES = [1.414, 1.0, 0.707]  # scales for multi-scale inference (query images)
OBJECTS = [
    'animated_cards',
    'photorealistic_cards',
    'bookcovers',
    'paintings',
    'currency',
    'logos',
    'packaged_goods',
    'movie_posters',
]


def asmk_index_database(net, preprocess, asmk, dbimages, scales, cache_path=None):
    """Asmk evaluation step 'aggregate_database' and 'build_ivf'"""
    # Index database vectors
    if cache_path and os.path.exists(cache_path):
        asmk_dataset = asmk.build_ivf(None, None, cache_path=cache_path)
    else:
        dset = ImagesFromList(image_paths=dbimages, imsize=MAX_IMAGE_SIZE, transform=preprocess)
        vecs, imids = extract_vectors_local(net, dset, scales)
        asmk_dataset = asmk.build_ivf(vecs, imids, cache_path=cache_path)
    return asmk_dataset


def asmk_query_ivf(net, preprocess, asmk_dataset, qimages, cache_path, scales, imid_offset=0):
    """Asmk evaluation step 'query_ivf'"""
    # Query vectors
    qdset = ImagesFromList(image_paths=qimages, imsize=MAX_IMAGE_SIZE, transform=preprocess)
    qvecs, qimids = extract_vectors_local(net, qdset, scales)
    qimids += imid_offset
    _, query_ids, ranks, scores = asmk_dataset.query_ivf(qvecs, qimids)

    with open(cache_path, "w") as handle:
        for idx, query_id in enumerate(query_ids):
            pred_dict = {}
            pred_dict['query_id'] = int(query_id)
            pred_dict['candidate_db_ids'] = ranks[idx, :100].tolist()
            pred_dict['global_scores'] = scores[idx, :100].tolist()
            handle.write(f'{json.dumps(pred_dict)}\n')


def extract_vectors_local(net, dataset, scales):
    """Return tuple (local descriptors, image ids)"""
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    tik = time.time()
    with torch.no_grad():
        vecs, imids = [], []
        for imid, inp in tqdm.tqdm(enumerate(loader), total=len(loader)):
            output, _, _, _ = net.forward_local(inp.to(DEVICE), features_num=FEATURE_NUM, scales=scales)

            vecs.append(output.cpu().numpy())
            imids.append(np.full((output.shape[0],), imid))

    logger.info(f"Average feature extraction time per image: {(time.time() - tik) / len(dataset):.4f} s")
    return np.vstack(vecs), np.hstack(imids)


def make_dirs(file_path):
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


@click.command()
@click.option('--object_type', required=True, help='Which object type to evaluate for')
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
@click.option('--codebook_path', default='artifacts/codebook.pkl', help='Path to the built codebook file')
@click.option('--search_artifacts_name',
              type=click.Choice(['all', 'oods']),
              help='The name of folder that contains search artifacts')
def main(object_type: str, data_root_path: str, codebook_path: str, search_artifacts_name: str):
    ivf_cache_file = f'artifacts/{search_artifacts_name}/ivf.pkl'
    prediction_file = f'retrieval_results/{object_type}_{search_artifacts_name}.ndjson'
    os.makedirs(os.path.dirname(ivf_cache_file), exist_ok=True)
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)

    # Build model
    state = torch.load(PRETRAINED_WEIGHTS_PATH, map_location='cpu')
    state['net_params']['pretrained'] = None  # no need for imagenet pretrained model
    print(state['net_params'])
    model = fire_network.init_network(**state['net_params']).to(DEVICE)
    model.load_state_dict(state['state_dict'])
    model.eval()
    print(f'training: {model.training}')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with open(ASMK_CONFIG_PATH, 'r') as handle:
        asmk_parameters = yaml.safe_load(handle)

    # Load codebook / words
    tik = time.time()
    logger.info("Loading codebook for ASMK evaluation")
    asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_parameters)
    asmk = asmk.train_codebook(None, cache_path=codebook_path)
    logger.info(f"Finished loading codebook in {time.time() - tik} seconds")

    # Compute ASMK features for database images; then run queries
    query_image_list_file = os.path.join(data_root_path, 'metadata', f'{object_type}_query.ndjson')
    query_image_info = list(map(json.loads, open(query_image_list_file)))
    query_image_paths = [info['image'] for info in query_image_info]
    query_image_paths = [os.path.join(data_root_path, l.strip()) for l in query_image_paths]
    print(f'Number of query images: {len(query_image_paths)}')

    database_image_list_file = os.path.join(data_root_path, 'metadata', f'{search_artifacts_name}_database_images.txt')
    with open(database_image_list_file) as fp:
        database_image_paths = fp.readlines()
        database_image_paths = [os.path.join(data_root_path, l.strip()) for l in database_image_paths]
    print(f'Number of database images: {len(database_image_paths)}')

    asmk_dataset = asmk_index_database(net=model,
                                       preprocess=preprocess,
                                       asmk=asmk,
                                       dbimages=database_image_paths,
                                       scales=DATABASE_IMAGE_SCALES,
                                       cache_path=ivf_cache_file)
    asmk_query_ivf(net=model,
                   preprocess=preprocess,
                   asmk_dataset=asmk_dataset,
                   qimages=query_image_paths,
                   scales=QUERY_IMAGE_SCALES,
                   cache_path=prediction_file)

    logger.info(f"Finished retrieval in {int(time.time()-tik) // 60} min")


if __name__ == '__main__':
    main()
