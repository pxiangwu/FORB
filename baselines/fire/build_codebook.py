import os
import time

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


def asmk_train_codebook(net, preprocess, image_paths, asmk, cache_path=None):
    """train codebook"""
    if cache_path and os.path.exists(cache_path):
        return asmk.train_codebook(None, cache_path=cache_path)

    logger.info(f"Number of training images (for building codebook): {len(image_paths)}")

    logger.info(f"Image size (i.e., maximum size of longer side): {MAX_IMAGE_SIZE}")
    dset = ImagesFromList(image_paths=image_paths, imsize=MAX_IMAGE_SIZE, transform=preprocess)

    des_train, _ = extract_vectors_local(net, dset)
    asmk = asmk.train_codebook(des_train, cache_path=cache_path)
    return asmk


def extract_vectors_local(net, dataset):
    """Return tuple (local descriptors, image ids)"""
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    tik = time.time()
    with torch.no_grad():
        vecs, imids = [], []
        for imid, inp in tqdm.tqdm(enumerate(loader), total=len(loader)):
            output, _, _, _ = net.forward_local(inp.to(DEVICE), features_num=FEATURE_NUM, scales=[1.0])

            vecs.append(output.cpu().numpy())
            imids.append(np.full((output.shape[0],), imid))

    logger.info(f"Average feature extraction time per image: {(time.time() - tik) / len(dataset):.4f} s")
    return np.vstack(vecs), np.hstack(imids)


def make_dirs(file_path):
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


@click.command()
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
@click.option('--codebook_cache_file', default='artifacts/codebook.pkl', help='Path to the file that stores trained words (.pkl)')
def main(data_root_path, codebook_cache_file):
    make_dirs(codebook_cache_file)

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

    codebook_image_list_file = os.path.join(data_root_path, 'metadata', 'images_for_building_codebook.txt')
    with open(codebook_image_list_file) as fp:
        image_paths = fp.readlines()
        image_paths = [os.path.join(data_root_path, l.strip()) for l in image_paths]

    # Build codebook / words
    tik = time.time()
    logger.info("Start building codebook for FIRe + ASMK evaluation")
    asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_parameters)
    asmk = asmk_train_codebook(net=model,
                               preprocess=preprocess,
                               image_paths=image_paths,
                               asmk=asmk,
                               cache_path=codebook_cache_file)

    logger.info(f"Finished building codebook in {int(time.time()-tik) // 60} min")


if __name__ == '__main__':
    main()
