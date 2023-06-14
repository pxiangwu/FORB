import json
import os
import requests
import uuid

import click
import cv2
import numpy as np
import tqdm

TIME_OUT = 50
HEADERS = {
    'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
}


def decode_image(image_data):
    image = np.asarray(bytearray(image_data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

    if image.shape[-1] == 4:
        b, g, r, a = cv2.split(image)
        new_image = cv2.merge((b, g, r))
        zero_pixels = np.where(a == 0)
        new_image[zero_pixels] = 255
    else:
        new_image = image
    return new_image


@click.command()
@click.option('--metadata_file_path',
              required=True,
              help='The path to metadata file that contains lists of query or database images')
@click.option('--output_path', required=True, help='The output path for saving the downloaded images')
def main(metadata_file_path: str, output_path: str):
    image_info = list(map(json.loads, open(metadata_file_path)))
    image_urls = [info['image_url'] for info in image_info]
    image_names = [os.path.basename(info['image']) for info in image_info]
    print(f'Number of images: {len(image_urls)}')

    metadata_file_name = os.path.basename(metadata_file_path)
    split_pos = metadata_file_name.rfind('_')
    object_type = metadata_file_name[:split_pos]
    image_type = metadata_file_name[split_pos + 1:].split('.')[0]

    output_image_path = os.path.join(output_path, object_type, image_type)
    os.makedirs(output_image_path, exist_ok=True)

    for url, name in tqdm.tqdm(zip(image_urls, image_names), total=len(image_urls)):
        try:
            response = requests.get(url, timeout=TIME_OUT, headers=HEADERS)
            image_data = response.content
            image = decode_image(image_data)

            new_file_path = os.path.join(output_image_path, name)
            cv2.imwrite(new_file_path, image)
        except:
            continue


if __name__ == '__main__':
    main()
