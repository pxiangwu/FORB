import os
import random

import click

NUM_IMAGES = 15000

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


@click.command()
@click.option('--data_root_path', required=True, type=str, help='The root path to the benchmark data')
def main(data_root_path: str):
    image_list_file = 'artifacts/image_list.txt'
    os.makedirs(os.path.dirname(image_list_file), exist_ok=True)

    all_images_list = []
    for obj in OBJECTS:
        database_path = os.path.join(data_root_path, obj, 'database')
        image_list = os.listdir(database_path)
        image_list = [os.path.join(database_path, f) for f in image_list]
        all_images_list += image_list

    selected_images_list = random.sample(all_images_list, NUM_IMAGES)
    print('number of selected images:', len(selected_images_list))
    print('unique elements num:', len(set(selected_images_list)))

    with open(image_list_file, 'w') as fp:
        fp.write('\n'.join(selected_images_list))
        fp.write('\n')


if __name__ == '__main__':
    main()
