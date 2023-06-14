import json
import os

import click

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
    # create image list and labels for all database images
    all_images = []
    for obj in OBJECTS:
        src = f'{data_root_path}/metadata/{obj}_database.ndjson'
        database_info = list(map(json.loads, open(src)))
        src_images = [info['image'] for info in database_info]
        src_images = [os.path.join(data_root_path, l) for l in src_images]
        all_images += src_images

    print('Number of images:', len(all_images))
    image_list_file = 'all/image_list.txt'
    os.makedirs(os.path.dirname(image_list_file), exist_ok=True)
    with open(image_list_file, 'w') as fp:
        fp.write('\n'.join(all_images))
        fp.write('\n')

    label_file = 'all/labels.ndjson'
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'w') as fp:
        image_id = 1
        for image_path in all_images:
            info = {}
            info['image_url'] = image_path
            info['label'] = str(image_id)
            image_id += 1
            fp.write(f'{json.dumps(info)}\n')

    # create image list and labels for distractor images
    distractor_images = []
    for obj in OBJECTS:
        src = f'{data_root_path}/metadata/{obj}_database.ndjson'
        database_info = list(map(json.loads, open(src)))
        src_images = [info['image'] for info in database_info]
        src_images = [os.path.join(data_root_path, l) for l in src_images]

        gt_file = f'{data_root_path}/metadata/{obj}_gt.ndjson'
        gt_images = []
        gt_info = list(map(json.loads, open(gt_file)))
        for info in gt_info:
            gts = info['groundtruth_images']
            gts = [os.path.basename(gt) for gt in gts]
            gt_images += gts
        gt_images = set(gt_images)

        distractors = []
        for img in src_images:
            if os.path.basename(img) not in gt_images:
                distractors.append(img)
        distractor_images += distractors

    print('Number of images:', len(distractor_images))
    image_list_file = 'oods/image_list.txt'
    os.makedirs(os.path.dirname(image_list_file), exist_ok=True)
    with open(image_list_file, 'w') as fp:
        fp.write('\n'.join(distractor_images))
        fp.write('\n')

    label_file = 'oods/labels.ndjson'
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'w') as fp:
        image_id = 1
        for image_path in distractor_images:
            info = {}
            info['image_url'] = image_path
            info['label'] = str(image_id)
            image_id += 1
            fp.write(f'{json.dumps(info)}\n')


if __name__ == '__main__':
    main()
