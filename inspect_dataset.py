from pycocotools.coco import COCO
import matplotlib.patches as patches

from code import interact
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def show_(coco, img_id, im_path):

    img = coco.imgs[img_id]
    anns = coco.imgToAnns[img_id]

    filename = os.path.join(im_path, img['file_name'])
    img = Image.open(filename)
    np_img = np.asarray(img)

    fig, ax = plt.subplots(1)
    ax.imshow(np_img)
    coco.showAnns(anns)
    plt.axis('off')
    ax = plt.gca()

    for ann in anns:
        print('id: {}, object: {}, area: {:01f}'.format(ann['id'], coco.cats[ann['category_id']]['name'], ann['area']))
        topleft = (ann['bbox'][0], ann['bbox'][1])
        width = ann['bbox'][2]
        height = ann['bbox'][3]
        rect = patches.Rectangle(topleft,
                                 width,
                                 height,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


def instructions():
    print('\n\n\n_____________________\n\n')
    print('::: INSTRUCTIONS :::  \n')
    print('To inspect images and annotations, chose an index.')
    print('and use the show() function (e.g. show(13)).')
    print('\n')


def main(ann_file, im_path):

    plt.ion()
    coco = COCO(ann_file)
    ids = [im['id'] for im in coco.dataset['images']]

    def show(id):
        show_(coco, ids[id], im_path)

    instructions()

    interact(local=locals())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ann_file',
                        help='Path of augmented COCO dataset JSON file',
                        default='/home/mate/data/annotations/augmented_instances_val2017.json')
    parser.add_argument('--image_dir',
                        help='Directory of augmeted images',
                        default='/home/mate/data/val2017_augmented')
    args = parser.parse_args()
    main(ann_file=args.ann_file,
         im_path=args.image_dir)


