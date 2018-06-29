from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2
from random import randint
from math import cos, sin, radians
import json
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import random
import scipy.ndimage


# parameters
DATASET = 'train2017'
MARGIN = 5  # [px]
ANGLE = 15  # [+/- deg]
SCALE = 20  # [+/- %]
ALLOW_DISJOINT_OBJECTS = False
AREA_MAX = 1024  # [px^2]
AREA_MIN = 0  # [px^2]
BLUR_FILTER_SIZE = 5  # [px]
BLUR_EDGES_RANDOMLY = False
N = 2  # number of pastes
AUGMENT_ONE_OBJECT_PER_IMAGE = False  # if there are more small objects on an image, we only augment one
AUGMENT_OBJECTS_ONCE = True
LOOPED_AUGMENTATION = True

COCO_ROOT = '/home/mate/data'


class SegmentedObject:

    """ Class for segmented object that can be pasted on images. """

    def __init__(self, image_crop, mask, original_annotation, poly):
        self.image = image_crop
        self.mask = mask
        self.poly = poly
        
        # extra info
        self.area = original_annotation['area']
        self.iscrowd = original_annotation['iscrowd']
        self.category_id = original_annotation['category_id']
        self.source_image_id = original_annotation['image_id']
        self.original_annotation = original_annotation


def show_ann_on_image(coco, ann_ids, img=None):
    if img is None:
        ann0 = coco.anns[ann_ids[0]]
        img = coco.imgs[ann0['image_id']]
        np_img = io.imread(img['coco_url'])
    else:
        np_img = np.array(img)
    anns = []
    for ann_id in ann_ids:
        ann = coco.anns[ann_id]
        anns.append(ann)
        print('Object: {}, size: {:01f}'.format(coco.cats[ann['category_id']]['name'], ann['area']))
    plt.imshow(np_img);
    plt.axis('off')
    coco.showAnns(anns)
    plt.show()


def get_pil_image(coco_image):

    """ Given a COCO image dict loading the corresponding image as PIL.Image. """

    augmented_path = '{}/{}_augmented/'.format(COCO_ROOT, DATASET)
    dataset_path = '{}/{}/'.format(COCO_ROOT, DATASET)
    file_name = coco_image['file_name']
    if file_name is not None:
        if os.path.exists(os.path.join(augmented_path, file_name)):
            # if we augmented it already, continue pasting on that image
            img = Image.open(os.path.join(augmented_path, file_name))
        else:
            img = Image.open(os.path.join(dataset_path, file_name))
        return img


def get_object(ann, source_image):

    """ Create a SegmentedObject: create binary mask, get the relevant crops from original and mask. """

    if len(ann['segmentation']) > 1:
        return None
    if ann['area'] < AREA_MIN or ann['area'] > AREA_MAX:
        return None

    polygons = ann['segmentation']
    mask = binary_mask_from_polygons(polygons, source_image.width, source_image.height)
    mask_crop, origin_x, origin_y = crop_bbox_simple(mask, ann['bbox'])
    img_crop = crop_bbox_simple(source_image, ann['bbox'])

    # shift polygon origin
    shifted_polys = []
    for polygon in polygons:
        poly = np.array(polygon)
        poly = np.reshape(polygon, [np.max(poly.shape)/2, 2])
        poly = poly - np.array([origin_x, origin_y])
        shifted_polys.append(poly)

    obj = SegmentedObject(img_crop, mask_crop, ann, shifted_polys)
    return obj


def binary_mask_from_polygons(polygons, width, height):

    """ Creating a binary mask matrix from polygon. """

    rle = mask_util.frPyObjects(polygons, height, width)  # Run-Length Encoding
    decoded = mask_util.decode(rle)
    if decoded.ndim == 3:    # disjoint object, multiple outline polygons on different channels
        decoded = np.amax(decoded, 2)    # flattening by taking max along channels
    mask = np.squeeze(decoded).transpose()  # binary mask
    return mask


def binary_mask_from_polygons2(polygons, width, height):

    """ Alternative implementation of binary mask creation. """

    img = Image.new('L', (height, width), 0)
    for poly in polygons:
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    mask = np.array(img)
    return mask


def crop_bbox_simple(img, bbox):

    """ Cropping rectangular area from original or mask image, with added margins. """

    bbox = [int(i) for i in bbox]

    # calculating actual margin that accomodates rotation
    m = MARGIN + abs(bbox[2] - bbox[3]) / 2
    origin_x = bbox[0] - m
    origin_y = bbox[1] - m

    if type(img) is np.ndarray:
        # creating an empty image of sufficient size, and pasting original image on it
        padded_image = np.zeros([img.shape[0] + 2 * m, img.shape[1] + 2 * m])
        padded_image[m:m + img.shape[0], m:m + img.shape[1]] = img
        # cropping from padded image with margins
        xmin = bbox[0]  # Note: margin is both added and subtracted
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2] + 2 * m
        ymax = bbox[1] + bbox[3] + 2 * m
        cropped_with_margins = padded_image[xmin:xmax, ymin:ymax]
        return np.copy(np.uint8(cropped_with_margins)), origin_x, origin_y
    else:
        # similar calculation as above for PIL.Image
        padded_array = np.zeros([img.height + 2 * m, img.width + 2 * m, 3], dtype=np.uint8)
        padded_image = Image.fromarray(padded_array)
        padded_image.paste(img, box=(m, m))
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2] + 2 * m
        ymax = bbox[1] + bbox[3] + 2 * m
        cropped_with_margins = padded_image.crop(box=(xmin, ymin, xmax, ymax))
        return cropped_with_margins


def paste_object(obj, target_image, occupancy_image, n=N, anns=()):

    """ Paste the extracted SegmentedObject on an image n times, default target is the source image. """

    if obj is None:
        return target_image, anns, occupancy_image

    for i in range(n):
        overlap = True
        paste_trials = 0
        while overlap:
            if paste_trials > 10:
                # don't get stuck if image is already filled
                break
            obj_img = obj.image
            mask_img = Image.fromarray(obj.mask.transpose()*255)

            paste_param = get_paste_parameters(target_image, obj_img)

            # image transformation
            obj_img = obj_img.rotate(paste_param['angle'], resample=Image.BICUBIC, expand=False)
            mask_img = mask_img.rotate(paste_param['angle'], resample=Image.BICUBIC, expand=False)

            new_size = (np.array(obj_img.size) * paste_param['scale']).astype(np.int)
            obj_img = obj_img.resize(new_size, resample=Image.BICUBIC)
            mask_img = mask_img.resize(new_size, resample=Image.BICUBIC)

            overlap, new_occ_img = check_overlap(mask_img, paste_param, occupancy_image)
            if overlap:
                paste_trials += 1
                continue
            else:
                occupancy_image = new_occ_img

            if BLUR_EDGES_RANDOMLY:
                x = random.random()
                if x < 0.33:
                    mask_img = Image.fromarray(cv2.blur(np.array(mask_img), (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE)))
                    target_image.paste(obj_img, box=(paste_param['x'], paste_param['y']), mask=mask_img)
                elif x < 0.66:
                    paste_center = (paste_param['x']+mask_img.width/2, paste_param['y']+mask_img.height/2)
                    dilated = scipy.ndimage.binary_dilation(np.asarray(mask_img), iterations=4)
                    pasted = cv2.seamlessClone(np.asarray(obj_img), np.asarray(target_image),
                                               dilated.astype(np.uint8)*255,
                                               paste_center, cv2.NORMAL_CLONE)
                    target_image = Image.fromarray(pasted)
                else:
                    target_image.paste(obj_img, box=(paste_param['x'], paste_param['y']), mask=mask_img)

            else:
                target_image.paste(obj_img, box=(paste_param['x'], paste_param['y']), mask=mask_img)

            anns.append(create_new_ann(obj, paste_param))

    return target_image, anns, occupancy_image


def get_paste_parameters(target_image, obj_img):

    """ Generating parameters for object placement. """

    angle = randint(-ANGLE, ANGLE)
    scale = randint(100-SCALE, 100+SCALE)/100.0
    # margin is too big at the moment, might calculate better placement parameters from mask
    max_x_pos = max(0, target_image.width - int(scale * obj_img.width + MARGIN))
    max_y_pos = max(0, target_image.height - int(scale * obj_img.height + MARGIN))
    x = randint(0, max_x_pos)
    y = randint(0, max_y_pos)
    return {'x': x, 'y': y, 'angle': angle, 'scale': scale}


def check_overlap(mask_img, paste_param, occupancy_image):

    """ Adding binary mask to binary occupancy image, we have an overlap if any pixel gets higher than one. """

    mask = np.array(mask_img).transpose() / 255.0
    placed_mask = np.zeros(occupancy_image.shape)
    placed_mask[paste_param['x']:paste_param['x'] + mask.shape[0],
                paste_param['y']:paste_param['y'] + mask.shape[1]] = mask
    pasted = occupancy_image + placed_mask
    return np.max(pasted) > 1.0, pasted


def get_occupancy_image(existing_anns, coco_image):

    """ Joining binary masks to an occupancy image. """

    anns = existing_anns
    if len(anns) == 0:
        return np.zeros([coco_image['width'], coco_image['height']])
    masks = np.zeros([coco_image['width'], coco_image['height'], len(anns)])
    for i, ann in enumerate(anns):
        masks[:, :, i] = binary_mask_from_polygons(ann['segmentation'], coco_image['width'], coco_image['height'])

    occupancy_image = np.amax(masks, axis=2)
    return occupancy_image


def create_new_ann(obj, paste_param):

    """ Creating COCO annotation dict for object. """

    source_ann = obj.original_annotation
    transformed_polys = []
    transformed_np_polys = np.empty([0, 2])
    for p in obj.poly:
        poly = transform_polygon(paste_param, p, obj)
        transformed_polys.append(poly.reshape(-1).tolist())
        transformed_np_polys = np.vstack([transformed_np_polys, poly])
    new_ann = dict()
    new_ann.update({'image_id': source_ann['image_id'],
                    'area': source_ann['area']*paste_param['scale']*paste_param['scale'],
                    'iscrowd': source_ann['iscrowd'],
                    'category_id': source_ann['category_id'],
                    'id': int(9e12),
                    'bbox': get_bbox_from_poly(transformed_np_polys),
                    'segmentation': transformed_polys,
                    'augmented': True})

    return new_ann


def transform_polygon(param, poly, obj):

    """ Shift, scale and rotate polygon. """

    a = radians(param['angle'])
    rot = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])
    shift = np.array(obj.mask.shape)/2
    shifted_poly = poly - shift
    rotated_poly = shifted_poly.dot(rot)
    shift_back_poly = rotated_poly + shift
    scaled_poly = shift_back_poly * np.array([param['scale'], param['scale']])
    pasted_poly = scaled_poly + np.array([param['x'], param['y']])
    return pasted_poly


def get_bbox_from_poly(poly):

    """ Get x and y extremes from polygon that has form [[x1, y1], [x2, y2], ...] """

    xmin, ymin = np.min(poly, axis=0)
    xmax, ymax = np.max(poly, axis=0)
    return [xmin, ymin, xmax-xmin, ymax-ymin]


def save_augmented_image(target_image, coco_image):

    """ Save image to dir of augmented images. """

    file_name = coco_image['file_name']
    augmented_path = '{}/{}_augmented/'.format(COCO_ROOT, DATASET)
    out_path = os.path.join(augmented_path, file_name)
    target_image.save(out_path, quality=98)


class ImageWithAnns:

    """ Class for handling COCO image and the corresponding annotations jointly."""

    def __init__(self, coco_image, anns):
        self.image = coco_image
        self.anns = anns


def process_image_looped(image_w_anns):
    target_image = get_pil_image(image_w_anns.image)
    occupancy_image = get_occupancy_image(image_w_anns.anns, image_w_anns.image)

    all_anns = image_w_anns.anns

    paste_count = 0

    try:
        while paste_count < N:

            for ann in image_w_anns.anns:
                if ann['id'] >= 9e12:
                    # this is a pasted object, we don't paste it again
                    break

                # Cutting the object from the image and getting the corresponding annotation information.
                obj = get_object(ann, source_image=target_image)

                # Pasting the cut object back to the image, appending the annotations and updating the occupancy image.
                target_image, all_anns, occupancy_image = paste_object(obj,
                                                                       target_image,
                                                                       occupancy_image,
                                                                       n=1,
                                                                       anns=all_anns)

                if obj is not None:
                    # save_augmented_image(target_image, image_w_anns.image)
                    paste_count += 1

                if paste_count >= N:
                    save_augmented_image(target_image, image_w_anns.image)
                    return all_anns

            # if there are no small objects, then stop without adding image
            if paste_count == 0:
                return []

            # or if we past an object only once, then stop and save
            if AUGMENT_OBJECTS_ONCE:
                save_augmented_image(target_image, image_w_anns.image)
                return all_anns

    except ValueError as e:
        print(e)

    return []

def process_image(image_w_anns):

    """ Augmenting the image with randomly pasted objects. """

    target_image = get_pil_image(image_w_anns.image)
    occupancy_image = get_occupancy_image(image_w_anns.anns, image_w_anns.image)

    all_anns = image_w_anns.anns

    try:
        for ann in image_w_anns.anns:
            if ann['id'] >= 9e12:
                # this is a pasted object, we don't paste it again
                break

            # Cutting the object from the image and getting the corresponding annotation information.
            obj = get_object(ann, source_image=target_image)

            # Pasting the cut object back to the image, appending the annotations and updating the occupancy image.
            target_image, all_anns, occupancy_image = paste_object(obj,
                                                                   target_image,
                                                                   occupancy_image,
                                                                   anns=all_anns)
            if obj is not None:
                save_augmented_image(target_image, image_w_anns.image)
                if AUGMENT_ONE_OBJECT_PER_IMAGE:
                    break
    except ValueError as e:
        print(e)

    # save_augmented_image(target_image, image_w_anns.image) Trying saving only augmented images
    return all_anns


def main():

    # loading original annotations
    dataset_name = DATASET
    ann_file = '{}/annotations/instances_{}.json'.format(COCO_ROOT, dataset_name)
    coco = COCO(ann_file)

    augmented_path = '{}/{}_augmented/'.format(COCO_ROOT, DATASET)
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)

    # pairing images with annotations and creating an array that can be parallel processed
    images_with_annotations = []
    for image in coco.dataset['images']:
        anns = coco.imgToAnns[image['id']]
        images_with_annotations.append(ImageWithAnns(image, anns))

    # augmenting all images with multiple threads.
    augmented_anns_for_images = []
    pool = Pool(16)

    if LOOPED_AUGMENTATION:
        for results in tqdm.tqdm(pool.imap(process_image_looped, images_with_annotations), total=len(images_with_annotations)):
            augmented_anns_for_images.append(results)
    else:
        for results in tqdm.tqdm(pool.imap(process_image, images_with_annotations), total=len(images_with_annotations)):
            augmented_anns_for_images.append(results)




    # overwriting all annotations
    augmented_anns = [ann for per_image in augmented_anns_for_images for ann in per_image]
    coco.dataset['annotations'] = []
    augmented_coco_images = []
    image_ids = set()
    for idx, ann in enumerate(augmented_anns):
        ann['id'] = idx
        coco.dataset['annotations'].append(ann)
        if 'augmented' in ann.keys():
            if ann['image_id'] not in image_ids:
                image_ids.add(ann['image_id'])
                augmented_coco_images.append(coco.imgs[ann['image_id']])

    coco.dataset['images'] = augmented_coco_images

    out_ann_file = '{}/annotations/instances_{}_augmented.json'.format(COCO_ROOT, dataset_name)
    with open(out_ann_file, 'w') as out_file:
        json.dump(coco.dataset, out_file)
    print('Augmentations saved to {}.'.format(out_ann_file))


if __name__ == '__main__':
    main()

