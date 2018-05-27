from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from PIL import Image
import os
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2
from code import interact
from random import randint
from math import cos, sin, radians
from datetime import datetime
import json

# parameters
MARGIN = 5  # [px]
ANGLE = 15  # [+/- deg]
SCALE = 20  # [+/- %]
ALLOW_DISJOINT_OBJECTS = False
AREA_MAX = 1024  # [px^2]
AREA_MIN = 0  # [px^2]
BLUR_FILTER_SIZE = 5  # [px]
BLUR_EDGES = False
N = 3  # number of pastes
AUGMENT_ONE_OBJECT_PER_IMAGE = False  # if there are more small objects on an image, we only augment one


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


class DatasetAugmenter:
    def __init__(self, dataset):
        self.ann_file = './annotations/instances_{}.json'.format(dataset)
        self.augmented_ann_file = './annotations/instances_{}_augmented.json'.format(dataset)
        
        self.dataset_path = './{}/'.format(dataset)
        self.augmented_path = './{}_augmented/'.format(dataset)
        if not os.path.exists(self.augmented_path):
            os.makedirs(self.augmented_path)

        self.coco = COCO(self.ann_file)
        self.object_id = int(9e12)
        self.pasted_id_list = []


    def get_pil_image(self, ann_id=None, image_id=None):

        """ Given an annotation or image id loading the corresponding image as PIL.Image. """

        file_name = None
        if ann_id is not None:
            ann = self.coco.anns[ann_id]
            file_name = self.coco.imgs[ann['image_id']]['file_name']
        elif image_id is not None:
            file_name = self.coco.imgs[image_id]['file_name']
        if file_name is not None:
            if os.path.exists(os.path.join(self.augmented_path, file_name)):
                # if we augmented it already, continue pasting on that image
                img = Image.open(os.path.join(self.augmented_path, file_name))
            else:
                img = Image.open(os.path.join(self.dataset_path, file_name))
            return img


    def crop_bbox_simple(self, img, bbox, margin=0, padding_for_rotation=False):

        """ Cropping rectangular area from original or mask image, with added margins. """
        
        bbox = [int(i) for i in bbox]

        # calculating actual margin that accomodates rotation
        m = MARGIN + abs(bbox[2]-bbox[3]) / 2
        origin_x = bbox[0] - m
        origin_y = bbox[1] - m

        if type(img) is np.ndarray:
            # creating an empty image of sufficient size, and pasting original image on it
            padded_image = np.zeros([img.shape[0]+2*m, img.shape[1]+2*m])
            padded_image[m:m+img.shape[0], m:m+img.shape[1]] = img
            # cropping from padded image with margins
            xmin = bbox[0] # Note: margin is both added and subtracted
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2] + 2*m
            ymax = bbox[1] + bbox[3] + 2*m
            cropped_with_margins = padded_image[xmin:xmax,ymin:ymax]
            return np.copy(np.uint8(cropped_with_margins)), origin_x, origin_y
        else:
            # similar calculation as above for PIL.Image
            padded_array = np.zeros([img.height+2*m, img.width+2*m, 3], dtype=np.uint8)
            padded_image = Image.fromarray(padded_array)
            padded_image.paste(img, box=(m, m))
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2] + 2*m
            ymax = bbox[1] + bbox[3] + 2*m
            cropped_with_margins = padded_image.crop(box=(xmin, ymin, xmax, ymax))
            return cropped_with_margins

    def binary_mask_from_polygons(self, polygons, width, height):
        rle = mask_util.frPyObjects(polygons, height, width)  # Run-Length Encoding
        decoded = mask_util.decode(rle)
        if decoded.ndim == 3:    # disjoint object, multiple outline polygons on different channels
            decoded = np.amax(decoded, 2)    # flattening by taking max along channels
        mask = np.squeeze(decoded).transpose()  # binary mask
        return mask
    
    def get_object(self, ann_id, image_id=None):

        ''' Create a SegmentedObejct: create binary mask, get the relevant crops from original and mask. '''

        ann = self.coco.anns[ann_id]

        if len(ann['segmentation']) > 1:
            return None
        if ann['area'] < AREA_MIN or ann['area'] > AREA_MAX:
            return None

        if image_id is None:
            image_id = ann['image_id']

        img = self.get_pil_image(image_id=image_id)


        polygons = ann['segmentation']
        mask = self.binary_mask_from_polygons(polygons, img.width, img.height)
        mask_crop, origin_x, origin_y = self.crop_bbox_simple(mask, ann['bbox'], margin=MARGIN)
        img_crop = self.crop_bbox_simple(img, ann['bbox'], margin=MARGIN)

        # shift polygon origin
        shifted_polys = []
        for polygon in polygons:
            poly = np.array(polygon)
            poly = np.reshape(polygon, [np.max(poly.shape)/2, 2])
            poly = poly - np.array([origin_x, origin_y])
            shifted_polys.append(poly)

        obj = SegmentedObject(img_crop, mask_crop, ann, shifted_polys)
        return obj

    def show_ann_on_image(self, ann_ids, img=None):

        ''' Display object mask on image. '''
        if img is None:
            ann0 = self.coco.anns[ann_ids[0]]
            img = self.coco.imgs[ann0['image_id']]
            I = io.imread(img['coco_url'])
        else:
            I = np.array(img)
        anns = []
        for ann_id in ann_ids:
            ann = self.coco.anns[ann_id]
            anns.append(ann)
            print('Object: {}, size: {:01f}'.format(self.coco.cats[ann['category_id']]['name'], ann['area']))
            
        plt.imshow(I); plt.axis('off')
        self.coco.showAnns(anns)
        plt.show()

    def get_paste_parameters(self, target_image, obj_img):
        angle = randint(-ANGLE, ANGLE)
        scale = randint(100-SCALE, 100+SCALE)/100.0
        # margin is too big at the moment, might calculate better placement parameters from mask
        max_x_pos = max(0, target_image.width - int(scale * obj_img.width + MARGIN))
        max_y_pos = max(0, target_image.height - int(scale * obj_img.height + MARGIN))
        x = randint(0, max_x_pos)
        y = randint(0, max_y_pos)
        return {'x': x, 'y': y, 'angle': angle, 'scale': scale}

    def create_new_ann(self, obj, target_image, paste_param):
        source_ann = obj.original_annotation
        transformed_polys = []
        transformed_np_polys = np.empty([0, 2])
        for p in obj.poly:
            poly = self.transform_polygon(paste_param, p, obj)
            transformed_polys.append(poly.reshape(-1).tolist())
            transformed_np_polys = np.vstack([transformed_np_polys, poly])
        new_ann = dict()
        new_ann.update({'image_id': source_ann['image_id'],
                        'area': source_ann['area']*paste_param['scale']*paste_param['scale'],
                        'iscrowd': source_ann['iscrowd'],
                        'category_id': source_ann['category_id'],
                        'id': self.object_id,
                        'bbox': self.get_bbox_from_poly(transformed_np_polys),
                        'segmentation': transformed_polys})
        self.pasted_id_list.append(self.object_id)
        self.object_id += 1

        # update COCO dataset with new annotation
        self.coco.dataset['annotations'].append(new_ann)
        self.coco.anns[new_ann['id']] = new_ann
        self.coco.imgToAnns[new_ann['image_id']].append(new_ann)
        return new_ann

    def transform_polygon(self, param, poly, obj):

        """ Shift, scale and rotate polygon. """

        a = radians(param['angle'])
        rot = np.array([[cos(a), -sin(a)],[sin(a), cos(a)]])
        shift = np.array(obj.mask.shape)/2
        shifted_poly = poly - shift
        rotated_poly = shifted_poly.dot(rot)
        shift_back_poly = rotated_poly + shift
        scaled_poly = shift_back_poly * np.array([param['scale'], param['scale']])
        pasted_poly = scaled_poly + np.array([param['x'], param['y']])
        return pasted_poly

    def get_bbox_from_poly(self, poly):

        """ Get x and y extremes from polygon that has form [[x1, y1], [x2, y2], ...] """

        xmin, ymin = np.min(poly, axis=0)
        xmax, ymax = np.max(poly, axis=0)
        return [xmin, ymin, xmax, ymax]

    def paste_object(self, obj, n=N, target_image=None):

        ''' Paste the extracted SegmentedObject on an image n times, default target is the source image. '''

        if obj is None:
            return

        if target_image is None:
            target_image = self.get_pil_image(image_id=obj.source_image_id)

        anns = []
        for i in range(n):
            overlap = True
            paste_trials = 0
            while overlap:
                if paste_trials > 10:
                    break
                obj_img = obj.image
                mask_img = Image.fromarray(obj.mask.transpose()*255)

                paste_param = self.get_paste_parameters(target_image, obj_img)

                # image transformation
                obj_img = obj_img.rotate(paste_param['angle'], resample=Image.BICUBIC, expand=False)
                mask_img = mask_img.rotate(paste_param['angle'], resample=Image.BICUBIC, expand=False)

                new_size = (np.array(obj_img.size) * paste_param['scale']).astype(np.int)
                obj_img = obj_img.resize(new_size, resample=Image.BICUBIC)
                mask_img = mask_img.resize(new_size, resample=Image.BICUBIC)

                overlap = self.check_overlap(obj.source_image_id, mask_img, paste_param)
                if overlap:
                    paste_trials += 1
                    continue

                if BLUR_EDGES:
                    mask_img = Image.fromarray(cv2.blur(np.array(mask_img), (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE)))

                target_image.paste(obj_img, box=(paste_param['x'], paste_param['y']), mask=mask_img)
                ann = self.create_new_ann(obj, target_image, paste_param)
            #target_image.show()

        self.save_augmented_image(target_image, obj.source_image_id)

        # target_image.save('debug_img.jpg')
        # self.show_ann_on_image(self.pasted_id_list, img=target_image)

    def get_occupancy_image(self, image_id):
        anns = self.coco.imgToAnns[image_id]
        image = self.coco.imgs[image_id]
        occupancy_image = np.zeros([image['width'], image['height']])

        for ann in anns:
            mask = self.binary_mask_from_polygons(ann['segmentation'], image['width'], image['height'])
            occupancy_image = np.amax(np.stack([mask, occupancy_image], axis=2), axis=2)

        return occupancy_image

    def check_overlap(self, image_id, mask_img, paste_param):
        occupancy_image = self.get_occupancy_image(image_id)
        mask = np.array(mask_img).transpose()/255.0
        placed_mask = np.zeros(occupancy_image.shape)
        placed_mask[paste_param['x']:paste_param['x']+mask.shape[0],
                    paste_param['y']:paste_param['y']+mask.shape[1]] = mask

        pasted = occupancy_image + placed_mask

        img = Image.fromarray(np.uint8(pasted.transpose()*100))
        #img.show()

        return np.max(pasted) > 1.0

    def save_augmented_image(self, target_image, image_id):

        """ Save image to dir of augmented images. """
        file_name = self.coco.imgs[image_id]['file_name']
        out_path = os.path.join(self.augmented_path, file_name)
        target_image.save(out_path, quality=98)

    def save_augmented_dataset(self):
        with open(self.augmented_ann_file, 'w') as output_file:
            json.dump(self.coco.dataset, output_file)

    def process_dataset(self):
        # loop over all images in dataset
        start = datetime.now()
        num_images = len(self.coco.dataset['images'])
        obj_counter = 0
        for idx, coco_image in enumerate(self.coco.dataset['images']):
            image_got_augmented = False
            target_image = self.get_pil_image(image_id=coco_image['id'])
            # loop over annotations for image
            if idx % 100 == 0:
                now = datetime.now()
                print('[{}:{}:{}] Processed {}/{} \t\t number of pastes: {}'.format(now.hour, now.minute, now.second, num_images, idx, obj_counter*N))
            try:
                for ann in self.coco.imgToAnns[coco_image['id']]:
                    if ann['id'] >= 9e12:
                        break
                    obj = self.get_object(ann_id=ann['id'])
                    self.paste_object(obj, target_image=target_image)
                    if obj is not None:
                        obj_counter += 1
                        image_got_augmented = True
                    if obj is not None and AUGMENT_ONE_OBJECT_PER_IMAGE:
                        break
            except ValueError:
                # Todo: log errors to get more information
                pass
            finally:
                if not image_got_augmented:  # we keep the image even if it contained no small objects.
                    self.save_augmented_image(target_image, coco_image['id'])
        end = datetime.now()
        duration = end-start
        print('Dataset augmentation took {} seconds'.format(duration.seconds))


def main():
    dataset = 'val2017'
    aug = DatasetAugmenter(dataset)

    

    aug.process_dataset()
    aug.save_augmented_dataset()
    interact(local=locals())


if __name__ == '__main__':
    main()
