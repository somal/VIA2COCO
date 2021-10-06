import csv
import json
from datetime import datetime
from os.path import join
from typing import Tuple, List

import cv2


def create_image_info(image_id: int, file_name: str, image_size: Tuple[int, int],
                      date_captured: datetime = datetime.utcnow().isoformat(' '),
                      license_id: int = 1, coco_url: str = "", flickr_url: str = ""):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }


def create_annotation(annotation_id: int, image_id: int, category_id: int, is_crowd: int,
                      area: float,
                      bounding_box: Tuple[float, float, float, float], segmentation):
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }


def get_all_categories(rows: list) -> (list, set):
    print('Collecting all categories...')
    category_names = set([])
    for row in rows:
        region_attributes = row['region_attributes']
        region_attributes = json.loads(region_attributes)
        if 'type' in region_attributes:
            category_names.update([region_attributes['type']])
    coco_categories = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                       enumerate(category_names)]
    return coco_categories, category_names


def create_output_template(coco_categories: list) -> dict:
    coco_output = {
        'info': {
            "description": "Example Dataset",
            "url": "https://github.com/somal/VIA2COCO",
            "version": "0.1.0",
            "year": 2021,
            "contributor": "somal",
            "date_created": datetime.utcnow().isoformat(' ')},
        'licenses': [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "https://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        'categories': coco_categories,
        'images': [],
        'annotations': []
    }
    return coco_output


def get_category_id(category: str, categories: List[dict]) -> int:
    category_id = 0
    for c in categories:
        if category == c['name']:
            category_id = c['id']
    return category_id


def get_area(w: int, h: int) -> int:
    return w * h


def get_bbox(box: dict) -> (int, int, int, int):
    x, y, w, h = box['x'], box['y'], box['width'], box['height']
    return x, y, w, h


def get_segmentation(box: List[int]):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def convert(image_dir: str, annotation_path: str):
    """
    :param str image_dir: directory for your images
    :param str annotation_path: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    # Get the name(type) and supercategory(super_type) from VIA ANNOTATION

    print('Reading CSV file...')
    rows = []
    with open(annotation_path, newline='') as csvfile:
        annotation_reader = csv.DictReader(csvfile)
        for row in annotation_reader:
            rows.append(row)

    # Collect all categories
    coco_categories, category_names = get_all_categories(rows)
    # Template for output
    coco_output = create_output_template(coco_categories)

    print('Parsing...')
    annotation_id = 0
    image_id = 0
    files_passed = set()
    for row in rows:
        filename = row['filename']
        img = cv2.imread(join(image_dir, filename))
        if img is None:
            continue

        #
        if not len(coco_output['images']):
            # making image info and store it in coco_output['images']
            image_info = create_image_info(image_id, filename, img.shape[:2])
            coco_output['images'].append(image_info)
            files_passed.update([filename])

        if filename not in files_passed:
            image_id += 1
            image_info = create_image_info(image_id, filename, img.shape[:2])
            coco_output['images'].append(image_info)
            files_passed.update([filename])

        region_attributes = row['region_attributes']
        region_attributes = json.loads(region_attributes)
        if 'type' not in region_attributes:
            continue

        # Get category_id
        assert region_attributes['type'] in category_names
        category_id = get_category_id(region_attributes['type'], coco_output['categories'])

        # Create iscrowd
        iscrowd = 0

        # Get bbox
        box_dict = json.loads(row['region_shape_attributes'])
        box = get_bbox(box_dict)

        # Combine annotation
        annotation = create_annotation(annotation_id,
                                       image_id,
                                       category_id,
                                       iscrowd,
                                       get_area(box[2], box[3]),
                                       box,
                                       get_segmentation(box))
        coco_output['annotations'].append(annotation)
        annotation_id += 1
    print('Finished!')

    return coco_output


if __name__ == '__main__':
    IMG_FOLDER_PATH = r'C:\job\data\KKD\labelled\esaul_21\2'
    ANNOTATIONS_FILE_PATH = join(IMG_FOLDER_PATH, 'lbl', 'via_annotation2.csv')

    # Convert VIA annotations to COCO annotations
    annotations = convert(image_dir=IMG_FOLDER_PATH, annotation_path=ANNOTATIONS_FILE_PATH)

    # Save COCO annotation
    with open(join(join(IMG_FOLDER_PATH, 'lbl'), 'COCO_annotation.json'), 'w', encoding="utf-8") as outfile:
        json.dump(annotations, outfile, sort_keys=True, indent=4, ensure_ascii=False)
    exit(0)
