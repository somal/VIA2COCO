import csv
import json
from datetime import datetime
from os.path import join
from typing import Tuple, List, OrderedDict, Dict, Set, Any

import cv2


def create_image_info(image_id: int, file_name: str, image_size: Tuple[int, int],
                      date_captured: datetime = datetime.utcnow().isoformat(' '),
                      license_id: int = 1, coco_url: str = "", flickr_url: str = "") -> Dict[str, Any]:
    """
    Converts input data to COCO image information storing format.

    :param int image_id: unique identificator of the image
    :param str file_name: name of the image
    :param Tuple[int, int] image_size: size of the image in format (h, w)
    :param datetime date_captured: capture date of the image
    :param int license_id: license identificator
    :param str coco_url: link to online hosted image
    :param str flickr_url: link to online hosted image
    :return: dict of the image information in COCO format
    """
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


def create_annotation(annotation_id: int, image_id: int, category_id: int, is_crowd: int, area: int,
                      bounding_box: Tuple[int, int, int, int], segmentation: List[Tuple[int, int]]) -> dict:
    """
    Converts input data to COCO annotation information storing format.

    :param int annotation_id: unique identificator of the annotation
    :param int image_id: identificator of related image
    :param int category_id: identificator of related category (annotation class)
    :param int is_crowd:
        "iscrowd": 0 if your segmentation based on polygon (object instance)
        "iscrowd": 1 if your segmentation based uncompressed RLE (crowd)
    :param float area: area occupied by segmentation in pixels
    :param Tuple[float, float, float, float] bounding_box:
        coordinates of bbox in format (x,y,w,h)
    :param list segmentation: polygon coordinates
    :return: dict of the annotation information in COCO format
    """
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }


def get_all_categories(rows: List[OrderedDict[str, str]]) -> (List[Dict[int, Any]], Set[str]):
    """
    Gets all categories from annotation file.
    :param list rows: all rows of the annotation file
    :return: all categories of the annotation file in COCO format (coco_categories)
    and category_names
    """
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


def create_output_template(coco_categories: List[Dict[int, str]]) -> Dict[str, Any]:
    """
    Creates template for annotation file in COCO format.

    :param coco_categories: categories to pu into template
    :return: dict with info, license and categories added
    """
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


def get_category_id(category: str, categories: List[Dict[int, str]]) -> int:
    """
    Gets identificator for category given.

    :param category: category to find id
    :param categories: all categories
    :return: identificator of category
    """
    category_id = 0
    for c in categories:
        if category == c['name']:
            category_id = c['id']
    return category_id


def get_area(w: int, h: int) -> int:
    """
    Calculates area occupied by bbox in pixels.

    :param w: width of bbox
    :param h: height of bbox
    :return: area of bbox
    """
    return w * h


def get_bbox(box: Dict[str, Any]) -> (int, int, int, int):
    """
    Extracts bbox coordinates in format (x, y, w, h).

    :param box: bbox information
    :return: bbox coordinates
    """
    x, y, w, h = box['x'], box['y'], box['width'], box['height']
    return x, y, w, h


def get_segmentation(box: Tuple[int]):
    """
    Calculates segmentation polygon
    from bbox coordinates.

    :param box: bbox information
    :return: tuple of points of calculated polygon
    """
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def convert(image_dir: str, annotation_path: str):
    """
    Converts annotation from VIA format (.csv) to COCO format (.json).

    :param str image_dir: directory for your images
    :param str annotation_path: path for your annotation file
    :return: coco_output is a dictionary of COCO style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    print('Reading CSV file...')
    rows = []
    with open(annotation_path, newline='') as csvfile:
        annotation_reader = csv.DictReader(csvfile)
        for row in annotation_reader:
            rows.append(row)

    # Collect all categories
    coco_categories, category_names = get_all_categories(rows)
    # Create template for output
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

        # Create and add image info into coco_output
        if filename not in files_passed:
            if len(coco_output['images']):
                image_id += 1
            image_info = create_image_info(image_id, filename, img.shape[:2])
            coco_output['images'].append(image_info)
            files_passed.update([filename])

        # Get category_id
        region_attributes = json.loads(row['region_attributes'])
        if 'type' not in region_attributes:
            continue
        assert region_attributes['type'] in category_names
        category_id = get_category_id(region_attributes['type'], coco_output['categories'])

        # Create iscrowd field
        is_crowd = 0

        # Get bbox
        box_dict = json.loads(row['region_shape_attributes'])
        x, y, w, h = get_bbox(box_dict)

        # Combine annotation
        annotation = create_annotation(annotation_id=annotation_id,
                                       image_id=image_id,
                                       category_id=category_id,
                                       is_crowd=is_crowd,
                                       area=get_area(w, h),
                                       bounding_box=(x, y, w, h),
                                       segmentation=get_segmentation((x, y, w, h)))

        # Add annotation info into coco_output
        coco_output['annotations'].append(annotation)
        annotation_id += 1
    print('Converting finished.')

    return coco_output


if __name__ == '__main__':
    IMG_FOLDER_PATH = r'C:\job\data\KKD\labelled\rcocs-1_12\3'
    ANNOTATIONS_FILE_PATH = join(IMG_FOLDER_PATH, 'lbl', 'via_annotation2.csv')

    # Convert VIA annotations to COCO annotations
    annotations = convert(image_dir=IMG_FOLDER_PATH, annotation_path=ANNOTATIONS_FILE_PATH)

    print('Saving COCO annotation into JSON file...')
    with open(join(join(IMG_FOLDER_PATH, 'lbl'), 'COCO_annotation.json'), 'w', encoding="utf-8") as outfile:
        json.dump(annotations, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    print("Finished!")
    exit(0)
