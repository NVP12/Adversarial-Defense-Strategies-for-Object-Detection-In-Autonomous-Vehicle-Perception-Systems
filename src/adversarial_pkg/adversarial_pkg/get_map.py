import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

from .utils.utils import get_classes
from .utils.utils_map import get_coco_map, get_map
from .yolo import YOLO

def evaluate_map(
    test_txt_path,
    classes_path='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/model_data/coco_classes.txt',
    annotations_folder='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/VOCdevkit/VOC2007/Annotations',
    images_folder='/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/VOCdevkit/VOC2007/JPEGImages',
    map_out_path='map_out',
    image_ext='.png',
    map_mode=0,
    MINOVERLAP=0.5,
    confidence=0.001,
    nms_iou=0.5,
    score_threshold=0.5,
    map_vis=False,
    model_weights = '/home/manas/adversarial_proj/src/adversarial_pkg/adversarial_pkg/model_data/yolo4_weights.pth'
):
    """
    Runs mAP evaluation pipeline using a given test.txt.

    Parameters
    ----------
    test_txt_path : str
        Path to the test.txt file listing image IDs.
    classes_path : str
        Path to the class names file (default: COCO).
    annotations_folder : str
        Folder containing Pascal VOC-style XML annotations.
    images_folder : str
        Folder containing test images.
    map_out_path : str
        Output directory for detection results, ground truth, etc.
    image_ext : str
        Image extension (e.g., .jpg or .png).
    map_mode : int
        Evaluation mode (0: all, 1: pred only, 2: gt only, 3: eval only, 4: COCO mAP).
    MINOVERLAP : float
        IoU threshold for mAP.
    confidence : float
        Confidence threshold to keep predictions.
    nms_iou : float
        IoU threshold for NMS.
    score_threshold : float
        Threshold to report precision/recall at a specific confidence.
    map_vis : bool
        Whether to save visualization images.
    """

    # Load image IDs
    image_ids = open(test_txt_path).read().strip().split()

    os.makedirs(map_out_path, exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'images-optional'), exist_ok=True)

    # Load class names
    class_names, _ = get_classes(classes_path)

    if map_mode in [0, 1]:
        print("Load model.")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou, model_path=model_weights)
        print("Load model done.")

        print("Get prediction results.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(images_folder, f"{image_id}{image_ext}")
            image = Image.open(image_path)

            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional", f"{image_id}{image_ext}"))

            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Prediction result generation complete.")

    if map_mode in [0, 2]:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth", f"{image_id}.txt"), "w") as new_f:
                xml_path = os.path.join(annotations_folder, f"{image_id}.xml")
                root = ET.parse(xml_path).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                    else:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
        print("Ground truth generation complete.")

    if map_mode in [0, 3]:
        print("Calculating VOC-style mAP.")
        get_map(MINOVERLAP, True, score_threhold=score_threshold, path=map_out_path)
        print("VOC mAP calculation complete.")

    if map_mode == 4:
        print("Calculating COCO-style mAP.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("COCO mAP calculation complete.")

